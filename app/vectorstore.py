"""Vector storage behind one small interface: numpy file or Postgres + pgvector."""

import os
from pathlib import Path
from typing import Protocol

import numpy as np

from app.corpus import DATA_DIR

NPZ_PATH = DATA_DIR / "vectors.npz"


class VectorStore(Protocol):
    def rebuild(self, ids: list[str], vectors: np.ndarray, model: str) -> None: ...

    def search(self, query: np.ndarray, limit: int) -> list[tuple[str, float]]: ...

    def model(self) -> str | None: ...


class NumpyStore:
    def __init__(self, path: Path = NPZ_PATH):
        self.path = path
        self._ids: list[str] = []
        self._matrix = np.zeros((0, 0), dtype=np.float32)
        self._model: str | None = None
        if path.exists():
            data = np.load(path, allow_pickle=False)
            self._ids = [str(i) for i in data["ids"]]
            self._matrix = data["vectors"].astype(np.float32)
            self._model = str(data["model"])

    def rebuild(self, ids: list[str], vectors: np.ndarray, model: str) -> None:
        np.savez_compressed(self.path, ids=np.array(ids), vectors=vectors, model=np.array(model))
        self._ids, self._matrix, self._model = list(ids), vectors, model

    def search(self, query: np.ndarray, limit: int) -> list[tuple[str, float]]:
        if not self._ids:
            return []
        scores = self._matrix @ query
        top = np.argsort(-scores)[:limit]
        return [(self._ids[i], float(scores[i])) for i in top]

    def model(self) -> str | None:
        return self._model


def _literal(vector: np.ndarray) -> str:
    return "[" + ",".join(f"{x:.6f}" for x in vector) + "]"


class PgVectorStore:
    def __init__(self, dsn: str):
        self.dsn = dsn

    def _connect(self):
        import psycopg

        return psycopg.connect(self.dsn)

    def rebuild(self, ids: list[str], vectors: np.ndarray, model: str) -> None:
        dims = vectors.shape[1]
        with self._connect() as conn:
            conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            conn.execute("DROP TABLE IF EXISTS chunk_embeddings")
            conn.execute(
                f"CREATE TABLE chunk_embeddings (id text PRIMARY KEY, model text NOT NULL, "
                f"embedding vector({dims}) NOT NULL)"
            )
            with conn.cursor() as cur:
                cur.executemany(
                    "INSERT INTO chunk_embeddings (id, model, embedding) "
                    "VALUES (%s, %s, %s::vector)",
                    [(i, model, _literal(v)) for i, v in zip(ids, vectors, strict=True)],
                )

    def search(self, query: np.ndarray, limit: int) -> list[tuple[str, float]]:
        sql = (
            "SELECT id, 1 - (embedding <=> %s::vector) FROM chunk_embeddings "
            "ORDER BY embedding <=> %s::vector LIMIT %s"
        )
        lit = _literal(query)
        with self._connect() as conn:
            rows = conn.execute(sql, (lit, lit, limit)).fetchall()
        return [(str(r[0]), float(r[1])) for r in rows]

    def model(self) -> str | None:
        try:
            with self._connect() as conn:
                row = conn.execute("SELECT model FROM chunk_embeddings LIMIT 1").fetchone()
        except Exception:
            return None
        return str(row[0]) if row else None


def make_store() -> VectorStore:
    dsn = os.environ.get("DATABASE_URL")
    if dsn:
        return PgVectorStore(dsn)
    return NumpyStore()

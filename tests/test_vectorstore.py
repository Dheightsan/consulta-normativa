import numpy as np

from app import vectorstore
from app.vectorstore import NumpyStore, PgVectorStore, make_store


def test_numpy_store_roundtrip(tmp_path):
    path = tmp_path / "v.npz"
    store = NumpyStore(path)
    assert store.search(np.array([1.0, 0.0], dtype=np.float32), 3) == []
    vectors = np.array([[1, 0], [0, 1], [0.6, 0.8]], dtype=np.float32)
    store.rebuild(["a", "b", "c"], vectors, "m1")

    again = NumpyStore(path)
    assert again.model() == "m1"
    ids = [i for i, _ in again.search(np.array([0.0, 1.0], dtype=np.float32), 2)]
    assert ids == ["b", "c"]


class FakeConn:
    def __init__(self, log):
        self.log = log

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=None):
        self.log.append((sql, params))
        return self

    def fetchall(self):
        return [("19880-27", 0.91), ("20285-14", 0.5)]

    def cursor(self):
        return self

    def executemany(self, sql, rows):
        self.log.append((sql, list(rows)))


def test_pgvector_store_sql(monkeypatch):
    log = []
    store = PgVectorStore("postgresql://test")
    monkeypatch.setattr(store, "_connect", lambda: FakeConn(log))

    store.rebuild(["a"], np.array([[0.5, 0.5]], dtype=np.float32), "m")
    statements = " ".join(sql for sql, _ in log)
    assert "CREATE EXTENSION IF NOT EXISTS vector" in statements
    assert "vector(2)" in statements
    assert log[-1][1] == [("a", "m", "[0.500000,0.500000]")]

    found = store.search(np.array([1.0, 0.0], dtype=np.float32), 2)
    assert found == [("19880-27", 0.91), ("20285-14", 0.5)]
    assert "<=>" in log[-1][0]


def test_make_store_picks_backend(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    assert isinstance(make_store(), vectorstore.NumpyStore)
    monkeypatch.setenv("DATABASE_URL", "postgresql://test")
    assert isinstance(make_store(), PgVectorStore)

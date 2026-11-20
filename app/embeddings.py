"""Gemini embeddings over plain HTTP."""

import os
import time

import httpx
import numpy as np

BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
MODEL = "gemini-embedding-2"
DIMS = 768
BATCH = 8


def _unit(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return (matrix / np.where(norms == 0, 1, norms)).astype(np.float32)


class GeminiEmbedder:
    def __init__(
        self,
        api_key: str,
        model: str = MODEL,
        dims: int = DIMS,
        timeout: float = 30.0,
        transport: httpx.BaseTransport | None = None,
    ):
        self.api_key = api_key
        self.model = model
        self.dims = dims
        self.timeout = timeout
        self.transport = transport
        self._cache: dict[str, np.ndarray] = {}

    @classmethod
    def from_env(cls) -> "GeminiEmbedder | None":
        key = os.environ.get("GEMINI_API_KEY")
        if not key:
            return None
        return cls(key, os.environ.get("EMBEDDING_MODEL", MODEL))

    def _embed(self, texts: list[str]) -> np.ndarray:
        requests = [
            {
                "model": f"models/{self.model}",
                "content": {"parts": [{"text": t}]},
                "output_dimensionality": self.dims,
            }
            for t in texts
        ]
        url = f"{BASE_URL}/models/{self.model}:batchEmbedContents"
        with httpx.Client(timeout=self.timeout, transport=self.transport) as http:
            for attempt in range(6):
                resp = http.post(
                    url, headers={"x-goog-api-key": self.api_key}, json={"requests": requests}
                )
                if resp.status_code != 429 or attempt == 5:
                    break
                time.sleep(10 * (attempt + 1))
            resp.raise_for_status()
            rows = [e["values"] for e in resp.json()["embeddings"]]
        return np.array(rows, dtype=np.float32)

    def embed_documents(self, docs: list[tuple[str, str]]) -> np.ndarray:
        texts = [f"title: {title} | text: {body[:6000]}" for title, body in docs]
        parts = [self._embed(texts[i : i + BATCH]) for i in range(0, len(texts), BATCH)]
        return _unit(np.vstack(parts))

    def embed_query(self, question: str) -> np.ndarray:
        if question not in self._cache:
            text = f"task: question answering | query: {question}"
            if len(self._cache) > 500:
                self._cache.clear()
            self._cache[question] = _unit(self._embed([text]))[0]
        return self._cache[question]

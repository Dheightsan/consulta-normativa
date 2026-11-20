import json

import httpx
import numpy as np

from app import embeddings
from app.embeddings import GeminiEmbedder


def make(handler) -> GeminiEmbedder:
    return GeminiEmbedder("test-key", dims=3, transport=httpx.MockTransport(handler))


def ok(request: httpx.Request) -> httpx.Response:
    n = len(json.loads(request.content)["requests"])
    return httpx.Response(200, json={"embeddings": [{"values": [3.0, 4.0, 0.0]}] * n})


def test_query_is_prefixed_and_normalized():
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        seen["key"] = request.headers["x-goog-api-key"]
        return ok(request)

    vec = make(handler).embed_query("hola")
    assert np.allclose(vec, [0.6, 0.8, 0.0])
    req = seen["body"]["requests"][0]
    assert req["content"]["parts"][0]["text"] == "task: question answering | query: hola"
    assert req["output_dimensionality"] == 3
    assert seen["key"] == "test-key"


def test_documents_are_batched():
    calls = []

    def handler(request):
        calls.append(len(json.loads(request.content)["requests"]))
        return ok(request)

    docs = [("Ley 1, art. 1", "texto")] * (embeddings.BATCH + 2)
    out = make(handler).embed_documents(docs)
    assert out.shape == (embeddings.BATCH + 2, 3)
    assert calls == [embeddings.BATCH, 2]


def test_rate_limit_is_retried(monkeypatch):
    monkeypatch.setattr(embeddings.time, "sleep", lambda s: None)
    count = {"n": 0}

    def handler(request):
        count["n"] += 1
        return httpx.Response(429) if count["n"] < 3 else ok(request)

    assert make(handler).embed_query("hola").shape == (3,)
    assert count["n"] == 3


def test_from_env(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    assert GeminiEmbedder.from_env() is None
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    assert GeminiEmbedder.from_env() is not None

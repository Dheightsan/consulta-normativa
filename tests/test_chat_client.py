import json

import httpx

from app.chat_client import SYSTEM_MESSAGE, ChatClient
from app.guard import NOT_FOUND


def make_client(handler) -> ChatClient:
    return ChatClient(
        "http://models.test/v1/", "test-key", "test-model", transport=httpx.MockTransport(handler)
    )


def test_request_is_built_correctly(chunks):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers["authorization"]
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"choices": [{"message": {"content": " ok "}}]})

    assert make_client(handler).complete("cuánto demora", chunks[:1]) == "ok"
    assert seen["url"] == "http://models.test/v1/chat/completions"
    assert seen["auth"] == "Bearer test-key"
    body = seen["body"]
    assert body["model"] == "test-model"
    assert body["messages"][0] == {"role": "system", "content": SYSTEM_MESSAGE}
    assert NOT_FOUND in SYSTEM_MESSAGE
    user = body["messages"][1]["content"]
    assert "[Ley 20.285, art. 14]" in user
    assert "cuánto demora" in user


def test_server_error_returns_none(chunks):
    assert make_client(lambda r: httpx.Response(500)).complete("hola", chunks) is None


def test_bad_body_returns_none(chunks):
    client = make_client(lambda r: httpx.Response(200, json={"x": 1}))
    assert client.complete("hola", chunks) is None


def test_from_env_needs_key_and_base_url(monkeypatch):
    monkeypatch.delenv("MODEL_API_KEY", raising=False)
    monkeypatch.setenv("MODEL_BASE_URL", "http://models.test/v1")
    assert ChatClient.from_env() is None
    monkeypatch.setenv("MODEL_API_KEY", "k")
    monkeypatch.setenv("MODEL_NAME", "m")
    client = ChatClient.from_env()
    assert client is not None and client.model == "m"

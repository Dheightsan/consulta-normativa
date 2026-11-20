"""Client for an OpenAI-compatible chat completions endpoint."""

import os

import httpx

from app.corpus import Chunk
from app.guard import NOT_FOUND

SYSTEM_MESSAGE = (
    "Eres un asistente que responde preguntas sobre normativa pública chilena. "
    "Responde en español, de forma breve, usando solo los fragmentos entregados. "
    "Cada afirmación debe llevar su cita con el formato exacto 'Ley 20.285, art. 14'. "
    "No uses conocimiento externo ni agregues plazos, montos o números que no estén "
    "en los fragmentos. Si los fragmentos no responden la pregunta, responde "
    f"exactamente: {NOT_FOUND}"
)


class ChatClient:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        timeout: float = 30.0,
        transport: httpx.BaseTransport | None = None,
    ):
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.transport = transport

    @classmethod
    def from_env(cls) -> "ChatClient | None":
        api_key = os.environ.get("MODEL_API_KEY")
        base_url = os.environ.get("MODEL_BASE_URL")
        if not api_key or not base_url:
            return None
        return cls(base_url, api_key, os.environ.get("MODEL_NAME", "default"))

    def build_payload(self, question: str, chunks: list[Chunk]) -> dict:
        listing = "\n\n".join(f"[{c.ref}]\n{c.texto}" for c in chunks)
        return {
            "model": self.model,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": SYSTEM_MESSAGE},
                {"role": "user", "content": f"Fragmentos:\n\n{listing}\n\nPregunta: {question}"},
            ],
        }

    def complete(self, question: str, chunks: list[Chunk]) -> str | None:
        """Model text, or None on any failure so the caller can fall back."""
        try:
            with httpx.Client(timeout=self.timeout, transport=self.transport) as http:
                resp = http.post(
                    self.url,
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json=self.build_payload(question, chunks),
                )
                resp.raise_for_status()
                text = resp.json()["choices"][0]["message"]["content"]
        except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError):
            return None
        return text.strip() if isinstance(text, str) and text.strip() else None

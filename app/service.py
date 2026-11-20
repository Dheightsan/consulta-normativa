"""Answer flow: retrieve, ask the model if there is one, check, fall back."""

import re
from typing import Protocol

from pydantic import BaseModel

from app.corpus import Chunk
from app.guard import NOT_FOUND, check
from app.retrieval import Hit, HybridSearch, tokenize


class Model(Protocol):
    def complete(self, question: str, chunks: list[Chunk]) -> str | None: ...


class Cita(BaseModel):
    ley: str
    articulo: str
    extracto: str


class Respuesta(BaseModel):
    respuesta: str
    citas: list[Cita]
    fundamentada: bool
    modo: str


def extract(chunk: Chunk, question: str, limit: int = 380) -> str:
    """The paragraph of the article that best overlaps the question."""
    terms = set(tokenize(question))
    paragraphs = [p for p in re.split(r"\n\n", chunk.texto) if p.strip()]
    best = max(paragraphs, key=lambda p: len(terms & set(tokenize(p))), default=chunk.texto)
    if len(best) <= limit:
        return best
    return best[:limit].rsplit(" ", 1)[0] + "…"


def cita_for(chunk: Chunk, question: str) -> Cita:
    return Cita(ley=chunk.ley, articulo=chunk.numero, extracto=extract(chunk, question))


def _extractive(hit: Hit, question: str) -> Respuesta:
    cita = cita_for(hit.chunk, question)
    return Respuesta(
        respuesta=f"{hit.chunk.ref}: {cita.extracto}",
        citas=[cita],
        fundamentada=True,
        modo="extractiva",
    )


def answer_question(question: str, search: HybridSearch, model: Model | None) -> Respuesta:
    hits = search.search(question)
    if not hits:
        return Respuesta(respuesta=NOT_FOUND, citas=[], fundamentada=False, modo="sin_resultado")

    chunks = [h.chunk for h in hits]
    text = model.complete(question, chunks) if model else None
    if text is None:
        return _extractive(hits[0], question)
    if text.strip() == NOT_FOUND:
        return Respuesta(respuesta=NOT_FOUND, citas=[], fundamentada=False, modo="generativa")

    result = check(text, chunks)
    if not result.ok:
        return _extractive(hits[0], question)
    by_key = {(c.ley, c.numero): c for c in chunks}
    citas = [cita_for(by_key[key], question) for key in result.cited]
    return Respuesta(respuesta=text, citas=citas, fundamentada=True, modo="generativa")

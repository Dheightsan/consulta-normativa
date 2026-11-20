from functools import lru_cache
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from app.chat_client import ChatClient
from app.corpus import Chunk, load_chunks
from app.embeddings import GeminiEmbedder
from app.retrieval import HybridSearch
from app.service import Model, Respuesta, answer_question
from app.vectorstore import make_store

app = FastAPI(title="consulta-normativa")
INDEX = Path(__file__).parent / "static" / "index.html"


class PreguntaIn(BaseModel):
    pregunta: str = Field(min_length=1, max_length=500)


@lru_cache
def get_search() -> HybridSearch:
    embedder = GeminiEmbedder.from_env()
    store = make_store() if embedder else None
    return HybridSearch(load_chunks(), embedder, store)


def get_model() -> Model | None:
    return ChatClient.from_env()


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    return INDEX.read_text(encoding="utf-8")


@app.get("/health")
def health(search: HybridSearch = Depends(get_search)) -> dict:
    return {
        "status": "ok",
        "articulos": len(search.chunks),
        "busqueda": "hibrida" if search.has_vectors() else "bm25",
        "generacion": "modelo" if ChatClient.from_env() else "extractiva",
    }


@app.post("/preguntar")
def preguntar(
    body: PreguntaIn,
    search: HybridSearch = Depends(get_search),
    model: Model | None = Depends(get_model),
) -> Respuesta:
    return answer_question(body.pregunta, search, model)


@app.get("/articulo/{ley}/{numero}")
def articulo(ley: str, numero: str, search: HybridSearch = Depends(get_search)) -> Chunk:
    chunk = search.get(ley, numero)
    if chunk is None:
        raise HTTPException(status_code=404, detail="Artículo no encontrado")
    return chunk

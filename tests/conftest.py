import numpy as np
import pytest

from app.corpus import Chunk

CHUNKS = [
    Chunk(
        id="20285-14",
        ley="20.285",
        ley_nombre="Transparencia",
        numero="14",
        titulo="TITULO IV",
        texto=(
            "La autoridad deberá pronunciarse sobre la solicitud de acceso a la información "
            "en un plazo máximo de veinte días hábiles, contado desde la recepción.\n\n"
            "Este plazo podrá ser prorrogado por otros diez días hábiles."
        ),
    ),
    Chunk(
        id="19880-27",
        ley="19.880",
        ley_nombre="Procedimiento Administrativo",
        numero="27",
        titulo="CAPITULO II",
        texto=(
            "El procedimiento administrativo no podrá exceder de 6 meses, "
            "desde su iniciación hasta la fecha en que se emita la decisión final."
        ),
    ),
    Chunk(
        id="19886-8",
        ley="19.886",
        ley_nombre="Compras Públicas",
        numero="8",
        titulo="Párrafo 1",
        texto=(
            "Procederá la licitación privada si en las licitaciones públicas no se "
            "hubieren presentado interesados."
        ),
    ),
    Chunk(
        id="19886-8-bis",
        ley="19.886",
        ley_nombre="Compras Públicas",
        numero="8 bis",
        titulo="Párrafo 1",
        texto="Las bases de licitación deberán publicarse en el sistema de información.",
    ),
]


class FakeEmbedder:
    """Always returns the same query vector."""

    def embed_query(self, question: str) -> np.ndarray:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)


class FakeModel:
    def __init__(self, reply: str | None):
        self.reply = reply

    def complete(self, question, chunks):
        return self.reply


@pytest.fixture
def chunks():
    return list(CHUNKS)

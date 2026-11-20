"""Article chunks of the loaded laws."""

import json
from pathlib import Path

from pydantic import BaseModel

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


class Chunk(BaseModel):
    id: str
    ley: str
    ley_nombre: str
    numero: str
    titulo: str
    texto: str

    @property
    def ref(self) -> str:
        return f"Ley {self.ley}, art. {self.numero}"


def chunk_id(ley: str, numero: str) -> str:
    return f"{ley.replace('.', '')}-{numero.lower().replace(' ', '-')}"


def normalize_numero(raw: str) -> str:
    """'14', '14 bis', '8-bis', '8°' all map to '14', '14 bis', '8 bis', '8'."""
    text = raw.strip().lower().replace("-", " ").replace("°", "").replace("º", "")
    text = text.replace("quater", "quáter")
    return " ".join(text.split())


def load_chunks(path: Path | None = None) -> list[Chunk]:
    path = path or DATA_DIR / "chunks.json"
    with open(path, encoding="utf-8") as fh:
        return [Chunk(**row) for row in json.load(fh)]

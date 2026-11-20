"""Parse the LeyChile XML files into one chunk per article, then index them.

    python -m app.ingest
"""

import json
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

from app.corpus import DATA_DIR, Chunk, chunk_id, normalize_numero
from app.embeddings import GeminiEmbedder
from app.vectorstore import make_store

NS = {"n": "http://www.leychile.cl/esquemas"}
ARTICLE = re.compile(
    r"^\s*Art[ií]culo\s+(\d+)\s*[°º]?\s*(bis|ter|qu[aá]ter|quinquies|sexies)?\s*[.\-]",
    re.IGNORECASE,
)
SECTIONS = {"Título", "Párrafo", "Capítulo"}


def clean(text: str) -> str:
    """Join hard-wrapped lines, keep blank lines as paragraph breaks."""
    text = text.replace("\xa0", " ")
    paragraphs = re.split(r"\n\s*\n", text)
    lines = [" ".join(p.split()) for p in paragraphs]
    return "\n\n".join(p for p in lines if p)


def part_title(node: ET.Element) -> str:
    meta = node.find("n:Metadatos", NS)
    if meta is None:
        return ""
    title = meta.find("n:TituloParte", NS)
    text = (title.text or "").replace("\xa0", " ").strip() if title is not None else ""
    return " ".join(text.split())


def walk(node: ET.Element, trail: list[str], found: list[tuple[str, str, str]]) -> None:
    for child in node.findall("n:EstructurasFuncionales/n:EstructuraFuncional", NS):
        kind = child.get("tipoParte", "")
        if kind in SECTIONS:
            title = part_title(child)
            walk(child, [*trail, title] if title else trail, found)
            continue
        if kind == "Artículo" and child.get("transitorio") == "no transitorio":
            text = child.findtext("n:Texto", default="", namespaces=NS)
            m = ARTICLE.match(text)
            if m:
                numero = normalize_numero(" ".join(g for g in m.groups() if g))
                body = clean(text[m.end() :]).lstrip("-. ")
                found.append((numero, " / ".join(trail[-2:]), body))
        walk(child, trail, found)


def parse_law(meta: dict, folder: Path = DATA_DIR) -> list[Chunk]:
    root = ET.parse(folder / meta["archivo"]).getroot()
    found: list[tuple[str, str, str]] = []
    walk(root, [], found)
    chunks: dict[str, Chunk] = {}
    for numero, titulo, body in found:
        cid = chunk_id(meta["ley"], numero)
        if cid in chunks or not body:
            continue
        chunks[cid] = Chunk(
            id=cid,
            ley=meta["ley"],
            ley_nombre=meta["nombre"],
            numero=numero,
            titulo=titulo,
            texto=body,
        )
    return list(chunks.values())


def main() -> int:
    with open(DATA_DIR / "fuentes.json", encoding="utf-8") as fh:
        laws = json.load(fh)["leyes"]
    chunks: list[Chunk] = []
    for meta in laws:
        parsed = parse_law(meta)
        print(f"Ley {meta['ley']}: {len(parsed)} articulos")
        chunks.extend(parsed)
    with open(DATA_DIR / "chunks.json", "w", encoding="utf-8", newline="\n") as fh:
        json.dump([c.model_dump() for c in chunks], fh, ensure_ascii=False, indent=1)
        fh.write("\n")
    print(f"chunks.json: {len(chunks)} fragmentos")

    embedder = GeminiEmbedder.from_env()
    if embedder is None:
        print("Sin GEMINI_API_KEY: se omiten los embeddings (solo BM25).")
        return 0
    vectors = embedder.embed_documents([(c.ref, c.texto) for c in chunks])
    store = make_store()
    store.rebuild([c.id for c in chunks], vectors, embedder.model)
    print(f"embeddings: {len(chunks)} x {vectors.shape[1]} en {type(store).__name__}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

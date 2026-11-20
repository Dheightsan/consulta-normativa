"""Measure retrieval and answers over eval/preguntas.jsonl.

    python -m eval.run
"""

import json
import sys
from pathlib import Path

from app.chat_client import ChatClient
from app.corpus import load_chunks
from app.embeddings import GeminiEmbedder
from app.retrieval import HybridSearch
from app.service import answer_question
from app.vectorstore import make_store

QUESTIONS = Path(__file__).parent / "preguntas.jsonl"


def load_questions() -> list[dict]:
    with open(QUESTIONS, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def run_mode(name: str, search: HybridSearch, model, questions: list[dict]) -> dict:
    inside = [q for q in questions if q["esperado"]]
    outside = [q for q in questions if not q["esperado"]]
    top3 = cited = grounded = rejected = 0
    misses: list[str] = []
    for q in inside:
        hits = search.search(q["pregunta"], limit=3)
        refs = {f"{h.chunk.ley} {h.chunk.numero}" for h in hits}
        if refs & set(q["esperado"]):
            top3 += 1
        else:
            misses.append(f"recuperacion: {q['pregunta']} -> {sorted(refs)}")
        res = answer_question(q["pregunta"], search, model)
        grounded += res.fundamentada
        got = {f"{c.ley} {c.articulo}" for c in res.citas}
        if got & set(q["esperado"]):
            cited += 1
    for q in outside:
        res = answer_question(q["pregunta"], search, model)
        if not res.fundamentada:
            rejected += 1
        else:
            misses.append(f"no rechazo: {q['pregunta']}")
    return {
        "modo": name,
        "top3": f"{top3}/{len(inside)}",
        "fundamentadas": f"{grounded}/{len(inside)}",
        "cita_correcta": f"{cited}/{len(inside)}",
        "rechazos": f"{rejected}/{len(outside)}",
        "fallos": misses,
    }


def main() -> int:
    chunks = load_chunks()
    questions = load_questions()
    results = [run_mode("BM25", HybridSearch(chunks), None, questions)]
    embedder = GeminiEmbedder.from_env()
    if embedder:
        hybrid = HybridSearch(chunks, embedder, make_store())
        results.append(run_mode("Híbrido", hybrid, None, questions))
        model = ChatClient.from_env()
        if model:
            results.append(run_mode(f"Híbrido + {model.model}", hybrid, model, questions))
        if not hybrid.vectors_used:
            print("aviso: la ultima consulta vectorial fallo, el modo hibrido cayo a BM25")
    else:
        print("Sin GEMINI_API_KEY: solo BM25.")

    print("| Modo | Artículo en top-3 | Fundamentadas | Cita correcta | Rechazos correctos |")
    print("|---|---|---|---|---|")
    for r in results:
        print(
            f"| {r['modo']} | {r['top3']} | {r['fundamentadas']} | "
            f"{r['cita_correcta']} | {r['rechazos']} |"
        )
    for r in results:
        for line in r["fallos"]:
            print(f"[{r['modo']}] {line}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

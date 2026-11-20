"""BM25 over the articles plus a hybrid search that fuses it with embeddings."""

import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from app.corpus import Chunk, normalize_numero

STOPWORDS = {
    "a", "al", "algo", "con", "cual", "cuales", "cuanto", "cuantos", "cuanta", "cuantas",
    "de", "del", "el", "ella", "en", "es", "esta", "estan", "este", "hay", "la", "las", "lo",
    "los", "me", "mi", "para", "por", "que", "se", "si", "son", "su", "sus", "tu", "un",
    "una", "y", "yo", "como", "donde", "cuando", "o", "e", "u", "ley", "articulo", "ser",
    "debe", "pueden", "puede", "tiene", "tienen", "ha", "han", "sobre", "segun", "quien",
    "quienes", "entre", "desde", "hasta", "sin", "ni", "no", "mas", "muy", "otro", "otra",
}  # fmt: skip

RRF_K = 60
MIN_BM25 = 5.0
MIN_COSINE = 0.70
VECTOR_WEIGHT = 1.0
LAW_MENTION = re.compile(r"ley\s*(?:n[°º.]*\s*)?(\d{2}\.?\d{3})", re.IGNORECASE)
ARTICLE_MENTION = re.compile(
    r"art(?:[ií]culo|\.)?s?\s*(\d+)\s*[°º]?\s*(bis|ter|qu[aá]ter|quinquies|sexies)?",
    re.IGNORECASE,
)


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in text if not unicodedata.combining(c))


def stem(word: str) -> str:
    if len(word) > 6 and word.endswith("ciones"):
        return word[:-6] + "cion"
    if len(word) > 6 and word.endswith("mente"):
        return word[:-5]
    if len(word) > 5 and word.endswith("es"):
        return word[:-2]
    if len(word) > 4 and word.endswith("s"):
        return word[:-1]
    return word


def tokenize(text: str) -> list[str]:
    words = re.findall(r"[a-z0-9]+", normalize(text))
    return [stem(w) for w in words if w not in STOPWORDS]


class BM25:
    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks
        self.k1, self.b = k1, b
        self.docs = [Counter(tokenize(f"{c.titulo} {c.texto}")) for c in chunks]
        self.lengths = [sum(d.values()) for d in self.docs]
        self.avg_len = (sum(self.lengths) / len(self.lengths)) if self.lengths else 0.0
        self.df: Counter[str] = Counter()
        for doc in self.docs:
            self.df.update(doc.keys())

    def _idf(self, term: str) -> float:
        n, df = len(self.docs), self.df.get(term, 0)
        return math.log(1 + (n - df + 0.5) / (df + 0.5))

    def scores(self, query: str) -> list[float]:
        terms = tokenize(query)
        out = []
        for doc, length in zip(self.docs, self.lengths, strict=True):
            score = 0.0
            for term in terms:
                tf = doc.get(term, 0)
                if tf:
                    norm = tf + self.k1 * (1 - self.b + self.b * length / self.avg_len)
                    score += self._idf(term) * tf * (self.k1 + 1) / norm
            out.append(score)
        return out


class Embedder(Protocol):
    def embed_query(self, question: str) -> np.ndarray: ...


class Store(Protocol):
    def search(self, query: np.ndarray, limit: int) -> list[tuple[str, float]]: ...


@dataclass
class Hit:
    chunk: Chunk
    score: float
    bm25: float
    cosine: float | None


class HybridSearch:
    def __init__(
        self,
        chunks: list[Chunk],
        embedder: Embedder | None = None,
        store=None,
        min_bm25: float = MIN_BM25,
        min_cosine: float = MIN_COSINE,
    ):
        self.chunks = chunks
        self.by_id = {c.id: c for c in chunks}
        self.bm25 = BM25(chunks)
        self.embedder = embedder
        self.store: Store | None = store
        self.min_bm25 = min_bm25
        self.min_cosine = min_cosine
        self.vectors_used = False

    def get(self, ley: str, numero: str) -> Chunk | None:
        ley = ley if "." in ley else f"{ley[:-3]}.{ley[-3:]}"
        wanted = normalize_numero(numero)
        for c in self.chunks:
            if c.ley == ley and c.numero == wanted:
                return c
        return None

    def _vector_hits(self, question: str, limit: int) -> dict[str, float]:
        self.vectors_used = False
        if self.embedder is None or self.store is None:
            return {}
        try:
            found = self.store.search(self.embedder.embed_query(question), limit)
        except Exception:
            return {}
        self.vectors_used = True
        return dict(found)

    def _mentioned(self, question: str) -> tuple[set[str], set[str]]:
        laws = {m.replace(".", "") for m in LAW_MENTION.findall(question)}
        found = ARTICLE_MENTION.findall(question)
        arts = {normalize_numero(" ".join(g for g in m if g)) for m in found}
        return laws, arts

    def search(self, question: str, limit: int = 3) -> list[Hit]:
        laws, arts = self._mentioned(question)
        pool = [i for i, c in enumerate(self.chunks) if not laws or c.ley.replace(".", "") in laws]
        if not pool:
            return []
        bm = self.bm25.scores(question)
        cos = self._vector_hits(question, 25)

        bm_rank = sorted(pool, key=lambda i: bm[i], reverse=True)[:25]
        fused: dict[str, float] = {}
        for rank, i in enumerate(r for r in bm_rank if bm[r] > 0):
            fused[self.chunks[i].id] = fused.get(self.chunks[i].id, 0) + 1 / (RRF_K + rank)
        allowed = {self.chunks[i].id for i in pool}
        for rank, cid in enumerate(c for c in cos if c in allowed):
            fused[cid] = fused.get(cid, 0) + VECTOR_WEIGHT / (RRF_K + rank)

        index = {c.id: i for i, c in enumerate(self.chunks)}
        best_bm = max((bm[i] for i in pool), default=0.0)
        best_cos = max((cos[c] for c in cos if c in allowed), default=0.0)
        direct = [c for c in self.chunks if laws and arts and c.numero in arts and c.id in allowed]
        # with vectors, cosine decides relevance; bm25 scores overlap too much
        if self.vectors_used:
            relevant = best_cos >= self.min_cosine
        else:
            relevant = best_bm >= self.min_bm25
        if not direct and not relevant:
            return []
        for c in direct:
            fused[c.id] = fused.get(c.id, 0) + 1.0

        ranked = sorted(fused, key=lambda cid: fused[cid], reverse=True)[:limit]
        return [Hit(self.by_id[cid], fused[cid], bm[index[cid]], cos.get(cid)) for cid in ranked]

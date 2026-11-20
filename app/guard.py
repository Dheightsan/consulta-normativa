"""Check that citations and numbers in an answer exist in the retrieved articles."""

import re
from dataclasses import dataclass

from app.corpus import Chunk, normalize_numero
from app.retrieval import normalize

NOT_FOUND = "No encontré eso en los documentos cargados."

SUFFIX = r"(?:bis|ter|qu[aá]ter|quinquies|sexies)"
CITATION = re.compile(
    rf"Ley\s+(?:N[°º.]*\s*)?(\d{{1,2}}\.?\d{{3}}),?\s+art(?:[ií]culos?|\.)\s*(\d+)\s*[°º]?\s*({SUFFIX})?",
    re.IGNORECASE,
)
BARE_ARTICLE = re.compile(rf"art(?:[ií]culos?|\.)\s*(\d+)\s*[°º]?\s*({SUFFIX})?", re.IGNORECASE)
LAW_MENTION = re.compile(r"Ley\s+(?:N[°º.]*\s*)?\d{1,2}\.?\d{3}", re.IGNORECASE)
NUMBER = re.compile(r"\d+(?:[.,]\d+)*")

WORD_NUMBERS = {
    "un": 1, "uno": 1, "una": 1, "dos": 2, "tres": 3, "cuatro": 4, "cinco": 5, "seis": 6,
    "siete": 7, "ocho": 8, "nueve": 9, "diez": 10, "once": 11, "doce": 12, "trece": 13,
    "catorce": 14, "quince": 15, "veinte": 20, "treinta": 30, "cuarenta": 40,
    "cincuenta": 50, "sesenta": 60, "noventa": 90, "cien": 100, "ciento": 100,
}  # fmt: skip


@dataclass
class GuardResult:
    ok: bool
    cited: list[tuple[str, str]]


def _digits(token: str) -> str:
    return re.sub(r"[.,]", "", token)


def _ley(raw: str) -> str:
    raw = raw.replace(".", "")
    return f"{raw[:-3]}.{raw[-3:]}"


def known_numbers(chunks: list[Chunk]) -> set[str]:
    source = " ".join(c.texto for c in chunks)
    nums = {_digits(n) for n in NUMBER.findall(source)}
    for word in re.findall(r"[a-z]+", normalize(source)):
        if word in WORD_NUMBERS:
            nums.add(str(WORD_NUMBERS[word]))
    for c in chunks:
        nums.update(re.findall(r"\d+", c.numero))
        nums.add(c.ley.replace(".", ""))
    return nums


def check(answer: str, chunks: list[Chunk]) -> GuardResult:
    valid = {(c.ley, c.numero) for c in chunks}
    numeros = {c.numero for c in chunks}

    cited = []
    for ley, num, suffix in CITATION.findall(answer):
        pair = (_ley(ley), normalize_numero(f"{num} {suffix}".strip()))
        if pair not in valid:
            return GuardResult(False, [])
        cited.append(pair)
    if not cited:
        return GuardResult(False, [])

    known = known_numbers(chunks)
    for num, suffix in BARE_ARTICLE.findall(answer):
        if normalize_numero(f"{num} {suffix}".strip()) not in numeros and num not in known:
            return GuardResult(False, [])

    stripped = BARE_ARTICLE.sub(" ", LAW_MENTION.sub(" ", answer))
    if any(_digits(n) not in known for n in NUMBER.findall(stripped)):
        return GuardResult(False, [])
    return GuardResult(True, list(dict.fromkeys(cited)))

from app.retrieval import HybridSearch, tokenize
from tests.conftest import FakeEmbedder


class FakeStore:
    def __init__(self, ranking):
        self.ranking = ranking

    def search(self, query, limit):
        return self.ranking[:limit]


def test_tokenize_drops_stopwords_and_accents():
    assert tokenize("¿Cuáles son los plazos de las reclamaciones?") == ["plazo", "reclamacion"]


def test_bm25_finds_the_right_article(chunks):
    question = "en cuántos días hábiles debe pronunciarse la autoridad"
    hits = HybridSearch(chunks, min_bm25=0.5).search(question)
    assert hits[0].chunk.id == "20285-14"


def test_unrelated_question_returns_nothing(chunks):
    assert HybridSearch(chunks, min_bm25=0.5).search("receta de empanadas de pino") == []


def test_law_mention_restricts_the_search(chunks):
    question = "hasta cuánto puede exceder el procedimiento según la ley 19.880"
    hits = HybridSearch(chunks, min_bm25=0.5).search(question)
    assert {h.chunk.ley for h in hits} == {"19.880"}


def test_explicit_article_reference_comes_first(chunks):
    hits = HybridSearch(chunks, min_bm25=0.5).search("qué dice el artículo 8 bis de la ley 19886")
    assert hits[0].chunk.numero == "8 bis"


def test_get_accepts_both_law_formats(chunks):
    search = HybridSearch(chunks, min_bm25=0.5)
    assert search.get("20285", "14").id == "20285-14"
    assert search.get("19.886", "8-bis").id == "19886-8-bis"
    assert search.get("20285", "99") is None


def test_hybrid_uses_vectors_when_bm25_has_nothing(chunks):
    store = FakeStore([("19880-27", 0.9), ("20285-14", 0.5)])
    search = HybridSearch(chunks, FakeEmbedder(), store)
    hits = search.search("receta de empanadas de pino")
    assert hits[0].chunk.id == "19880-27"
    assert hits[0].cosine == 0.9
    assert search.vectors_used


def test_low_cosine_is_rejected(chunks):
    store = FakeStore([("19880-27", 0.3)])
    assert HybridSearch(chunks, FakeEmbedder(), store).search("receta de empanadas") == []


def test_vector_failure_falls_back_to_bm25(chunks):
    class Broken:
        def search(self, query, limit):
            raise RuntimeError("down")

    search = HybridSearch(chunks, FakeEmbedder(), Broken(), min_bm25=0.5)
    hits = search.search("licitación privada si no hubo interesados")
    assert hits[0].chunk.id == "19886-8"
    assert not search.vectors_used

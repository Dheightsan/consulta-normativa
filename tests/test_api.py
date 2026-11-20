import pytest
from fastapi.testclient import TestClient

from app.guard import NOT_FOUND
from app.main import app, get_model, get_search
from app.retrieval import HybridSearch
from tests.conftest import CHUNKS, FakeModel

QUESTION = "en cuántos días hábiles debe pronunciarse la autoridad"


@pytest.fixture
def client():
    app.dependency_overrides[get_search] = lambda: HybridSearch(list(CHUNKS), min_bm25=0.5)
    app.dependency_overrides[get_model] = lambda: None
    yield TestClient(app)
    app.dependency_overrides.clear()


def use_model(reply):
    app.dependency_overrides[get_model] = lambda: FakeModel(reply)


def ask(client, question):
    return client.post("/preguntar", json={"pregunta": question}).json()


def test_health(client):
    data = client.get("/health").json()
    assert data["status"] == "ok"
    assert data["articulos"] == len(CHUNKS)
    assert data["busqueda"] == "bm25"


def test_home_page(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "/preguntar" in r.text


def test_unknown_question_is_not_found(client):
    data = ask(client, "receta de empanadas de pino")
    assert data == {
        "respuesta": NOT_FOUND,
        "citas": [],
        "fundamentada": False,
        "modo": "sin_resultado",
    }


def test_extractive_answer_cites_the_article(client):
    data = ask(client, QUESTION)
    assert data["fundamentada"] is True
    assert data["modo"] == "extractiva"
    assert data["citas"][0]["ley"] == "20.285"
    assert data["citas"][0]["articulo"] == "14"
    assert data["respuesta"].startswith("Ley 20.285, art. 14:")


def test_model_answer_with_valid_citation(client):
    use_model("Tiene veinte días hábiles (Ley 20.285, art. 14).")
    data = ask(client, QUESTION)
    assert data["modo"] == "generativa"
    assert [c["articulo"] for c in data["citas"]] == ["14"]


def test_model_that_invents_a_number_falls_back_to_extract(client):
    use_model("Tiene 45 días hábiles (Ley 20.285, art. 14).")
    data = ask(client, QUESTION)
    assert data["modo"] == "extractiva"
    assert "45" not in data["respuesta"]


def test_model_that_cites_a_missing_article_falls_back(client):
    use_model("Tiene veinte días hábiles (Ley 20.285, art. 99).")
    assert ask(client, QUESTION)["modo"] == "extractiva"


def test_model_says_not_found(client):
    use_model(NOT_FOUND)
    data = ask(client, QUESTION)
    assert data["respuesta"] == NOT_FOUND
    assert data["fundamentada"] is False


def test_model_failure_falls_back_to_extract(client):
    use_model(None)
    assert ask(client, QUESTION)["modo"] == "extractiva"


def test_empty_question_is_rejected(client):
    assert client.post("/preguntar", json={"pregunta": ""}).status_code == 422


def test_article_lookup(client):
    r = client.get("/articulo/19886/8 bis")
    assert r.status_code == 200
    assert r.json()["numero"] == "8 bis"
    assert client.get("/articulo/20.285/14").json()["ley"] == "20.285"
    assert client.get("/articulo/20285/999").status_code == 404

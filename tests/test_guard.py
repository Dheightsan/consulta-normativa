from app.guard import check


def test_valid_citation_and_numbers(chunks):
    text = "Veinte días hábiles, prorrogables por 10 más (Ley 20.285, art. 14)."
    result = check(text, chunks)
    assert result.ok
    assert result.cited == [("20.285", "14")]


def test_number_words_in_source_count(chunks):
    assert check("Son 20 días hábiles. Ley 20.285, art. 14.", chunks).ok


def test_invented_number_is_blocked(chunks):
    assert not check("Son 15 días hábiles. Ley 20.285, art. 14.", chunks).ok


def test_answer_without_citation_is_blocked(chunks):
    assert not check("Son veinte días hábiles.", chunks).ok


def test_citation_outside_retrieved_articles_is_blocked(chunks):
    assert not check("Son veinte días hábiles. Ley 20.285, art. 15.", chunks).ok
    assert not check("Son veinte días hábiles. Ley 19.880, art. 14.", chunks).ok


def test_bare_article_reference_must_exist(chunks):
    assert not check("Ley 20.285, art. 14, y además el artículo 77.", chunks).ok


def test_bis_articles_and_two_laws(chunks):
    text = "Ley 19.886, art. 8 bis, y Ley 19.880, art. 27, con 6 meses."
    assert check(text, chunks).cited == [("19.886", "8 bis"), ("19.880", "27")]

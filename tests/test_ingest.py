from app.ingest import clean, parse_law

XML = """<?xml version="1.0" encoding="UTF-8"?>
<Norma xmlns="http://www.leychile.cl/esquemas">
 <EstructurasFuncionales>
  <EstructuraFuncional tipoParte="Capítulo" transitorio="no transitorio">
   <Texto>CAPITULO I</Texto>
   <Metadatos><TituloParte>CAPITULO I Disposiciones</TituloParte></Metadatos>
   <EstructurasFuncionales>
    <EstructuraFuncional tipoParte="Artículo" transitorio="no transitorio">
     <Texto>     Artículo 1º.- Primera
linea del articulo.

     Segundo inciso.</Texto>
    </EstructuraFuncional>
    <EstructuraFuncional tipoParte="Artículo" transitorio="no transitorio">
     <Texto>     Artículo 2º bis.- Otro texto.</Texto>
    </EstructuraFuncional>
    <EstructuraFuncional tipoParte="Artículo" transitorio="transitorio">
     <Texto>     Artículo 3º.- Transitorio.</Texto>
    </EstructuraFuncional>
    <EstructuraFuncional tipoParte="Artículo" transitorio="no transitorio">
     <Texto>     Artículo primero.- Modifícase otra ley.</Texto>
    </EstructuraFuncional>
   </EstructurasFuncionales>
  </EstructuraFuncional>
 </EstructurasFuncionales>
</Norma>
"""


def test_clean_joins_wrapped_lines():
    assert clean("uno\ndos\n\n   tres\ncuatro") == "uno dos\n\ntres cuatro"


def test_parse_law_by_article(tmp_path):
    (tmp_path / "ley.xml").write_text(XML, encoding="utf-8")
    meta = {"ley": "99.999", "nombre": "Prueba", "archivo": "ley.xml"}
    chunks = parse_law(meta, tmp_path)
    assert [c.numero for c in chunks] == ["1", "2 bis"]
    assert chunks[0].id == "99999-1"
    assert chunks[0].titulo == "CAPITULO I Disposiciones"
    assert chunks[0].texto == "Primera linea del articulo.\n\nSegundo inciso."
    assert chunks[1].id == "99999-2-bis"

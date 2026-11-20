# consulta-normativa

[![CI](https://github.com/Dheightsan/consulta-normativa/actions/workflows/ci.yml/badge.svg)](https://github.com/Dheightsan/consulta-normativa/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/)

Servicio FastAPI que responde preguntas en español sobre tres leyes chilenas y cita el
artículo exacto. Si la respuesta no está en los textos cargados, dice
`No encontré eso en los documentos cargados.` en vez de inventar.

Corpus (textos vigentes de LeyChile/BCN, dominio público según el art. 11 de la Ley 17.336):

- Ley 20.285, Transparencia y acceso a la información pública
- Ley 19.880, Bases de los Procedimientos Administrativos
- Ley 19.886, Compras Públicas

La URL de origen y la fecha de versión de cada texto están en `data/fuentes.json`.

## Para qué sirve en un servicio público

Un funcionario o un ciudadano pregunta "¿cuántos días tiene el servicio para responder
una solicitud de información?" y recibe la respuesta con la cita (`Ley 20.285, art. 14`)
y el extracto del artículo, para verificarla en un clic. Pensado como base de un
asistente interno de consultas normativas: mismo diseño con otras leyes o reglamentos.

## Cómo correrlo

Python 3.12 o superior.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
uvicorn app.main:app --reload
```

Abrir `http://localhost:8000`. Corre sin ninguna clave: usa solo BM25 y responde con
el texto del artículo más relevante. Las claves son opcionales y se leen del entorno
(ver `.env.example`):

| Variable | Para qué |
|---|---|
| `GEMINI_API_KEY` | Embeddings (búsqueda híbrida). Sin ella, solo BM25. |
| `MODEL_BASE_URL`, `MODEL_API_KEY`, `MODEL_NAME` | Modelo de chat con API compatible con OpenAI. Sin ellas, respuesta extractiva. |
| `DATABASE_URL` | Postgres con pgvector (por ejemplo Neon). Sin ella, los vectores van en `data/vectors.npz`. |

Para el modelo de chat sirve el endpoint compatible de Gemini
(`https://generativelanguage.googleapis.com/v1beta/openai`), el de OpenAI o el de
Anthropic (`https://api.anthropic.com/v1`).

Con Postgres local:

```bash
docker compose up -d db
export DATABASE_URL=postgresql://postgres:postgres@localhost:5432/normativa
python -m app.ingest      # crea la tabla y carga los embeddings
```

Reconstruir el índice desde los textos fuente: `python -m app.ingest`.

## Endpoints

- `POST /preguntar` con `{"pregunta": "..."}` devuelve `respuesta`, `citas`
  (ley, artículo, extracto), `fundamentada` y `modo` (`generativa`, `extractiva` o
  `sin_resultado`).
- `GET /articulo/{ley}/{numero}`, por ejemplo `/articulo/20285/14` o `/articulo/19886/8 bis`.
- `GET /health`.

## Cómo funciona

1. `app.ingest` parsea el XML de LeyChile y deja un fragmento por artículo (ley, número,
   título o párrafo, texto). Los artículos transitorios y los que modifican otras leyes
   no se incluyen.
2. La búsqueda combina BM25 con embeddings de `gemini-embedding-2` (768 dimensiones) y
   une los dos rankings por posición. Si la pregunta menciona una ley o un artículo,
   se filtra o se prioriza.
3. El modelo recibe solo los 3 mejores artículos y debe citar con el formato
   `Ley N, art. X`.
4. Una guarda revisa la respuesta: cada cita debe corresponder a un artículo recuperado
   y cada número debe aparecer en esos artículos. Si falla, se devuelve el extracto
   del mejor artículo. Si el modelo dice que no hay respuesta, se devuelve el mensaje
   de "no encontré".
5. Si falla la API de embeddings o la de chat, el servicio sigue con BM25 o con
   respuesta extractiva.

## Evaluación

`python -m eval.run` usa 25 preguntas de `eval/preguntas.jsonl`: 20 con respuesta en las
leyes (con su artículo esperado) y 5 fuera del corpus. Resultado medido el 20-11-2026:

| Modo | Artículo correcto en top-3 | Respuestas fundamentadas | Cita con el artículo esperado | Rechazos correctos |
|---|---|---|---|---|
| Solo BM25 | 15/20 | 20/20 | 10/20 | 2/5 |
| Híbrido | 18/20 | 20/20 | 16/20 | 5/5 |
| Híbrido + gemini-2.5-flash | 18/20 | 19/20 | 17/20 | 5/5 |

"Fundamentadas" cuenta respuestas con cita verificada, no que sean correctas. Los dos
fallos de recuperación del modo híbrido son "qué puedo hacer si me niegan la información"
(el artículo 24 queda fuera del top-3) y "plazo máximo de duración de un procedimiento"
(el artículo 27 de la Ley 19.880 queda fuera del top-3).

## Límites

- El conjunto de evaluación es chico (25 preguntas) y lo escribí yo. Los umbrales de
  relevancia y el peso de los embeddings los ajusté mirando esas mismas preguntas, así
  que las cifras son optimistas.
- Solo hay tres leyes. No incluye reglamentos, dictámenes de Contraloría ni
  jurisprudencia, y no distingue versiones anteriores de un artículo.
- Trabaja por artículo completo. Si la respuesta depende de varios artículos, puede
  quedarse con uno solo.
- Con el modelo, la guarda asegura que las citas y los números existan en los artículos
  recuperados, no que la interpretación sea correcta. No reemplaza una revisión jurídica.
- El backend de Postgres con pgvector está probado con tests y una conexión simulada,
  no contra una base real.
- Sin autenticación, límites de uso ni registro de consultas. No está listo para
  exponerse tal cual.

## Despliegue

Incluye `Dockerfile`, `docker-compose.yml` (API más Postgres con pgvector) y `render.yaml`.
No hay nada desplegado.

## Cómo lo hice

Lo hice en una tarde con Claude Code. Lo que más revisé fue que cada respuesta cite el artículo de donde sale y que no invente nada cuando la ley no lo dice. Por eso en la evaluación también están las preguntas que falla.

## Licencia

MIT, autor Dheightsan. Los textos de ley son de dominio público.

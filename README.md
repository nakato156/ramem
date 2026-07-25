# RAMEM V1

RAMEM es un chat de terminal local con memoria conversacional RAG. Conserva los intercambios en
SQLite, recupera evidencia relevante de conversaciones anteriores y responde con el Gemma
finetuneado mostrando citas `[D1]`, `[D2]`, etc. No ejecuta comandos, no inspecciona repositorios y
no modifica archivos.

## Estado del proyecto

La implementación funcional de V1 está completa en la rama de desarrollo. El código incluye
almacenamiento conversacional, recuperación híbrida, REPL, backends de generación, administración,
adaptadores de benchmark y gates automatizados. **Esto no equivale a una release publicada**:
RaMem-Memory-ES sigue siendo un borrador de 10 casos, faltan el holdout revisado, las mediciones con
100.000 mensajes, la comparación GGUF/BF16 y la publicación en PyPI/Hugging Face.

El estado verificable, los pendientes y la relación entre documentos están en
[`docs/README.md`](docs/README.md). No se deben interpretar métricas smoke como evidencia de
release.

## Instalación

RAMEM requiere Python 3.12. El nombre del paquete en PyPI es `ramem-cli` —`ramem` pertenece a otro
proyecto—, pero el ejecutable conserva el nombre `ramem`.

```bash
# Runtime recomendado: GGUF/llama.cpp
uv tool install "ramem-cli[llama]"

# Alternativa de referencia para GPU
uv tool install "ramem-cli[transformers]"
```

El embedder oficial, EmbeddingGemma, forma parte de la instalación base. Los pesos del generador
se distribuyen por separado bajo los términos de Gemma.

## Primer uso

```bash
export HF_TOKEN=hf_...              # si el repositorio de pesos es gated
ramem setup
ramem model pull
ramem model verify
ramem doctor
ramem --new
```

`ramem` reanuda la última conversación; `ramem --new` crea otra. Para scripts:

```bash
ramem ask --new "Mi café favorito es el de Villa Rica"
ramem ask "¿Qué café prefiero?"
```

En el REPL, `Enter` agrega una línea y `Esc+Enter` envía. `Ctrl+C` cancela únicamente la
generación activa; una segunda interrupción sale. `/help` enumera `/new`, `/sessions`, `/resume`,
`/rename`, `/context`, `/sources`, `/memory`, `/forget`, `/model`, `/status` y `/exit`.

## Administración y recuperación

```bash
ramem session list
ramem session show UUID
ramem session delete UUID --yes
ramem memory search "viaje a Cusco"
ramem memory stats
ramem memory rebuild
```

SQLite es la fuente canónica. LanceDB es un índice descartable: si queda inconsistente, ejecutar
`ramem memory rebuild`. Los trabajos interrumpidos vuelven a `pending` al iniciar y se reintentan
hasta cinco veces. Un fallo del modelo conserva el mensaje de usuario ya confirmado.

Los datos se guardan en el directorio de aplicación del sistema (`platformdirs`), nunca en el
repositorio. Se puede fijar una ubicación para copias o pruebas:

```bash
export RAMEM_DATA_DIR=/ruta/privada/ramem
export RAMEM_CONFIG=/ruta/config.yaml
```

Consultar [formato de almacenamiento](docs/storage-format.md) para respaldos, restauración y
borrado verificable.

## Configuración

[`configs/default.yaml`](configs/default.yaml) usa:

- LanceDB híbrido nativo (vector + FTS + RRF) y MMR para diversidad;
- `google/embeddinggemma-300m` truncado a 256 dimensiones Matryoshka;
- GGUF Q4_K_M mediante llama.cpp por defecto;
- presupuestos 256 instrucciones + 768 reciente + 2.048 memoria + 1.024 respuesta = 4.096 tokens.

Transformers solo se activa explícitamente con `generation.provider: transformers`. Para pruebas
sin descargar modelos se admite `retrieval.embedding_provider: hashing`; no es una configuración
de release.

La precedencia, las rutas y todas las claves están documentadas en
[`docs/configuration.md`](docs/configuration.md).

## Evaluación

```bash
ramem benchmark \
  --dataset data/benchmarks/ramem-memory-es-dev.jsonl \
  --output reports/metrics/ramem-memory-es-dev.json

ramem-release-gates reports/metrics/release-metrics.json \
  --output reports/metrics/release-gates.json
```

El dataset incluido es un borrador de desarrollo y no sustituye la revisión humana ni el holdout.
LongMemEval y LoCoMo se ejecutan y califican con sus herramientas oficiales; el protocolo está en
[evaluación de memoria](docs/memory-evaluation.md). No se publica una cifra de release hasta
cumplir todos los gates.

Ragas se adoptará únicamente como dependencia opcional de evaluación semántica. Precision y recall
por IDs seguirán en el runner determinista de RAMEM porque las variantes ID-based de Ragas aún
pertenecen a su API legacy. Las métricas con juez LLM solo podrán convertirse en gates después de
calibrarlas contra revisión humana en español. La decisión y el plan de integración están en
[evaluación con Ragas](docs/ragas-evaluation.md) y
[ADR-0005](docs/adr/0005-ragas-evaluation.md).

## Desarrollo

```bash
uv sync --extra dev
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run pytest
uv build
```

La selección tecnológica y su matriz ponderada están en
[ADR-0004](docs/adr/0004-rag-framework-selection.md). RAMEM reutiliza LanceDB,
SentenceTransformers, llama.cpp/Transformers, prompt-toolkit y Rich; el código propio se limita a
las reglas conversacionales específicas del producto.

Las herramientas históricas de entrenamiento, exportación y QA permanecen disponibles para
mantener el generador, pero no forman parte del camino principal del chat. Véanse
[arquitectura](docs/architecture.md) y [checklist de release](docs/release.md).

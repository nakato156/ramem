# ADR 0004: componentes RAG nativos antes que un metaframework

- Estado: aceptado
- Fecha: 2026-07-25

## Contexto

RAMEM V1 necesita una memoria conversacional local y auditable, no un agente ni un sistema de
extracción de perfiles. Debe conservar SQLite como fuente canónica, usar EmbeddingGemma, recuperar
evidencia lexical y densa, fusionarla con RRF, ejecutar GGUF o Transformers, borrar sesiones por
completo y reconstruir el índice.

La prioridad es reutilizar software mantenido y probado. Solo se conservará lógica propia cuando el
contrato de RAMEM sea deliberadamente distinto.

Esta decisión cubre el **runtime RAG**. Las herramientas de evaluación pueden vivir en un entorno
aislado; ADR-0005 adopta Ragas bajo ese límite sin convertirlo en orquestador de producción.

## Matriz ponderada

Escala: 1 (inadecuado) a 5 (excelente). El total es la suma ponderada sobre 5.

| Alternativa | Ajuste V1 25% | Fiabilidad 20% | Local 15% | Persistencia 15% | Híbrido 15% | Velocidad 10% | Total |
|---|---:|---:|---:|---:|---:|---:|---:|
| LanceDB nativo + librerías especializadas | 4.8 | 4.5 | 5.0 | 4.7 | 5.0 | 4.7 | **4.78** |
| Qdrant local + `langchain-qdrant` | 4.7 | 4.6 | 4.0 | 4.8 | 5.0 | 4.2 | **4.59** |
| LlamaIndex modular | 4.6 | 3.6 | 5.0 | 4.2 | 4.8 | 3.7 | **4.34** |
| LangChain/LangGraph | 4.5 | 3.2 | 4.8 | 4.5 | 4.5 | 3.5 | **4.19** |
| Haystack | 4.2 | 3.0 | 4.8 | 3.4 | 4.3 | 3.4 | **3.87** |
| Mem0 | 2.3 | 3.1 | 4.0 | 3.6 | 3.0 | 4.2 | **3.21** |

## Evidencia

- LanceDB implementa búsqueda híbrida vectorial/FTS y usa `RRFReranker` por defecto:
  <https://docs.lancedb.com/search/hybrid-search>.
- LanceDB documenta evaluación, ANN, filtros y rerankers nativos:
  <https://docs.lancedb.com/reranking/eval>.
- Qdrant ofrece RRF denso/disperso y borrado transaccional mediante WAL:
  <https://qdrant.tech/documentation/search/hybrid-queries/> y
  <https://qdrant.tech/documentation/concepts/points/>.
- Qdrant local persiste en disco, pero su propia guía lo orienta a cantidades pequeñas de vectores:
  <https://qdrant.tech/documentation/frameworks/langchain/>.
- LlamaIndex dispone de `QueryFusionRetriever`, BM25 y `LanceDBVectorStore`, pero la prueba de
  compatibilidad realizada el 2026-07-25 (`core 0.14.23`, LanceDB `0.5.0`, BM25 `0.7.1`) requirió
  añadir manualmente `pandas`, no declarado por la integración.
- LangGraph tiene SQLite checkpointers y recuperación de pasos, pero su Store persistente recomendado
  para producción es Postgres/Redis/Mongo, y el checkpoint completo duplicaría el historial canónico:
  <https://docs.langchain.com/oss/python/langgraph/persistence>.
- La prueba del 2026-07-25 con LangChain `1.3.14` y `langchain-community 0.4.2` emitió una
  advertencia de deprecación de esa superficie. Una nueva matriz debe repetir estas pruebas, no
  asumir que sus resultados permanecen estables.
- Haystack tiene pipelines explícitos y generadores llama.cpp/Hugging Face, pero su chat store
  persistente sigue siendo experimental y la integración LanceDB disponible es comunitaria `0.1.1`
  de 2024: <https://pypi.org/project/lancedb-haystack/>.
- Mem0 infiere y actualiza hechos mediante un LLM de forma predeterminada. Eso contradice la decisión
  V1 de recuperar evidencia histórica sin memorias derivadas:
  <https://github.com/mem0ai/mem0/blob/main/LLM.md>.

## Decisión

No introducir un metaframework completo en V1. Se reutilizarán componentes especializados:

- **LanceDB:** almacenamiento vectorial reconstruible, FTS, búsqueda híbrida, RRF, filtros y ANN.
- **SentenceTransformers:** EmbeddingGemma y truncamiento Matryoshka.
- **llama-cpp-python / Transformers:** generación local y streaming.
- **SQLite:** transacciones WAL, historial canónico, migraciones, trabajos recuperables y borrado.
- **prompt-toolkit / Rich:** REPL, historial, multilínea, Markdown y streaming.

La búsqueda lexical SQLite FTS5 se conserva como fallback verificable y como herramienta de
diagnóstico; el camino normal usa el híbrido nativo de LanceDB.

## Lógica propia permitida

1. Esquema conversacional inmutable y coordinación SQLite–índice reconstruible.
2. Chunking que preserve fronteras e IDs de mensajes; los splitters documentales no cumplen esto.
3. Construcción de consulta con los dos mensajes de usuario anteriores.
4. Exclusión de la ventana reciente y diversidad posterior a RRF.
5. Presupuesto exacto del prompt y mapeo/validación de citas `[D1]`.
6. CLI, privacidad de trazas y gates específicos de RAMEM.

No se implementarán ANN, FTS, RRF, embeddings, inferencia, streaming Markdown ni almacenamiento
vectorial desde cero.

## Consecuencias

- Menos dependencias y menor superficie de cambios que LlamaIndex/LangChain/Haystack.
- Se mantiene la portabilidad local y el objetivo de 100 000 mensajes sin servidor adicional.
- La interfaz interna seguirá desacoplada para poder sustituir LanceDB por Qdrant si los benchmarks
  reales demuestran que no cumple recall o latencia.
- Ragas y cualquier evaluador futuro se mantienen fuera del grafo de dependencias del runtime; véase
  [`0005-ragas-evaluation.md`](0005-ragas-evaluation.md).

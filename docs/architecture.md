# Arquitectura RAMEM V1

Estado: implementada en código; validación científica y publicación pendientes.

## Flujo principal

```text
mensaje
  → commit SQLite del mensaje de usuario
  → ventana reciente + consulta expandida
  → LanceDB hybrid (EmbeddingGemma + FTS + RRF)
  → deduplicación + MMR + presupuesto
  → prompt no confiable con evidencias [D#]
  → llama.cpp GGUF o Transformers
  → validación de citas
  → commit del asistente
  → chunks + index_jobs → LanceDB
```

La persistencia del mensaje ocurre antes de cargar o invocar el generador. Por ello, cancelar o
fallar la inferencia no pierde entradas confirmadas. La respuesta y su trabajo de indexación se
crean solo tras completar la generación.

## Componentes y límites

| Componente | Responsabilidad | Implementación |
|---|---|---|
| `ConversationStore` | fuente canónica, migraciones, WAL, inmutabilidad y cola | SQLite/FTS5 |
| `ConversationChunker` | intercambios ≤512 tokens, solapamiento y offsets | regla RAMEM |
| `LanceMemoryIndex` | vector, FTS, híbrido, RRF e índice reconstruible | LanceDB |
| `EmbeddingGemmaEmbedder` | embeddings query/document Matryoshka | SentenceTransformers |
| `ConversationalRetriever` | exclusión reciente, deduplicación y diversidad | LanceDB + regla RAMEM |
| `ConversationalContextBuilder` | ventana y presupuesto exacto del tokenizer | regla RAMEM |
| `GeneratorBackend` | streaming/tokenización/cancelación | llama.cpp o Transformers |
| `ConversationService` | transacción lógica del turno y citas | regla RAMEM |
| CLI | REPL, Markdown, historial, comandos | Typer, prompt-toolkit, Rich |
| Evaluación | retrieval, respuesta, sistema y experimentos | runner RAMEM, suites oficiales y Ragas opcional |

La evaluación ponderada de alternativas está en
[`docs/adr/0004-rag-framework-selection.md`](adr/0004-rag-framework-selection.md). LanceDB nativo
evita mantener implementaciones propias de ANN, FTS o RRF y conserva una instalación local
embebida. LangChain, LlamaIndex, Haystack, Qdrant y Mem0 siguen siendo alternativas documentadas,
pero añadir una capa de orquestación completa no mejora los límites específicos de RAMEM V1.

## Persistencia y recuperación

SQLite usa `journal_mode=WAL`, `synchronous=FULL`, claves foráneas y transacciones
`BEGIN IMMEDIATE`. `messages` y `memory_chunks` no admiten `UPDATE`; solo la metadata de una sesión
se puede renombrar. `schema_migrations` rechaza bases creadas por una versión futura.

LanceDB nunca es fuente de verdad. Cada alta o baja de chunk crea primero un `index_job` en la misma
transacción SQLite. Un trabajo `running` encontrado al reiniciar vuelve a `pending`; el trabajador
drena lotes y limita los reintentos fallidos a cinco. `memory rebuild` recrea todo desde SQLite y
comprueba igualdad de conteos.

## Recuperación y confianza

La consulta concatena el mensaje actual con los dos mensajes de usuario previos. Los chunks que
tocan la ventana reciente se excluyen para evitar duplicarla. En modo `hybrid`, LanceDB ejecuta
búsqueda densa + FTS y su `RRFReranker`; luego RAMEM elimina duplicados y usa MMR, con recencia solo
como desempate.

Las memorias se renderizan bajo la sección `MEMORIA` como datos no confiables. El prompt de sistema
declara explícitamente que sus instrucciones no cambian la prioridad. Solo identificadores
asignados al contexto actual son citas válidas; los inventados se eliminan de la salida y se
registran como inválidos sin guardar el texto privado en las trazas.

## Modelos

`backend=auto` significa siempre llama.cpp y GGUF Q4_K_M. Transformers es el backend explícito de
referencia. Ambos reciben la misma lista de mensajes, parámetros deterministas y tokenizer del
artefacto publicado. `model pull` resuelve una referencia de Hugging Face a un SHA inmutable,
descarga ese snapshot y valida tamaño y SHA-256 de cada archivo declarado.

Los artefactos de modelo no heredan automáticamente Apache-2.0: conservan los términos de Gemma.

## Evaluación fuera del runtime

La evaluación no participa en una conversación normal:

```text
salida RAMEM + IDs recuperados + referencias
  → métricas deterministas RAMEM, incluidos precision/recall por IDs
  → evaluadores oficiales LongMemEval/LoCoMo
  → Ragas opcional para jueces semánticos calibrados
  → ReleaseMetrics → ramem-release-gates
```

Ragas no sustituye el retriever, LanceDB, los benchmarks oficiales ni la validación determinista de
citas. Se instalará en un entorno separado porque incorpora dependencias de metaframework que no
deben ampliar el runtime local. Véase [`ragas-evaluation.md`](ragas-evaluation.md).

## Invariantes

1. SQLite es la única fuente canónica de conversaciones.
2. Una entrada confirmada se persiste antes de inferir.
3. El índice vectorial se puede destruir y reconstruir sin perder conversaciones.
4. El texto recuperado nunca adquiere autoridad de sistema.
5. Ninguna traza diagnóstica guarda mensajes, consultas o fragmentos recuperados.
6. Un resultado smoke no se presenta como benchmark de release.

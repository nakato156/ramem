# Traspaso operativo de RAMEM

Última actualización: 2026-07-25 (America/Lima).

## Estado ejecutivo

RAMEM V1 ya está implementado como CLI conversacional con memoria RAG. El camino principal incluye
sesiones persistentes, chunking conversacional, recuperación híbrida, selección por presupuesto,
generación GGUF/Transformers, citas, REPL, administración y evaluación reproducible.

La implementación no debe confundirse con una publicación terminada. El siguiente objetivo es
producir evidencia de release, no reconstruir el RAG ni volver a entrenar sin diagnóstico.

Rama de integración actual:

```text
codex/ramem-v1
```

Commit base de la implementación V1:

```text
6fd6d75f36ba5e00db4a6b56fc459025e4d943c7
```

Antes de ejecutar trabajos, comprobar siempre `git status --short --branch` y `git rev-parse HEAD`;
el estado remoto puede avanzar después de este documento.

## Qué está implementado

- CLI `ramem`, `ramem --new` y `ramem ask`.
- Comandos `setup`, `doctor`, `model`, `session`, `memory` y `benchmark`.
- SQLite WAL como fuente canónica, migraciones y tablas append-only.
- Cola recuperable `index_jobs` y reconstrucción completa de LanceDB.
- Chunking por intercambios con IDs y offsets de origen.
- EmbeddingGemma/Matryoshka y proveedor hashing para pruebas.
- Recuperación LanceDB densa, FTS, híbrida, RRF, deduplicación y MMR.
- Contexto reciente + memoria recuperada bajo presupuesto.
- Backends llama.cpp GGUF y Transformers con streaming y cancelación.
- Manifiesto y verificación SHA-256 de modelo.
- RaMem-Memory-ES smoke, adaptadores LongMemEval/LoCoMo y release gates.
- Unitarias, integración y pruebas PTY; estas últimas se ejecutan en Linux.

La descripción técnica canónica está en [`architecture.md`](architecture.md). Los detalles de
persistencia y recuperación están en [`storage-format.md`](storage-format.md).

## Evidencia disponible y límites

### Validación local del código

El 2026-07-25, el árbol documentado pasó Ruff, formato, mypy y 52 pruebas; 2 pruebas PTY se omitieron
en Windows y deben confirmarse en Linux CI. `uv build` produjo wheel y sdist. Esta evidencia valida
la implementación local, no los benchmarks científicos ni la publicación.

### Generador finetuneado

El historial completo de fase está en [`resume-fase1.md`](resume-fase1.md). Los resultados
registrados son:

| Evaluación | Casos | Token F1 adaptador | Alcance |
|---|---:|---:|---|
| interna | 256 | 0,7168 | ajuste al formato grounded |
| MLQA español external-dev | 500 | 0,6881 | generalización del generador |

Estas métricas no miden memoria conversacional. `valid_citations = 1,0` valida IDs, no soporte
semántico.

### Recuperación

El piloto histórico MIRACL obtuvo Recall@10 0,9907 denso y 0,9182 RRF sobre 54 consultas
condicionadas. No representa el corpus completo ni invalida el híbrido conversacional actual. Véase
[`retrieval_evaluation.md`](retrieval_evaluation.md).

### Memoria V1

`reports/metrics/ramem-memory-es-dev-smoke.json` registra Recall@10 1,0 sobre 10 casos, 32 mensajes y
16 chunks. Es una prueba funcional demasiado pequeña y construida para autorizar una afirmación de
calidad o rendimiento.

## Decisiones vigentes

1. SQLite sigue siendo fuente de verdad y LanceDB un índice reconstruible.
2. LanceDB y librerías especializadas se prefieren a un metaframework completo
   ([ADR-0004](adr/0004-rag-framework-selection.md)).
3. Ragas se adopta únicamente como evaluación opcional
   ([ADR-0005](adr/0005-ragas-evaluation.md)).
4. No reentrenar el generador hasta identificar un fallo atribuible al modelo.
5. No ejecutar ningún holdout antes de congelar candidato, datos, prompts, hashes y umbrales.
6. No usar métricas smoke, MLQA ni MIRACL piloto como sustituto de evaluación de memoria.

## Próximos trabajos, en orden

### 1. Consolidar CI y paquete

1. Ejecutar Ruff, formato, mypy y pytest en Linux para incluir PTY.
2. Instalar el wheel en un entorno vacío.
3. Reproducir `setup`, `doctor`, primera conversación, cierre y recuperación solo con el README.
4. Registrar commit, SO, CPU/GPU, RAM y versiones.

### 2. Cerrar RaMem-Memory-ES

1. Ampliar development con revisión humana independiente.
2. Auditar evidencia, temporalidad, actualización y abstención.
3. Crear holdout fuera del flujo de desarrollo.
4. Congelar hashes y reglas de acceso.
5. Ejecutar comparaciones de ventana reciente, truncado, oráculo y recuperación.

### 3. Evaluar recuperación y escala

1. Comparar 768/256/128 dimensiones con embeddings base reutilizados y renormalizados.
2. Comparar chunking y presupuesto sin barrido cartesiano indiscriminado.
3. Medir recall, precisión, MRR/nDCG, latencia y memoria.
4. Ejecutar la prueba de 100.000 mensajes en la máquina documentada.
5. Sustituir LanceDB solo si la evidencia contradice los gates.

### 4. Pilotar Ragas

1. Crear un entorno opcional fijado a `ragas>=0.4.3,<0.5`.
2. Exportar IDs y contextos con el contrato de
   [`ragas-evaluation.md`](ragas-evaluation.md).
3. Confirmar precision/recall por IDs con el runner determinista RAMEM.
4. Etiquetar 20–50 casos españoles y calibrar Faithfulness, NoiseSensitivity y
   FactualCorrectness.
5. Mantener los resultados como diagnósticos hasta demostrar acuerdo humano suficiente.

### 5. Cerrar modelos y publicación

1. Producir GGUF Q4_K_M con herramientas oficiales de llama.cpp.
2. Compararlo con BF16 usando el benchmark congelado.
3. Medir tokens/s y primer token en CPU de referencia.
4. Publicar pesos, tokenizer, template, manifiesto, hashes, métricas y términos.
5. Publicar `ramem-cli` y completar [`release.md`](release.md).

## Entornos reproducibles

Los extras `training` y `retrieval` son incompatibles deliberadamente:

```bash
uv sync --extra dev --extra training
uv sync --extra dev --extra retrieval
```

El runtime normal no necesita ninguno. Ragas debe vivir en otro entorno y no mezclarse con esos
extras. Al cambiar de fase, ejecutar la resolución correspondiente desde `uv.lock`.

## Lightning

Lightning se usa para mantener/evaluar el generador y para benchmarks GPU grandes; no es necesario
para el uso normal del CLI. El procedimiento está en
[`LIGHTNING_RUNBOOK.md`](LIGHTNING_RUNBOOK.md). Sustituir cualquier referencia a rama por la rama o
tag congelado para el experimento y confirmar el SHA antes de consumir GPU.

## Definición de terminado

RAMEM V1 puede publicarse cuando todas las casillas de [`release.md`](release.md) tengan evidencia:

- CI completa, instalación limpia y recuperación documentada;
- benchmark español humano y suites externas congeladas;
- gates de calidad, durabilidad, borrado, latencia y tokens cumplidos;
- comparación GGUF/BF16 aprobada;
- artefactos remotos verificables y licencias correctas;
- código, documentación, tags y manifiestos coherentes.

Ragas mejora el diagnóstico, pero su integración no es requisito obligatorio de V1 salvo que una
métrica Ragas se incorpore formalmente a los gates antes de congelar el candidato.

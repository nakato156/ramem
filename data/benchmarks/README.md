# RaMem-Memory-ES

`ramem-memory-es-dev.jsonl` es la semilla de desarrollo del benchmark conversacional en español.
Contiene ejemplos para extracción, cruce entre sesiones, actualización, temporalidad y abstención.
Actualmente contiene 10 casos: 2 por categoría, 32 mensajes y 16 intercambios indexables.

Estado de gobernanza: **borrador pendiente de revisión humana independiente**. No debe confundirse
con el holdout ni utilizarse para afirmar que se cumplieron los gates de release. El holdout se
mantendrá fuera del repositorio de trabajo y solo podrá ejecutarse con
`RAMEM_RELEASE_CANDIDATE_FROZEN=yes`.

Cada línea valida contra `MemoryBenchmarkCase` y declara frases exactas de evidencia. Para ejecutar:

```bash
ramem benchmark --dataset data/benchmarks/ramem-memory-es-dev.jsonl \
  --k 10 --output reports/metrics/ramem-memory-es-dev.json
```

Campos principales:

- `case_id`, `corpus_id`, `category` y `split`;
- `sessions[]` con `source_id`, título e intercambios;
- `query`;
- `relevant_phrases` o `relevant_session_ids`.

Los `source_id` permiten evaluar precision/recall por IDs en RAMEM y conservar trazabilidad al
exportar contextos para Ragas, sin depender de similitud textual. Un caso de abstención sin evidencia
relevante obtiene recall 1,0 por definición del runner; la calidad de la abstención de la respuesta
se mide en una evaluación de generación separada.

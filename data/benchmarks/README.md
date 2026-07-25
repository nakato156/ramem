# RaMem-Memory-ES

`ramem-memory-es-dev.jsonl` es la semilla de desarrollo del benchmark conversacional en español.
Contiene ejemplos para extracción, cruce entre sesiones, actualización, temporalidad y abstención.

Estado de gobernanza: **borrador pendiente de revisión humana independiente**. No debe confundirse
con el holdout ni utilizarse para afirmar que se cumplieron los gates de release. El holdout se
mantendrá fuera del repositorio de trabajo y solo podrá ejecutarse con
`RAMEM_RELEASE_CANDIDATE_FROZEN=yes`.

Cada línea valida contra `MemoryBenchmarkCase` y declara frases exactas de evidencia. Para ejecutar:

```bash
ramem benchmark --dataset data/benchmarks/ramem-memory-es-dev.jsonl \
  --k 10 --output reports/metrics/ramem-memory-es-dev.json
```

# Evaluación de memoria

RAMEM separa tres niveles: recuperación, respuesta y gates de sistema. No reemplaza con métricas
propias los evaluadores oficiales de benchmarks externos.

## RaMem-Memory-ES

El esquema JSONL está implementado en `ramem.evaluation.memory`. Cada caso declara categoría,
sesiones, consulta y frases de evidencia. El comando `ramem benchmark --dataset ...` crea un
almacenamiento temporal, ingiere las conversaciones, mide Recall@k global/por categoría y latencia,
y destruye el índice aislado al terminar.

`data/benchmarks/ramem-memory-es-dev.jsonl` es un borrador para desarrollo. Antes del release se
requieren:

1. revisión independiente de conversaciones y evidencia exacta;
2. suficientes casos por extracción, cross-session, actualización, temporalidad y abstención;
3. separación de development y holdout;
4. hashes y umbrales congelados;
5. `RAMEM_RELEASE_CANDIDATE_FROZEN=yes` para abrir el holdout.

## LongMemEval y LoCoMo

Se usan las distribuciones y scripts oficiales:

- [LongMemEval](https://github.com/xiaowu0162/LongMemEval) para extracción, razonamiento
  multi-sesión, actualización, temporalidad y abstención.
- [LoCoMo](https://github.com/snap-research/locomo) como segunda prueba de conversaciones largas.
- [memory-benchmarks](https://github.com/mem0ai/memory-benchmarks) como runner comparativo
  mantenido para ambos datasets.

Los repositorios se clonan fuera de RAMEM y se fijan a los commits de
`data/datasets_manifest.yaml`. Sus datos no se incorporan al wheel ni al repositorio. RAMEM debe
exportar hipótesis con el formato oficial y el evaluador oficial calcula la exactitud; para
recuperación se conservan IDs de sesión/evidencia originales. LongMemEval está bajo MIT y LoCoMo
bajo CC-BY-NC-4.0, por lo que LoCoMo solo se usa en evaluación no comercial.

Para medir recuperación con exactamente el pipeline RAMEM, adaptar primero los archivos oficiales:

```bash
ramem-adapt-memory-benchmark longmemeval longmemeval_s_cleaned.json longmem.jsonl
ramem-adapt-memory-benchmark locomo locomo10.json locomo.jsonl
ramem benchmark --dataset longmem.jsonl --output longmem-retrieval.json
```

El adaptador conserva IDs oficiales de sesión/evidencia. La exactitud de respuestas sigue
calculándose con los scripts oficiales; RAMEM no replica ni altera el judge.

## Comparaciones congeladas

Con el mismo dataset, prompt, semilla y generador:

- solo ventana reciente;
- historial truncado;
- contexto oráculo;
- memoria recuperada;
- Transformers BF16;
- GGUF Q4_K_M.

`ramem-release-gates` consume un JSON de métricas y falla con código distinto de cero si cualquier
umbral del plan V1 no se cumple. No se rellenan valores faltantes con estimaciones.

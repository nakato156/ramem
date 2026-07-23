# Resumen fase 1: modelo finetuneado

Ultima actualizacion: 2026-07-23 (America/Lima).

## Alcance

Esta fase evalua solamente el modelo finetuneado de QA en espanol con respuestas citadas. No incluye
retrieval, chunking, ranking, RAG ni el pipeline RaMem end-to-end.

La pregunta de decision es:

> El checkpoint finetuneado actual es suficiente para pasar a una fase RAG controlada, o necesita
> mas ajuste/evaluacion antes?

Decision actual: el checkpoint es viable y queda aceptado para evaluacion go/no-go. No hay evidencia
que justifique otro entrenamiento a ciegas. Antes de declararlo final, falta una validacion mas
estricta de fidelidad, abstencion y test externo reservado.

## Artefactos cerrados

Modelo fusionado canonico:

`artifacts/export/gemma-1b-ramem-seed42/model.safetensors`

Verificacion:

- Base: `google/gemma-3-1b-it`
- Formato: `merged_transformers_safetensors`
- Dtype: BF16
- Bytes: `1,999,811,208`
- SHA-256: `03a514d1dda8ddff236b6f4056cf89184e553a691ae01a0946c1aea08ca21e23`
- Parametros cargados: `999,885,952`
- Vocabulario: `262,144`
- Parametros no finitos: `0`

Adaptador final:

`artifacts/training/gemma-1b-t4-seed42/adapter-final/adapter_model.safetensors`

Verificacion:

- Bytes: `26,139,992`
- SHA-256: `cde9f18b52323eed44ad35cfe74968c131284b36c65f26d533340ecd4e55d23d`

Manifest:

`artifacts/export/gemma-1b-ramem-seed42/ramem_export_manifest.json`

El modelo canonico fue reconstruido localmente desde el modelo base autorizado y el adaptador, y
coincide exactamente con el SHA-256 registrado para el export de Lightning. Tambien fue verificado en
Lightning CPU el 2026-07-23.

## Evaluacion interna

Fuente:

`artifacts/evaluation/gemma-1b-t4-seed42/summary.json`

Muestra: 256 ejemplos.

| Metrica | Base | Adaptador | Delta |
|---|---:|---:|---:|
| Exact Match | 0.0352 | 0.5469 | +0.5117 |
| Token F1 | 0.1843 | 0.7168 | +0.5325 |
| Cita `[D1]` | 0.8125 | 1.0000 | +0.1875 |
| Latencia media | 5.007 s | 1.454 s | -3.552 s |
| Pico VRAM | 1,023,609,856 B | 1,091,137,024 B | +67,527,168 B |
| Runtime total | 1282.223 s | 372.860 s | -909.363 s |

Lectura: el adaptador mejora ampliamente exactitud, F1 y formato de cita frente al modelo base, con
latencia media mucho menor.

## Evaluacion external-dev

Fuente:

`artifacts/evaluation/mlqa-es-external-dev-seed42/`

Muestra: 500 ejemplos de MLQA espanol external-dev. Este resultado es development externo, no test
final.

| Metrica | Base | Adaptador | Delta |
|---|---:|---:|---:|
| Exact Match | 0.0360 | 0.4860 | +0.4500 |
| Token F1 | 0.2146 | 0.6881 | +0.4735 |
| Cita `[D1]` | 0.7100 | 1.0000 | +0.2900 |
| IDs de cita validos | 0.2760 | 1.0000 | +0.7240 |
| Latencia media | 5.315 s | 1.521 s | -3.794 s |
| Pico VRAM | 1,088,303,104 B | 1,154,899,456 B | +66,596,352 B |
| Runtime total | 2658.703 s | 761.701 s | -1897.002 s |

Decision registrada:

- Estado: `accepted_external_dev`
- Criterios aprobados: EM, Token F1, delta F1, tasa de cita, citas validas y latencia.
- IC bootstrap 95% del delta Token F1: `[0.4373, 0.5091]`
- IC bootstrap 95% del delta Exact Match: `[0.4060, 0.4940]`

Criterios usados:

| Criterio | Umbral | Resultado adaptador | Pasa |
|---|---:|---:|---:|
| Min Exact Match | 0.25 | 0.4860 | Si |
| Min Token F1 | 0.45 | 0.6881 | Si |
| Min delta F1 | 0.10 | +0.4735 | Si |
| Min tasa de cita | 0.98 | 1.0000 | Si |
| Min citas validas | 0.98 | 1.0000 | Si |
| Max latencia media | 3.0 s | 1.521 s | Si |

Categorias de error del adaptador en external-dev:

| Categoria | Casos | Proporcion |
|---|---:|---:|
| Exacto | 243 | 48.6% |
| Parcial alto | 119 | 23.8% |
| Parcial bajo | 55 | 11.0% |
| Incorrecto/sin solapamiento | 83 | 16.6% |

## Interpretacion

El finetuning funciona. La mejora no parece ruido: el delta de Token F1 es grande y su intervalo de
confianza queda lejos de cero. El adaptador tambien mejora el cumplimiento del formato de citas y
reduce la latencia media.

La conclusion practica es:

- No reentrenar todavia.
- No declarar final todavia.
- Usar el checkpoint actual para una evaluacion go/no-go mas estricta.
- Si pasa esa evaluacion, congelarlo y avanzar a RAG.

## Limitaciones actuales

La metrica `valid_citations = 1.0` no demuestra fidelidad semantica. Verifica que el modelo emita el
ID esperado, pero no prueba que cada afirmacion este soportada por el contexto citado.

Faltan pruebas explicitas de:

- Abstencion cuando no hay evidencia suficiente.
- Resistencia a contexto distractor.
- Preguntas ambiguas.
- Soporte semantico de citas.
- Revision manual de errores sin solapamiento.
- Test externo final reservado.

## Ragas en esta fase

Ragas puede ayudar como capa complementaria para validar fidelidad semantica del modelo finetuneado,
especialmente en los casos donde la metrica deterministica solo valida formato de cita.

Uso recomendado en fase 1:

- `Faithfulness` para medir soporte de afirmaciones en el contexto.
- `Factual Correctness` contra referencias.
- `Response Relevancy`.
- Una metrica custom de cita-soporte.
- Una metrica custom de abstencion.

Condiciones:

- No usar RaMem como juez de si mismo.
- Usar un juez mas fuerte, con modelo/version/prompt/temperatura congelados.
- Adaptar prompts a espanol.
- Calibrar contra revision humana antes de usar Ragas como gate.
- Empezar con unos 100 casos estratificados de external-dev, incluyendo peores casos.

Ragas no reemplaza EM, Token F1, external-dev, external-final ni revision humana.

## Siguiente decision

Secuencia recomendada:

1. Congelar el protocolo de evaluacion final del modelo finetuneado.
2. Revisar manualmente una muestra estratificada de external-dev.
3. Correr piloto Ragas en esa muestra y calibrar contra humanos.
4. Ejecutar external-final reservado una sola vez.
5. Si confirma external-dev, congelar checkpoint y pasar a RAG.
6. Si falla, diagnosticar causa antes de cualquier reentrenamiento.

## Estado de validacion tecnica

El 2026-07-23 quedaron sincronizados local, GitHub y Lightning CPU.

Commit verificado:

`7533b9ca859afdbc76a99d4150b0c49be4f1efe6`

Validaciones pasadas localmente y en Lightning CPU:

- `uv lock --check`
- `uv run ruff check .`
- `uv run mypy --no-incremental src tests`
- `uv run pytest` con `23 passed`

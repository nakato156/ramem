# Configuración de RAMEM V1

RAMEM carga `configs/default.yaml` solo cuando se pasa con `--config` o se define `RAMEM_CONFIG`.
Sin esa referencia usa los mismos valores predeterminados incorporados en `ramem.config`.
Las claves desconocidas se rechazan.

## Rutas y precedencia

```text
--config ARCHIVO
  > RAMEM_CONFIG
  > valores incorporados
```

`RAMEM_DATA_DIR` fija la raíz de SQLite, LanceDB, modelos y trazas, salvo que el YAML declare rutas
específicas en `storage` o `telemetry`. `RAMEM_ARTIFACTS_DIR` redirige las trazas para los workflows
históricos de experimentación. Véase [`storage-format.md`](storage-format.md) para `prompt-history`.

## Recuperación

| Clave | Predeterminado | Significado |
|---|---:|---|
| `mode` | `hybrid` | `dense`, `lexical` o híbrido |
| `lexical_k` / `dense_k` | 20 / 20 | candidatos antes de fusión |
| `final_k` | 5 | candidatos antes de empaquetar |
| `rrf_k` | 60 | constante de reciprocal-rank fusion |
| `embedding_dimension` | 256 | 128, 256 o 768 con EmbeddingGemma |
| `chunk_tokens` | 512 | límite objetivo del chunk conversacional |
| `memory_token_budget` | 2048 | espejo de compatibilidad; no gobierna el empaquetado |
| `mmr_lambda` | 0,75 | equilibrio relevancia/diversidad |
| `near_duplicate_threshold` | 0,92 | umbral coseno de deduplicación |

`embedding_provider: hashing` existe para pruebas sin descarga. No es configuración de release.
El presupuesto efectivo de memoria es `context.token_budget`. En V1,
`retrieval.memory_token_budget` se conserva por compatibilidad con configuraciones experimentales y
debe mantener el mismo valor para evitar ambigüedad.

## Contexto y generación

Los presupuestos de instrucciones, ventana reciente, memoria y respuesta no pueden superar
`total_token_budget`. La configuración incluida asigna 256 + 768 + 2.048 + 1.024 = 4.096 tokens.

`generation.provider: auto` selecciona llama.cpp. `transformers` debe elegirse explícitamente. Para
un GGUF local se puede fijar `gguf_path`; si no, RAMEM usa el directorio administrado por
`model pull`. `model_revision` puede ser una rama o tag de entrada, pero `model pull` la resuelve a
un SHA de Hugging Face antes de aceptar el snapshot.

## Ejemplo aislado

```yaml
version: 1
retrieval:
  mode: hybrid
  lexical_k: 20
  dense_k: 20
  final_k: 5
  rrf_k: 60
  embedding_dimension: 256
  embedding_model_id: google/embeddinggemma-300m
  embedding_provider: hashing
  chunk_tokens: 512
  memory_token_budget: 2048
  mmr_lambda: 0.75
  near_duplicate_threshold: 0.92
context:
  token_budget: 2048
  chars_per_token: 4
  recent_token_budget: 768
  instruction_token_budget: 256
  total_token_budget: 4096
generation:
  provider: auto
  model_id: nakato156/ramem-gemma-1b
  model_revision: main
  gguf_filename: ramem-gemma-1b-q4_k_m.gguf
  gguf_path: null
  adapter_path: null
  load_in_4bit: true
  max_new_tokens: 1024
  context_length: 4096
  temperature: 0.0
  top_p: 1.0
  gpu_layers: 0
  seed: 42
storage:
  index_path: null
  lance_path: null
telemetry:
  traces_dir: null
```

Este ejemplo usa hashing deliberadamente para una prueba aislada. Sustituirlo por
`embeddinggemma` antes de medir calidad.

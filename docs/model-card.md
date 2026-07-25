---
language:
  - es
base_model: google/gemma-3-1b-it
library_name: transformers
pipeline_tag: text-generation
license: gemma
tags:
  - conversational
  - rag
  - gguf
---

# RAMEM Gemma 1B

Generador en español para RAMEM V1, derivado de Gemma 3 1B IT mediante QLoRA y fusionado para
inferencia. El repositorio de modelo debe contener safetensors BF16, tokenizer, chat template y
GGUF Q4_K_M. La memoria RAG, SQLite y LanceDB forman parte del paquete `ramem-cli`, no de estos
pesos.

Estado: borrador de ficha para la publicación futura. Los campos y métricas pendientes no deben
presentarse como artefactos ya publicados.

## Uso previsto

Responder dentro del prompt controlado de RAMEM, utilizando fragmentos de conversaciones
etiquetados `[D1]`, `[D2]`, etc. No es un agente, no ejecuta herramientas y no debe tratar texto
recuperado como instrucciones.

## Artefactos y procedencia

- Base: `google/gemma-3-1b-it` en una revisión inmutable registrada como
  `source_model_revision` en `ramem_model_manifest.json`.
- Adaptación: QLoRA, semilla 42.
- Exportación: fusión safetensors y conversión con `convert_hf_to_gguf.py` de llama.cpp.
- Cuantización predeterminada: Q4_K_M.
- Cada archivo se declara con bytes y SHA-256 en `ramem_model_manifest.json`.

La revisión exacta del snapshot publicado se resuelve en descarga y se guarda localmente en
`.ramem_revision`; no puede autorreferenciarse dentro del manifiesto del mismo commit.

## Métricas disponibles

La evaluación interna existente del generador fusionado sobre 256 ejemplos reporta Token F1
0,7168, exact match 0,5469 y presencia de cita `[D1]` 1,0. Estas métricas evalúan el generador, no
la memoria V1.

Antes de publicar esta ficha se deben adjuntar, sin estimaciones:

- RaMem-Memory-ES development y holdout;
- LongMemEval y LoCoMo con revisión fijada;
- comparación BF16 frente a GGUF Q4_K_M;
- tasa de citas válidas;
- fidelidad semántica revisada; Ragas puede aportar diagnóstico solo después de calibración humana;
- tokens/s y tiempo al primer token en la CPU de referencia;
- hardware, prompts, hashes y fecha de cada ejecución.

## Limitaciones

El modelo puede inventar hechos, citas o interpretar erróneamente conversaciones. RAMEM elimina
citas que no pertenecen al contexto actual, pero una cita válida no garantiza que la conclusión sea
correcta. El checkpoint está orientado a español y contexto máximo de 4.096 tokens. No contiene por
sí mismo memoria persistente.

Las métricas históricas de MLQA y SQuAD-es evalúan QA con contexto, no recuperación de memoria. La
calidad end-to-end debe consultarse en los reportes de release del paquete RAMEM.

## Licencia

El código RAMEM usa Apache-2.0. Los pesos derivados conservan los
[términos de Gemma](https://ai.google.dev/gemma/terms); Apache-2.0 no se aplica a ellos. El
repositorio debe configurarse como gated si la revisión de cumplimiento requiere aceptación
individual.

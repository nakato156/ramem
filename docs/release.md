# Checklist de release RAMEM V1

## Código y paquete

- [ ] Ruff, formato, mypy, unitarias, migraciones, integración y PTY verdes en CI.
- [ ] `uv build` produce wheel y sdist; instalación en entorno vacío ejecuta `ramem doctor`.
- [ ] El nombre `ramem-cli` está disponible y el paquete se publica en PyPI.
- [ ] README reproducido desde una máquina limpia.
- [ ] Changelog y tag `v1.0.0`.

## Datos y métricas

- [ ] RaMem-Memory-ES revisado por humanos y holdout congelado.
- [ ] LongMemEval y LoCoMo fijados a commit/licencia y ejecutados con sus evaluadores oficiales.
- [ ] Recall@10 global ≥0,90 y por categoría ≥0,80.
- [ ] Mejora ≥10 puntos frente a ventana reciente.
- [ ] ≥80% de exactitud oráculo usando ≤10% de tokens.
- [ ] Citas válidas ≥0,98.
- [ ] p95 recuperación+packing ≤1,5 s con 100.000 mensajes.

## Modelo

- [ ] Safetensors BF16 fusionado, tokenizer/chat template y GGUF Q4_K_M canónicos.
- [ ] Conversión con herramientas oficiales de llama.cpp y manifiesto SHA-256.
- [ ] Pérdida Q4 ≤2 puntos Token F1 frente a BF16.
- [ ] CPU ≥5 tokens/s y primer token <5 s en máquina documentada.
- [ ] Model card, métricas, términos de Gemma y gating revisados.
- [ ] Repositorio Hugging Face publicado; `model pull` resuelve y verifica un SHA inmutable.

## Sistema

- [ ] Cero pérdida/duplicación tras restart, crash y cancelación.
- [ ] Borrado completo verificado.
- [ ] Primera conversación, salida y recuperación funcionan únicamente con el README.

El archivo generado por `ramem-release-gates` es la evidencia mecánica de los umbrales; este
checklist conserva las aprobaciones humanas y publicaciones externas que no pueden automatizarse.
Copiar `configs/release/metrics-template.json`, reemplazar cada valor con evidencia observada y
adjuntar el reporte. La ficha de modelo lista para completar está en `docs/model-card.md`.

# Exportación para RAMEM V1

Estado: procedimiento preparado; la comparación GGUF/BF16 y el artefacto remoto final siguen
pendientes y figuran en `docs/release.md`.

1. Fusionar el adaptador con `ramem-export`.
2. Preparar el directorio compatible con llama.cpp mediante `prepare_gguf_source.py`.
3. Ejecutar el conversor oficial `convert_hf_to_gguf.py`.
4. Cuantizar con `llama-quantize ... Q4_K_M`.
5. Comparar Q4_K_M contra los safetensors BF16 con el benchmark congelado.
6. Construir el manifiesto verificable:

```bash
python scripts/export/build_release_manifest.py artifacts/export/release \
  --repo-id nakato156/ramem-gemma-1b \
  --source-model-revision COMMIT_HF_BASE_INMUTABLE \
  --source-commit COMMIT_GIT
```

El manifiesto incluye tamaño y SHA-256 de cada archivo. Los pesos conservan los términos de Gemma;
la licencia Apache-2.0 del código no sustituye esos términos.

La revisión inmutable del repositorio de salida no se incluye dentro de su propio manifiesto
(hacerlo crearía una referencia circular). `ramem model pull` resuelve la revisión configurada a
un SHA de Hugging Face, la guarda en `.ramem_revision` y luego valida todos los hashes.

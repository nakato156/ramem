# Changelog

## 1.0.0 - Unreleased

- CLI conversacional interactiva y no interactiva con sesiones persistentes.
- Memoria híbrida local con SQLite FTS5, LanceDB, EmbeddingGemma, RRF y MMR.
- Backends GGUF/llama.cpp y Transformers con streaming, cancelación y tokenización exacta.
- Modelo descargable por snapshot inmutable y manifiesto SHA-256.
- Benchmark RaMem-Memory-ES, evaluador de gates, pruebas de integración y pseudoterminal.
- Adaptadores de LongMemEval y LoCoMo que preservan IDs oficiales de evidencia.
- Documentación operativa de arquitectura, almacenamiento, evaluación, Ragas y release.

### Estado de publicación

- El código V1 está implementado, pero la release permanece sin publicar.
- RaMem-Memory-ES incluido es una muestra smoke de 10 casos, no el benchmark humano final.
- Ragas está decidido como capa opcional futura de evaluación; no es dependencia del runtime ni gate
  de release mientras no exista calibración humana en español.
- Siguen pendientes los artefactos GGUF finales, benchmarks completos, validación en hardware de
  referencia y publicación en PyPI/Hugging Face.

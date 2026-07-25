# ADR 0002: baseline de recuperación offline

- Estado: reemplazado para producción por ADR-0004
- Alcance vigente: pruebas sin descarga y diagnóstico

El primer baseline usa SQLite FTS5, vectores hashing deterministas y RRF. Sigue disponible para
pruebas sin modelos y validación de orquestación. El camino normal V1 usa EmbeddingGemma y LanceDB
híbrido según [`0004-rag-framework-selection.md`](0004-rag-framework-selection.md).

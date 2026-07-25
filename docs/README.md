# Documentación de RAMEM

Última auditoría integral: 2026-07-25.

## Estado canónico

RAMEM V1 está implementado como CLI conversacional local con memoria RAG. La implementación y la
publicación son estados distintos: el paquete todavía figura como `1.0.0 - Unreleased` y no puede
declararse listo hasta completar [`release.md`](release.md).

| Área | Estado | Evidencia o documento |
|---|---|---|
| CLI, sesiones y administración | Implementado | [`README.md`](../README.md), pruebas de integración |
| SQLite, WAL, migraciones y borrado | Implementado | [`storage-format.md`](storage-format.md) |
| LanceDB híbrido, RRF y MMR | Implementado | [`architecture.md`](architecture.md) |
| GGUF y Transformers | Interfaces implementadas | artefactos y comparación final pendientes |
| RaMem-Memory-ES | Smoke implementado | 10 casos; revisión humana y holdout pendientes |
| LongMemEval y LoCoMo | Adaptadores implementados | ejecución oficial completa pendiente |
| Ragas | Decidido, no integrado | [`ragas-evaluation.md`](ragas-evaluation.md) |
| PyPI y Hugging Face | Pendiente | [`release.md`](release.md) |

El reporte `reports/metrics/ramem-memory-es-dev-smoke.json` prueba que el runner funciona sobre la
muestra incluida; no prueba generalización ni cumplimiento de gates.

## Trazabilidad con el plan V1

| Fase original | Estado actual |
|---|---|
| 1. Realinear el proyecto | completa |
| 2. Almacenamiento conversacional | implementada y probada |
| 3. Memoria RAG | implementada; benchmarks completos pendientes |
| 4. Integrar Gemma | backends y verificación implementados; GGUF final pendiente |
| 5. Construir REPL | implementada; PTY Linux pendiente en CI |
| 6. Evaluar memoria | runners/adaptadores implementados; datos humanos y holdout pendientes |
| 7. Publicar | pendiente |

El fuera de alcance se mantiene: no hay API HTTP, herramientas, importadores, multiusuario,
sincronización, perfiles derivados, grafos de memoria ni bucles autónomos.

## Última validación local

El 2026-07-25, sobre Windows:

- Ruff y comprobación de formato: correctos.
- mypy estricto: correcto sobre 55 archivos fuente.
- pytest: 52 aprobadas y 2 PTY omitidas por plataforma.
- `uv build`: wheel y sdist construidos correctamente.
- enlaces Markdown locales: sin destinos rotos.

Las dos pruebas PTY deben ejecutarse en Linux CI antes de marcar ese gate. Esta instantánea queda
obsoleta cuando cambie código ejecutable; la CI y los reportes del commit candidato son la evidencia
final.

## Guía de lectura

- Usuario e instalación: [`README.md`](../README.md).
- Arquitectura actual: [`architecture.md`](architecture.md).
- Configuración y precedencia: [`configuration.md`](configuration.md).
- Persistencia, respaldo y borrado: [`storage-format.md`](storage-format.md).
- Evaluación de memoria y benchmarks: [`memory-evaluation.md`](memory-evaluation.md).
- Evaluación semántica opcional: [`ragas-evaluation.md`](ragas-evaluation.md).
- Lista exacta de pendientes para publicar: [`release.md`](release.md).
- Traspaso operativo actual: [`HANDOFF.md`](HANDOFF.md).
- Mantenimiento del generador: [`model-card.md`](model-card.md),
  [`LIGHTNING_RUNBOOK.md`](LIGHTNING_RUNBOOK.md) y
  [`EXTERNAL_EVALUATION.md`](EXTERNAL_EVALUATION.md).
- Decisiones técnicas: [`adr/README.md`](adr/README.md).

## Documentos históricos

`resume-fase1.md`, `retrieval_evaluation.md` y partes del runbook de Lightning registran cómo se
obtuvo y evaluó el generador antes de construir V1. Se conservan por reproducibilidad, pero no
describen el estado global actual. Cada uno incluye una advertencia de alcance.

## Convenciones de evidencia

- **Implementado** significa que existe código y cobertura automatizada proporcional.
- **Validado** requiere una ejecución registrada sobre el entorno o dataset declarado.
- **Gate cumplido** requiere el reporte mecánico y, cuando corresponda, aprobación humana.
- **Publicado** requiere comprobar el artefacto remoto; un manifiesto local no basta.

No completar casillas ni reemplazar valores faltantes con estimaciones.

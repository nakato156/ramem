# Registro de decisiones de arquitectura

| ADR | Estado | Decisión |
|---|---|---|
| [0001](0001-explicit-state-machine.md) | aceptado | máquina de estados explícita, sin agente |
| [0002](0002-offline-retrieval-baseline.md) | reemplazado para producción | hashing + FTS como baseline de pruebas |
| [0003](0003-deny-by-default-datasets.md) | aceptado | datasets bloqueados hasta verificar procedencia |
| [0004](0004-rag-framework-selection.md) | aceptado | LanceDB y librerías especializadas para runtime |
| [0005](0005-ragas-evaluation.md) | aceptado | Ragas opcional y aislado para evaluación |

Un ADR aceptado describe una decisión, no evidencia de que todos sus gates experimentales estén
cumplidos. Las decisiones nuevas reemplazan explícitamente a las anteriores cuando hay conflicto.

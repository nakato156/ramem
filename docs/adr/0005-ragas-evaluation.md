# ADR 0005: Ragas solo como capa opcional de evaluación

- Estado: aceptado
- Fecha: 2026-07-25

## Contexto

RAMEM necesita medir si recupera la evidencia correcta y si el generador responde apoyándose en
ella. Las métricas actuales de recall y citas son reproducibles, pero una cita válida no demuestra
entailment. Implementar desde cero evaluadores semánticos, generación de afirmaciones y
experimentación aumentaría el riesgo.

## Alternativas

| Alternativa | Cobertura | Fiabilidad inicial | Coste de integración | Aislamiento runtime | Decisión |
|---|---:|---:|---:|---:|---|
| Solo métricas RAMEM | media | alta en IDs | bajo | excelente | insuficiente para semántica |
| Ragas opcional | alta | alta en IDs, calibrable en LLM | medio | excelente si se aísla | **elegida** |
| DeepEval opcional | alta | calibrable | medio | excelente si se aísla | reserva |
| Evaluadores propios completos | variable | no demostrada | alto | excelente | rechazada |

## Decisión

Adoptar Ragas de forma incremental:

1. conservar precision/recall por IDs en el runner determinista RAMEM;
2. pilotar con Ragas fidelidad, ruido y correctitud;
3. calibrar contra humanos antes de crear gates;
4. mantener entorno y dependencia separados del CLI.

El runner RAMEM conserva la autoridad sobre umbrales de release y los benchmarks externos conservan
sus evaluadores oficiales.

## Consecuencias

- Se reutiliza software especializado sin contaminar el runtime.
- Se evita depender de las métricas ID-based legacy de Ragas 0.4.3.
- Las métricas LLM tienen coste, variación y riesgo de privacidad; deben congelarse y auditarse.
- El pin menor `>=0.4.3,<0.5` se revisará explícitamente por la transición de API.
- La generación sintética solo puede alimentar development tras revisión humana.

El procedimiento operativo está en [`../ragas-evaluation.md`](../ragas-evaluation.md).

# Evaluación opcional con Ragas

Estado: decisión aceptada; integración pendiente y fuera del runtime V1.

## Decisión

Ragas se usará como capa complementaria de evaluación, instalada en un entorno separado. No
implementa la memoria de RAMEM, no reemplaza LanceDB, LongMemEval, LoCoMo ni los gates
deterministas, y no se añadirá a las dependencias normales del CLI.

La versión evaluada es Ragas 0.4.3. Se fijará inicialmente `ragas>=0.4.3,<0.5` y se utilizará la API
moderna `ragas.metrics.collections`; la API anterior está deprecada.

## Matriz de utilidad

Escala 0–10 para el estado actual de RAMEM.

| Uso | Ahora | Futuro | Decisión |
|---|---:|---:|---|
| Recall/precision por IDs de evidencia | 2 | 2 | mantener en RAMEM; API Ragas legacy |
| Fidelidad de respuesta al contexto | 8 | 9 | adoptar con calibración humana |
| Sensibilidad a recuerdos irrelevantes | 9 | 9 | adoptar con calibración humana |
| Correctitud frente a respuesta de referencia | 8 | 8 | adoptar en development |
| Validación exacta de citas `[D#]` | 3 | 3 | conservar validador RAMEM |
| Generación sintética conversacional | 4 | 7 | solo candidatos revisados |
| Runtime de recuperación/generación | 0 | 0 | excluir |
| Comparación continua de configuraciones | 7 | 9 | adoptar después del piloto |

## Métricas

### Métricas exactas fuera de Ragas

- `IDBasedContextRecall`: proporción de IDs de evidencia esperados recuperados.
- `IDBasedContextPrecision`: proporción de IDs recuperados que son evidencia esperada.

Estas métricas no requieren juez LLM y encajan con los IDs que preservan RaMem-Memory-ES,
LongMemEval y LoCoMo. Sin embargo, Ragas 0.4.3 todavía las documenta mediante
`ragas.metrics`/`SingleTurnSample`, su API legacy. RAMEM debe implementar el cociente determinista y
probarlo con casos conocidos, sin incorporar esa API deprecada. Los IDs siguen viajando en el
artefacto común para trazabilidad.

### Piloto semántico

- `Faithfulness`: soporte de las afirmaciones de la respuesta en el contexto recuperado.
- `NoiseSensitivity`: afirmaciones incorrectas inducidas por contexto relevante o irrelevante;
  menor es mejor.
- `FactualCorrectness`: correspondencia de afirmaciones con la respuesta de referencia.
- `ResponseRelevancy`: diagnóstico secundario de pertinencia, no gate aislado.

Una cita sintácticamente válida no prueba soporte semántico. A la vez, Ragas no conoce por sí mismo
el contrato `[D1]`; RAMEM debe conservar su validador exacto y, si hace falta, añadir una métrica
propia cita-afirmación.

## Contrato de datos

El adaptador debe producir por caso:

| Campo | Origen RAMEM |
|---|---|
| `user_input` | consulta final |
| `response` | respuesta del backend evaluado |
| `retrieved_contexts` | texto de chunks empaquetados, en orden |
| `retrieved_context_ids` | IDs estables de chunk/sesión/evidencia |
| `reference` | respuesta humana u oficial |
| `reference_context_ids` | IDs oficiales de evidencia |

Los campos `retrieved_context_ids` y `reference_context_ids` alimentan las métricas RAMEM; las
colecciones semánticas de Ragas usan los campos textuales aplicables. Además se debe conservar
dataset, split, hash, commit, configuración, modelo juez, revisión, prompt, semilla, resultado y
razón por ejemplo.

## Fiabilidad y español

Antes de usar una métrica con juez LLM como gate:

1. seleccionar 20–50 casos españoles estratificados y revisarlos manualmente;
2. incluir respuestas correctas, parciales, alucinadas, abstenciones y contexto distractor;
3. adaptar y revisar los prompts en español;
4. fijar proveedor, modelo, revisión, temperatura y prompt;
5. medir acuerdo con las etiquetas humanas y publicar la matriz de errores;
6. congelar la configuración antes de evaluar el holdout.

RAMEM no debe juzgarse con el mismo Gemma 1B que genera las respuestas. HHEM puede reducir el coste
de parte de la comprobación, pero también requiere validación específica en español.

## Aislamiento, privacidad y coste

- instalar Ragas en un extra o proyecto `uv` separado;
- no mezclarlo con los extras `training` o `retrieval`;
- activar caché en disco y conservar la clave de versión del juez;
- establecer `RAGAS_DO_NOT_TRACK=true`;
- no enviar conversaciones privadas a un proveedor externo sin consentimiento y política explícita;
- usar datos públicos o anonimizados para el piloto;
- revisar el grafo de dependencias antes de cada actualización menor.

Ragas 0.4.3 incorpora dependencias de LangChain en su núcleo. Su aislamiento evita contradecir
ADR-0004 y mantiene pequeño el runtime de RAMEM.

## Fases

1. **Contrato común:** exportador + precision/recall por IDs calculados por RAMEM.
2. **Piloto Ragas:** métricas semánticas sobre muestra española humana; resultados solo diagnósticos.
3. **Alineación:** ajustar prompts y demostrar acuerdo suficiente.
4. **Experimentos:** comparar ventana reciente, truncado, oráculo, recuperación y backends.
5. **Futuro:** generar casos candidatos de ruido, temporalidad y actualización, siempre revisados y
   nunca incorporados al holdout observado.

## Fuentes primarias

- [Ragas 0.4.3 en PyPI](https://pypi.org/project/ragas/)
- [Métricas disponibles](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/)
- [Context Recall](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/context_recall/)
- [Context Precision](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/context_precision/)
- [Faithfulness](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/faithfulness/)
- [Noise Sensitivity](https://docs.ragas.io/en/latest/concepts/metrics/available_metrics/noise_sensitivity/)
- [Factual Correctness](https://docs.ragas.io/en/latest/concepts/metrics/available_metrics/factual_correctness/)
- [Adaptación de idioma](https://docs.ragas.io/en/latest/howtos/customizations/metrics/metrics_language_adaptation/)
- [Caching](https://docs.ragas.io/en/stable/howtos/customizations/_caching/)
- [Migración a 0.4](https://docs.ragas.io/en/stable/howtos/migrations/migrate_from_v03_to_v04/)
- [Generación de datasets](https://docs.ragas.io/en/stable/concepts/test_data_generation/rag/)
- [Artículo original](https://arxiv.org/abs/2309.15217)

# RaMem handoff

Última actualización: 2026-07-23 (America/Lima).

## Estado ejecutivo

La fase activa es el modelo finetuneado de QA en español con respuestas citadas. El objetivo
inmediato no es completar RAG/RaMem end-to-end, sino decidir si el checkpoint finetuneado actual es
suficiente para pasar a la fase RAG o si requiere mas evaluacion/ajuste.

Decision actual: el checkpoint finetuneado queda aceptado para una evaluacion go/no-go controlada.
No hay evidencia que justifique reentrenar a ciegas. La mejora frente a Gemma 3 1B base es amplia,
consistente y estadisticamente clara en external-dev. Antes de congelarlo como modelo final, falta
validarlo mejor en fidelidad semantica, abstencion y el test externo reservado.

Resumen dedicado de fase 1:

`docs/resume-fase1.md`

Estado de repositorios al iniciar este handoff:

- Local, `origin/main` y Lightning deben apuntar al commit que contiene este handoff; verificar con
  `git status -sb` y `git rev-parse HEAD`.
- El 2026-07-23 Lightning CPU fue sincronizado y validado sin T4.

## 1. Cierre de artefactos

### Modelo fusionado canónico

Ruta:

`artifacts/export/gemma-1b-ramem-seed42/model.safetensors`

Verificación:

- Bytes: `1,999,811,208`
- SHA-256: `03a514d1dda8ddff236b6f4056cf89184e553a691ae01a0946c1aea08ca21e23`
- Formato: safetensors, BF16
- Parámetros cargados: `999,885,952`
- Vocabulario: `262,144`
- Parámetros no finitos: `0`

El modelo fue reconstruido localmente desde el modelo base autorizado y el adaptador, y produjo
exactamente el mismo tamaño y SHA-256 que el export de Lightning. Esto cierra la dependencia de los
108 MiB que faltaban en la transferencia SSH.

Archivos canónicos adicionales:

- `config.json`
- `generation_config.json`
- `chat_template.jinja`
- `tokenizer.json`
- `tokenizer_config.json`
- `special_tokens_map.json`
- `ramem_export_manifest.json`

### Adaptador final

Ruta:

`artifacts/training/gemma-1b-t4-seed42/adapter-final/adapter_model.safetensors`

Verificación:

- Bytes: `26,139,992`
- SHA-256: `cde9f18b52323eed44ad35cfe74968c131284b36c65f26d533340ecd4e55d23d`

### Material redundante

`artifacts/superseded/` conserva temporalmente:

- La transferencia parcial de Lightning.
- Un rebuild FP16 válido pero no canónico.

Ambos pueden eliminarse cuando se desee recuperar espacio. No deben utilizarse para evaluación ni
distribución. La copia BF16 canónica es la única fuente de pesos fusionados.

### Arreglo del exportador

El exportador ahora acepta:

- `device: auto | cpu | cuda`
- `dtype: auto | float16 | bfloat16`

Esto evita que un export CPU en Windows consulte una GPU oculta mediante
`CUDA_VISIBLE_DEVICES` y permite reconstrucciones BF16 deterministas.

## Evidencia disponible

### Generación interna, 256 ejemplos

| Métrica | Base | Adaptador |
|---|---:|---:|
| Exact Match | 0.0352 | 0.5469 |
| Token F1 | 0.1843 | 0.7168 |
| Cita `[D1]` | 0.8125 | 1.0000 |
| Latencia media | 5.007 s | 1.454 s |

Fuente: `artifacts/evaluation/gemma-1b-t4-seed42/summary.json`.

### MLQA español external-dev, 500 ejemplos

| Métrica | Base | Adaptador |
|---|---:|---:|
| Exact Match | 0.0360 | 0.4860 |
| Token F1 | 0.2146 | 0.6881 |
| Cita `[D1]` | 0.7100 | 1.0000 |
| IDs de cita válidos | 0.2760 | 1.0000 |
| Latencia media | 5.315 s | 1.521 s |

- Estado precomprometido: `accepted_external_dev`.
- IC bootstrap 95% del delta de Token F1: `[0.4373, 0.5091]`.
- Errores del adaptador: 243 exactos, 119 parciales altos, 55 parciales bajos y 83 sin
  solapamiento.
- El test externo final continúa reservado.

Fuente: `artifacts/evaluation/mlqa-es-external-dev-seed42/`.

### Lectura de la fase finetuning

El modelo finetuneado es viable como checkpoint de fase 1. La recomendacion tecnica es no ajustar
mas el entrenamiento todavia. El siguiente paso no debe ser otro run de training, sino una
validacion mas estricta del checkpoint actual:

- Ejecutar el external-final reservado una sola vez cuando los criterios de revision esten cerrados.
- Revisar manualmente una muestra estratificada: exactos, parciales altos, parciales bajos y sin
  solapamiento.
- Medir fidelidad semantica de respuestas y citas. La metrica actual `valid_citations = 1.0`
  verifica IDs validos, no demuestra soporte semantico de cada afirmacion.
- Probar robustez minima: sin evidencia, contexto distractor, preguntas ambiguas y casos donde debe
  abstenerse.

Si esas validaciones confirman external-dev, el checkpoint se congela y se pasa a RAG. Si fallan, el
diagnostico debe separar problemas de entrenamiento, formato de datos, citas, abstencion y
generalizacion.

### Recuperación MIRACL piloto, 54 consultas

| Métrica | BM25 | Denso 768 | RRF |
|---|---:|---:|---:|
| Recall@1 | 0.4355 | 0.5960 | 0.5405 |
| Recall@5 | 0.7981 | 0.9416 | 0.8814 |
| Recall@10 | 0.8340 | 0.9907 | 0.9182 |
| Recall@20 | 0.8893 | 0.9954 | 0.9429 |
| nDCG@10 | 0.7731 | 0.9496 | 0.8848 |
| MRR@10 | 0.8119 | 0.9537 | 0.9228 |

- EmbeddingGemma procesó 10,121 documentos en 713.6 s (`14.18 docs/s`) en la GTX 1650.
- Pico observado: aproximadamente 2.52 GiB de VRAM y 69 °C.
- RRF empeora frente al denso puro; no priorizarlo sin cambiar pesos o método de fusión.
- Es un piloto condicionado: 10,000 negativos y todos los relevantes encontrados en un shard de
  500,000 documentos. No representa el resultado final sobre MIRACL completo.

Fuente: `artifacts/evaluation/retrieval-e02-pilot/`.

## Entornos reproducibles

Los extras de `uv` son deliberadamente incompatibles:

- `uv sync --extra training`: Torch 2.5.1 + CUDA 12.1.
- `uv sync --extra retrieval`: Torch 2.6 + CUDA 12.4.

No intentar instalar `training` y `retrieval` juntos. Al cambiar de fase, ejecutar el `uv sync`
correspondiente. El repositorio fija ambas resoluciones en `uv.lock`.

## Orden de próximos pasos

### 2. Cerrar modelo finetuneado

Prioridad actual. Antes de invertir en RAG completo:

1. Definir el protocolo del external-final y dejarlo inmutable.
2. Ejecutar el test externo reservado una sola vez.
3. Hacer revision manual estratificada de external-dev y external-final.
4. Incorporar Ragas solo como capa complementaria de validacion semantica, no como reemplazo de las
   metricas deterministas.
5. Decidir:
   - congelar checkpoint y pasar a RAG, o
   - corregir finetuning/evaluacion si aparece una falla concreta.

Gates recomendados para congelar:

- External-final cercano a external-dev.
- Token F1 claramente superior a base.
- EM y F1 por encima de criterios precomprometidos.
- Citas validas por formato y muestra manual con soporte semantico aceptable.
- Casos de abstencion y ruido sin fallas sistematicas graves.

### 3. Fase posterior: E03 dimensiones de EmbeddingGemma

Ejecutar 768, 256 y 128 dimensiones sobre exactamente el mismo corpus y consultas. Esto pertenece a
la fase RAG y no debe bloquear la decision actual del modelo finetuneado.

Antes de correr:

1. Modificar el evaluador para calcular embeddings 768 una sola vez.
2. Guardar o reutilizar la matriz y truncarla a 256/128.
3. Renormalizar después del truncado.
4. Producir una tabla comparativa con calidad, bytes, latencia y memoria.

Criterio precomprometido:

- Seleccionar 256 si su pérdida relativa de nDCG@10 frente a 768 es como máximo 2 puntos.
- En caso contrario conservar 768.

Recurso: GTX 1650 local para el piloto. No requiere T4.

### 4. Fase posterior: E04 chunking

Construir fragmentos reales, preservando `docid`, `parent_id`, offsets y hashes:

1. 128 tokens con 10% de solapamiento.
2. 256 tokens con 10% de solapamiento.
3. 512 tokens con 10% de solapamiento.
4. Elegir el tamaño ganador.
5. Probar otros solapamientos solo sobre el ganador.
6. Comparar parent-child chunking como experimento final separado.

No ejecutar el producto cartesiano completo.

Recursos: CPU para preparar; GTX 1650 para pilotos.

### 5. Fase posterior: E05 top-k y presupuesto de contexto

En development únicamente:

- `top-k`: 3, 5, 8.
- Contexto: 1K, 2K, 4K tokens.

Elegir la combinación más pequeña sobre la frontera calidad/latencia. Registrar truncado, número de
chunks realmente incluidos y evidencia descartada.

### 6. Fase posterior: integracion end-to-end

Implementar y persistir:

```text
pregunta
  -> embedding de consulta
  -> índice denso
  -> top-k chunks
  -> empaquetado de contexto con IDs
  -> RaMem
  -> respuesta y citas
```

Comparaciones mínimas:

- Contexto oráculo frente a contexto recuperado.
- BM25 frente a denso.
- 768 frente a la dimensión ganadora.
- Gemma base frente al adaptador.

Métricas deterministas obligatorias:

- Recall@n, MRR y nDCG.
- Exact Match y Token F1.
- Precisión/recall de IDs citados.
- Latencia end-to-end, RAM y VRAM.
- Cobertura de respuesta y abstención.

### 7. Ragas para validar el modelo finetuneado

Decisión: adoptar Ragas como capa complementaria de validación semántica del checkpoint finetuneado
y, mas adelante, del RAG. Nunca debe reemplazar MLQA, metricas deterministas o revision humana.

Ragas es especialmente util ahora porque el actual `valid_citations` solo verifica que los IDs
emitidos sean `[D1]`; no demuestra que una afirmacion este sustentada por el texto citado.

Métricas propuestas:

- `Faithfulness`: afirmaciones respaldadas por el contexto.
- `Context Precision` y `Context Recall`.
- `Noise Sensitivity`.
- `Response Relevancy`.
- `Factual Correctness`.
- Métrica personalizada de entailment cita-afirmación.
- Métrica personalizada de abstención cuando falta evidencia.

Diseño del piloto Ragas:

1. Crear un extra aislado `ragas-eval`; no mezclarlo inicialmente con `training` o `retrieval`.
2. Usar la API moderna `ragas.metrics.collections`, no ejemplos legacy.
3. Mapear por ejemplo:
   - `user_input`
   - `response`
   - `retrieved_contexts`
   - `reference`
   - `reference_contexts` o IDs de referencia
4. Adaptar prompts a español con `adapt_instruction=True`.
5. Evaluar primero unos 100 casos estratificados de external-dev, incluyendo los peores.
6. No usar RaMem como juez de sí mismo.
7. Preferir un juez más fuerte y fijar modelo, versión, temperatura y prompt.
8. Probar HHEM-2.1-Open local como alternativa de fidelidad, pero calibrarlo en español.
9. Etiquetar manualmente una muestra y medir acuerdo humano-juez; objetivo recomendado:
   `Cohen κ >= 0.7` antes de convertir la métrica en gate.
10. Activar caché en disco y guardar scores y razones por ejemplo.

Ragas también puede generar pruebas sintéticas single-hop/multi-hop mediante un grafo de
conocimiento. Usarlas solo para robustez en development y siempre con revisión humana; nunca
reemplazar el test externo reservado.

Documentación revisada:

- <https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/>
- <https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/faithfulness/>
- <https://docs.ragas.io/en/stable/howtos/customizations/metrics/metrics_language_adaptation/>
- <https://docs.ragas.io/en/stable/howtos/customizations/_caching/>
- <https://docs.ragas.io/en/stable/concepts/test_data_generation/rag/>

### 8. Robustez del modelo finetuneado

Añadir casos revisados con:

- Sin evidencia o sin respuesta.
- Documentos contradictorios.
- Varias fuentes relevantes.
- Evidencia distribuida entre chunks.
- Preguntas ambiguas.
- Instrucciones maliciosas dentro de documentos.
- Contextos largos y distractores.

Analizar especialmente los 83/500 errores externos sin solapamiento antes de reentrenar.

### 9. Evaluacion final y release

Para el modelo finetuneado, ejecutar primero:

1. Revision manual estratificada.
2. Ragas/calibracion semantica en development.
3. Test externo reservado una sola vez.
4. Decision de congelar checkpoint.

Solo en la fase RAG, despues de congelar dimension, chunking, top-k, presupuesto y prompts:

1. Ejecutar MIRACL completo en T4.
2. Crear un release candidate inmutable.
3. No ajustar configuracion despues de observar el test.
4. Publicar pesos, tokenizer, manifiesto, hashes, configuracion, metricas y limitaciones.

## Pendientes de sincronización

Lightning CPU ya fue sincronizado el 2026-07-23. Para verificar de nuevo:

```bash
cd /teamspace/studios/this_studio/ramem
git status --short
git pull --ff-only origin main
git rev-parse HEAD
```

No es necesario volver a descargar el modelo desde Lightning: el modelo local canonico coincide
exactamente con el SHA-256 registrado alli y verificado en Lightning.

## Definición de “RaMem terminado”

RaMem puede considerarse release candidate cuando:

- El modelo y tokenizer canónicos están verificados y reproducibles.
- E03–E05 están cerrados con decisiones precomprometidas.
- El pipeline end-to-end supera gates deterministas en development externo.
- Las métricas semánticas de Ragas están calibradas contra humanos.
- Robustez y abstención tienen resultados aceptables.
- Código local, GitHub y Lightning apuntan al mismo commit.
- El test final reservado se ejecutó una sola vez sin retroajustes.

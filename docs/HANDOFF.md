# RaMem handoff

Última actualización: 2026-07-22 (America/Lima).

## Estado ejecutivo

RaMem es un prototipo viable de QA en español con recuperación y respuestas citadas. El adaptador
mejora de forma amplia y estadísticamente clara a Gemma 3 1B base en desarrollo externo. El
recuperador denso EmbeddingGemma también supera a BM25 en el piloto. Aún falta validar el sistema
end-to-end con contexto recuperado automáticamente y ejecutar una sola vez el test final reservado.

Estado de repositorios al iniciar este handoff:

- Local y `origin/main`: deben apuntar al commit que contiene este handoff; verificar con
  `git status -sb` y `git rev-parse HEAD`.
- Lightning: inaccesible por SSH (`Permission denied (publickey)`); no asumir que está actualizado.
- Cuando vuelva el Studio, ejecutar un `git pull --ff-only origin main` sin necesidad de encender T4.

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

### 2. E03: dimensiones de EmbeddingGemma

Ejecutar 768, 256 y 128 dimensiones sobre exactamente el mismo corpus y consultas.

Antes de correr:

1. Modificar el evaluador para calcular embeddings 768 una sola vez.
2. Guardar o reutilizar la matriz y truncarla a 256/128.
3. Renormalizar después del truncado.
4. Producir una tabla comparativa con calidad, bytes, latencia y memoria.

Criterio precomprometido:

- Seleccionar 256 si su pérdida relativa de nDCG@10 frente a 768 es como máximo 2 puntos.
- En caso contrario conservar 768.

Recurso: GTX 1650 local para el piloto. No requiere T4.

### 3. E04: chunking

Construir fragmentos reales, preservando `docid`, `parent_id`, offsets y hashes:

1. 128 tokens con 10% de solapamiento.
2. 256 tokens con 10% de solapamiento.
3. 512 tokens con 10% de solapamiento.
4. Elegir el tamaño ganador.
5. Probar otros solapamientos solo sobre el ganador.
6. Comparar parent-child chunking como experimento final separado.

No ejecutar el producto cartesiano completo.

Recursos: CPU para preparar; GTX 1650 para pilotos.

### 4. E05: top-k y presupuesto de contexto

En development únicamente:

- `top-k`: 3, 5, 8.
- Contexto: 1K, 2K, 4K tokens.

Elegir la combinación más pequeña sobre la frontera calidad/latencia. Registrar truncado, número de
chunks realmente incluidos y evidencia descartada.

### 5. Integración end-to-end

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

### 6. Incorporación de Ragas

Decisión: adoptar Ragas como capa complementaria de validación semántica, nunca como reemplazo de
MIRACL, MLQA, métricas deterministas o revisión humana.

Ragas es especialmente útil porque el actual `valid_citations` solo verifica que los IDs emitidos
sean `[D1]`; no demuestra que una afirmación esté sustentada por el texto citado.

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

### 7. Robustez

Añadir casos revisados con:

- Sin evidencia o sin respuesta.
- Documentos contradictorios.
- Varias fuentes relevantes.
- Evidencia distribuida entre chunks.
- Preguntas ambiguas.
- Instrucciones maliciosas dentro de documentos.
- Contextos largos y distractores.

Analizar especialmente los 83/500 errores externos sin solapamiento.

### 8. Evaluación final y release

Solo después de congelar dimensión, chunking, top-k, presupuesto y prompts:

1. Ejecutar MIRACL completo en T4.
2. Crear un release candidate inmutable.
3. Ejecutar el test externo reservado una sola vez.
4. No ajustar configuración después de observar el test.
5. Publicar pesos, tokenizer, manifiesto, hashes, configuración, métricas y limitaciones.

## Pendientes de sincronización

Cuando Lightning vuelva a estar accesible:

```bash
cd /teamspace/studios/this_studio/ramem
git status --short
git pull --ff-only origin main
git rev-parse HEAD
```

No es necesario volver a descargar el modelo desde Lightning: el modelo local canónico coincide
exactamente con el SHA-256 previamente registrado allí.

## Definición de “RaMem terminado”

RaMem puede considerarse release candidate cuando:

- El modelo y tokenizer canónicos están verificados y reproducibles.
- E03–E05 están cerrados con decisiones precomprometidas.
- El pipeline end-to-end supera gates deterministas en development externo.
- Las métricas semánticas de Ragas están calibradas contra humanos.
- Robustez y abstención tienen resultados aceptables.
- Código local, GitHub y Lightning apuntan al mismo commit.
- El test final reservado se ejecutó una sola vez sin retroajustes.

# Próximos experimentos de RAMEM V1

Estado: backlog posterior a la implementación funcional. Todo ajuste se selecciona con development;
el holdout permanece cerrado hasta congelar el candidato.

## Prioridad 1: validar evaluadores

1. Ampliar RaMem-Memory-ES más allá de los 10 casos smoke.
2. Revisar de forma independiente consultas, respuestas, evidencia exacta y categoría.
3. Incluir extracción, cruce entre sesiones, actualización, temporalidad, abstención, ruido e
   instrucciones maliciosas dentro de memoria.
4. Verificar métricas deterministas con casos de control conocidos.
5. Fijar commits y licencias de LongMemEval/LoCoMo.

## E03: dimensiones de EmbeddingGemma

Calcular embeddings 768 una vez, truncar a 256/128 y renormalizar. Comparar Recall@1/5/10/20,
precision@k, nDCG@10, MRR@10, p50/p95, RAM e índice sobre el mismo corpus y orden.

Regla precomprometida: conservar 256 si la pérdida relativa de nDCG@10 frente a 768 es como máximo
dos puntos y todos los mínimos de recall por categoría se mantienen. En caso contrario, usar 768.

## E04: chunking conversacional

Comparar 128, 256 y 512 tokens preservando intercambios, IDs, roles y offsets. Usar un mensaje
adyacente de solapamiento; si un mensaje excede el límite, dividirlo sin perder procedencia. Probar
variantes de solapamiento solo sobre el tamaño ganador. No aplicar splitters documentales que corten
fronteras conversacionales.

## E05: top-k y presupuesto

En development:

- `top-k`: 3, 5 y 8;
- memoria: 1K, 2K y 4K tokens;
- ventana reciente fija salvo experimento separado.

Elegir la configuración más pequeña en la frontera calidad/latencia. Registrar resultados
recuperados, incluidos y descartados; `top-k` de recuperación y número de chunks empaquetados no son
necesariamente iguales.

## E06: comparación end-to-end

Ejecutar con el mismo generador y casos:

- solo ventana reciente;
- historial truncado;
- contexto oráculo;
- memoria recuperada;
- Transformers BF16;
- GGUF Q4_K_M.

Medir exactitud oficial, Token F1 cuando aplique, recall/precision de evidencia, citas, fidelidad
revisada, tokens de contexto, latencia, RAM/VRAM, primer token y tokens/s.

## E07: piloto Ragas

1. Instalar Ragas en entorno aislado.
2. Validar precision/recall por IDs del runner RAMEM contra cálculos manuales.
3. Etiquetar 20–50 casos españoles estratificados.
4. Comparar Faithfulness, NoiseSensitivity y FactualCorrectness contra humanos.
5. Congelar juez y prompts solo si el acuerdo es suficiente.

La generación sintética puede proponer casos de development después de E07, pero todos se revisan y
ninguno se incorpora al holdout observado. Véase
[`ragas-evaluation.md`](ragas-evaluation.md).

## E08: escala y durabilidad

Con 100.000 mensajes en la máquina local de referencia:

- medir p50/p95 de recuperación y empaquetado;
- reiniciar con trabajos `running` y confirmar su retorno a `pending`;
- cancelar generación y simular fallo del backend;
- reconstruir LanceDB desde SQLite;
- borrar sesiones y verificar ausencia en SQLite, FTS y LanceDB;
- registrar disco, RAM, CPU y duración del rebuild.

Cada ejecución conserva configuración resuelta, commit, entorno, semilla, hashes de datos, metadatos
del índice, predicciones y timings. El test final se ejecuta una sola vez tras congelar todo lo
anterior.

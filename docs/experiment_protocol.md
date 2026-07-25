# Experiment protocol

Selection uses development data only. Trained modules use seeds 13, 42, and 2026, while paired
comparisons preserve example order. Each run must retain resolved configuration, code revision,
environment, dataset hashes, timings, predictions, and per-example errors. Release tests are run
only after a configuration is frozen.

CI validates schemas, policies and pure infrastructure without downloading benchmark rows. Scientific
metrics are produced only from versioned public or manually reviewed RaMem datasets downloaded by an
explicit experiment command.

## Memoria conversacional

Las comparaciones de ventana reciente, historial truncado, oráculo y recuperación usan exactamente
los mismos casos, generador, prompt y orden. Registrar IDs recuperados, chunks empaquetados, tokens
descartados, citas, latencia de recuperación, tiempo al primer token y salida completa. No seleccionar
dimensión, chunking, top-k y presupuesto sobre el holdout.

## Evaluadores aprendidos

Una métrica con juez LLM registra proveedor, modelo y revisión, prompt, ejemplos adaptados, idioma,
temperatura, caché y coste. Antes de usarla como gate debe compararse con etiquetas humanas y
conservar desacuerdos. Actualizar el juez o Ragas crea una nueva serie experimental; no se mezclan
scores entre versiones.

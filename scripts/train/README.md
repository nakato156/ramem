# Training

El entrypoint mantenido es `ramem-train`. El checkpoint de V1 ya existe: no ejecutar otro
entrenamiento sin un fallo diagnosticado del generador y un protocolo precomprometido. Véase
`docs/LIGHTNING_RUNBOOK.md`; usa filas reales descargadas de SQuAD-es y un checkpoint Gemma gated.

El entrenamiento no forma parte de la instalación ni del uso normal de `ramem-cli`.

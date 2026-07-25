# ADR 0003: datasets bloqueados por defecto

- Estado: aceptado

Todo dataset público comienza bloqueado. Un loader solo puede acceder después de registrar revisión,
licencia, split autorizado y checksum. El entrenamiento recibe un error explícito ante fuentes
reservadas para test. CI no depende de datasets externos; las filas reales se descargan únicamente
mediante comandos explícitos. La misma regla aplica a datasets generados con Ragas: solo entran en
development tras revisión y nunca en un holdout ya observado.

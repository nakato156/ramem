# Evaluation

Estos scripts mantienen la evaluación histórica del generador. MLQA external-dev ya produjo la
evidencia registrada en `docs/resume-fase1.md`; repetirlo requiere una nueva pregunta experimental y
un commit congelado.

`lightning_t4_final_holdout.sh` is deliberately locked and must not run until the release candidate
is frozen. See `docs/EXTERNAL_EVALUATION.md`.

La evaluación de memoria V1 usa `ramem benchmark`, los adaptadores oficiales y
`ramem-release-gates`; véase `docs/memory-evaluation.md`.

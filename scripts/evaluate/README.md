# Evaluation

`lightning_t4_external_dev.sh` is the next approved run. It downloads MLQA Spanish validation,
audits exact overlap with SQuAD-es train, performs the paired base/adapter evaluation, and writes the
precommitted acceptance analysis.

`lightning_t4_final_holdout.sh` is deliberately locked and must not run until the release candidate
is frozen. See `docs/EXTERNAL_EVALUATION.md`.

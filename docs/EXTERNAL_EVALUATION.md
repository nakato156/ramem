# External generation evaluation protocol

## Purpose

The internal SQuAD-es slice measures task fit but not external generalization. Model selection now
uses all 500 rows from `facebook/mlqa`, configuration `mlqa.es.es`, split `validation`. The official
MLQA test split is never downloaded by the automated development workflow.

The comparison is paired: base Gemma and the seed-42 RaMem adapter receive identical examples in
identical deterministic order. Generation is greedy with a 1,024-token input limit and 64 new-token
limit. Each prediction records its prompt, reference, token counts, latency, exact match, token F1,
presence of `[D1]`, and whether every emitted citation is valid.

## Precommitted acceptance gate

These thresholds were committed before observing MLQA results:

- exact match at least 0.25;
- token F1 at least 0.45;
- adapter-minus-base token F1 at least +0.10;
- `[D1]` rate at least 0.98;
- valid-citations-only rate at least 0.98;
- mean adapter latency at most 3 seconds on the Lightning T4.

The analysis also reports paired-bootstrap 95% confidence intervals with 10,000 resamples and
retains the 25 lowest-F1 adapter predictions. A failed gate produces `needs_review`; it does not
silently trigger more training.

## One-command Lightning execution

After switching the persistent Studio to a T4:

```bash
cd /teamspace/studios/this_studio/ramem
git pull --ff-only origin main
tmux new-session -d -s ramem-external-eval \
  'bash scripts/evaluate/lightning_t4_external_dev.sh'
tail -f artifacts/evaluation/t4-external-dev.log
```

Expected runtime is roughly one hour based on the earlier 256-row paired evaluation, or about 0.30
credits at 0.30 credits/hour. The dataset download and overlap audit are CPU work performed inside
the same script before model loading.

Successful output:

```text
artifacts/evaluation/mlqa-es-external-dev-seed42/resolved_config.json
artifacts/evaluation/mlqa-es-external-dev-seed42/predictions-base.jsonl
artifacts/evaluation/mlqa-es-external-dev-seed42/predictions-adapter.jsonl
artifacts/evaluation/mlqa-es-external-dev-seed42/summary.json
artifacts/evaluation/mlqa-es-external-dev-seed42/analysis.json
artifacts/evaluation/mlqa-es-external-dev-seed42/worst-cases-adapter.jsonl
artifacts/evaluation/mlqa-es-external-dev-seed42/external-dev-report.md
```

## Final holdout lock

XQuAD Spanish validation is a secondary, SQuAD-domain holdout. It must run only after the model,
prompt, decoding, acceptance thresholds, and code revision are frozen. Its downloader requires
`--release-test`; its wrapper additionally requires `RAMEM_RELEASE_CANDIDATE_FROZEN=yes` and refuses
to overwrite an existing output directory. Do not use its result to tune the candidate.

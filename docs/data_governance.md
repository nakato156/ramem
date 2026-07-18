# Data governance

Dataset use is deny-by-default. `data/datasets_manifest.yaml` records provenance, license status,
authorized splits, expected hashes, and intended function before any download. Entries with an
unverified license or hash remain blocked. Test-only data must never be consumed by training code.

Raw data is immutable. Derived artifacts record source hashes. User secrets, sensitive inferred
attributes, and assistant-generated claims are never persisted as facts. Persistent personal data
requires explicit consent and keeps source, time, confidence, and supersession history.

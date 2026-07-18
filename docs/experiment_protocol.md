# Experiment protocol

Selection uses development data only. Trained modules use seeds 13, 42, and 2026, while paired
comparisons preserve example order. Each run must retain resolved configuration, code revision,
environment, dataset hashes, timings, predictions, and per-example errors. Release tests are run
only after a configuration is frozen.

CI validates schemas, policies and pure infrastructure without downloading benchmark rows. Scientific
metrics are produced only from versioned public or manually reviewed RaMem datasets downloaded by an
explicit experiment command.

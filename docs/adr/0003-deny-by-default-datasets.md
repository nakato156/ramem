# ADR 0003: Deny-by-default datasets

Status: accepted

Every public dataset begins blocked. A loader may access it only after its revision, license,
authorized split and checksum are recorded. Training callers receive an explicit error for
test-only sources. CI has no dataset dependency; real rows are downloaded only by explicit commands.

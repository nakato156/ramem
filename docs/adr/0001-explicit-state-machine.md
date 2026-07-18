# ADR 0001: Explicit state machine

Status: accepted

Use a typed, explicit state machine in the core instead of an agent framework. This keeps module
boundaries measurable, makes traces reproducible, and permits one-variable-at-a-time ablations.

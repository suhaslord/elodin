# Xronos × Elodin phase 0

The first thing to prove is simple: the same frozen Elodin state trace must produce the same Xronos command trace.

Install Xronos in the Ubuntu environment:

```bash
python -m pip install -U xronos
```

Run the replay gate from this directory:

```bash
python replay_check.py sample_states.jsonl
```

Only after replay passes should `XronosBridge` be called from an Elodin `pre_step` callback and its command written to an `external_control` component.

This matches the Elodin pattern already used in the Voyager work: read components from `StepContext`, compute/receive an external command, then write the command back for the next physics update.

The scalar PD controller is intentionally tiny. It tests the runtime bridge, not spacecraft-control quality. Once the bridge is stable, replace the scalar state/command with the real vehicle state vector and actuator model.

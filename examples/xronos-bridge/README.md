# Xronos → Elodin stepping prototype

This prototype tests the smallest useful integration boundary between Xronos and Elodin:

1. Xronos owns the event/timing loop.
2. A Xronos reactor calls exactly one `Elodin` simulation step per timer event.
3. Elodin state is read back with `get_state` and published into the Xronos reactor graph.
4. A second reactor consumes the state and records a Xronos metric.

The shape intentionally mirrors Xronos's Webots example, where a `Simulator` reactor advances the external simulator and publishes a `done_step` event before the rest of the reactor network executes.

## Files

- `elodin_ball.py` — minimal deterministic Elodin ball simulation adapted from the existing `examples/ball` example.
- `main.py` — Xronos reactor wrapper that steps Elodin and publishes ball height.

## Run

From this directory, in an environment where both packages are installed:

```bash
pip install xronos
python main.py
```

The prototype runs 600 Elodin steps at a 120 Hz model timestep. Every 60 ticks the observer prints the ball height, while all height samples are also recorded through a Xronos metric.

## What this proves / does not prove

This is deliberately a narrow first pass. It exercises the core control boundary needed for a larger bridge without coupling Xronos to Elodin internals: Xronos schedules; Elodin advances; state returns through ports.

It does **not** yet cover bidirectional actuator commands, real-time pacing guarantees, telemetry schema mapping, or validation against a standalone Elodin trajectory. Those are the next useful checks before attempting a larger CubeSat/Voyager-style demo.

## Next validation

A clean validation should compare the ball `world_pos` sequence from:

- standalone Elodin stepping, and
- the same Elodin model stepped through Xronos.

For the same initial condition and number of steps, the trajectories should match within floating-point tolerance. After that, add a Xronos input port that changes a force/control component in Elodin to demonstrate bidirectional coupling.

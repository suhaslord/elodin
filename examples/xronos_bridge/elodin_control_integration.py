#!/usr/bin/env python3
"""Real Xronos -> Elodin external_control -> force-system smoke test."""

from __future__ import annotations

import sys
import typing as ty
from dataclasses import field

import elodin as el
import jax
import jax.numpy as jnp
import numpy as np

from elodin_bridge import XronosBridge


ControlCommand = ty.Annotated[
    jax.Array,
    el.Component(
        "control_command",
        el.ComponentType(el.PrimitiveType.F64, (1,)),
        metadata={"external_control": "true"},
    ),
]


@el.dataclass
class ControllerState(el.Archetype):
    control_command: ControlCommand = field(
        default_factory=lambda: jnp.array([0.0], dtype=jnp.float64)
    )


@el.map
def apply_external_control(command: ControlCommand, force: el.Force) -> el.Force:
    # Interpret the returned scalar controller command as +X body force (N).
    return force + el.SpatialForce(
        linear=jnp.array([command[0], 0.0, 0.0], dtype=jnp.float64)
    )


def main() -> None:
    world = el.World()
    world.spawn([el.Body(), ControllerState()], name="vehicle")

    bridge = XronosBridge([sys.executable, "xronos_controller.py"])
    commands: list[float] = []
    x_velocities: list[float] = []

    def post_step(tick: int, ctx: el.StepContext) -> None:
        reads = ctx.component_batch_operation(
            reads=["vehicle.world_pos", "vehicle.world_vel"]
        )
        pos = np.asarray(reads["vehicle.world_pos"], dtype=float).reshape(-1)
        vel = np.asarray(reads["vehicle.world_vel"], dtype=float).reshape(-1)

        # SpatialTransform is quaternion[0:4] + translation[4:7]; SpatialMotion
        # is angular[0:3] + linear[3:6].  Feed X position/velocity to Xronos.
        position_x = float(pos[4])
        velocity_x = float(vel[3])
        command = bridge.command(tick, position_x, velocity_x)
        commands.append(command)
        x_velocities.append(velocity_x)

        ctx.component_batch_operation(
            writes={"vehicle.control_command": np.asarray([command], dtype=np.float64)}
        )

    try:
        system = el.six_dof(sys=apply_external_control)
        world.run(
            system,
            simulation_rate=100.0,
            max_ticks=6,
            post_step=post_step,
            start_timestamp=0,
        )
    finally:
        bridge.close()

    if len(commands) < 2:
        raise RuntimeError(f"Expected multiple lockstep commands, got {len(commands)}")
    if not np.all(np.isfinite(commands)):
        raise RuntimeError("Xronos returned a non-finite command")

    print("real Xronos commands:", commands)
    print("Elodin x velocities:", x_velocities)
    print("external_control component: vehicle.control_command")
    print("force system: apply_external_control -> el.six_dof")
    print("XRONOS_ELODIN_EXTERNAL_CONTROL_PASS")


if __name__ == "__main__":
    main()

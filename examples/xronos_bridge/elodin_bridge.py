#!/usr/bin/env python3
"""Synchronous JSON-lines subprocess bridge for Elodin -> Xronos -> Elodin."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Sequence


class XronosBridge:
    def __init__(self, command: Sequence[str]) -> None:
        self._proc = subprocess.Popen(
            list(command),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            text=True,
            bufsize=1,
        )
        if self._proc.stdin is None or self._proc.stdout is None:
            raise RuntimeError("Failed to create controller pipes")

    def command(self, tick: int, position: float, velocity: float) -> float:
        if self._proc.poll() is not None:
            raise RuntimeError(f"Xronos controller exited with {self._proc.returncode}")

        request = {
            "tick": int(tick),
            "position": float(position),
            "velocity": float(velocity),
        }
        self._proc.stdin.write(json.dumps(request, separators=(",", ":")) + "\n")
        self._proc.stdin.flush()

        line = self._proc.stdout.readline()
        if line == "":
            raise RuntimeError("Xronos controller closed stdout")

        response = json.loads(line)
        if int(response["tick"]) != int(tick):
            raise RuntimeError(
                f"Lockstep violation: sent tick {tick}, received tick {response['tick']}"
            )
        return float(response["command"])

    def close(self) -> None:
        if self._proc.poll() is None:
            if self._proc.stdin is not None:
                self._proc.stdin.close()
            try:
                self._proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._proc.terminate()
                self._proc.wait(timeout=2)

    def __enter__(self) -> "XronosBridge":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


# Elodin wiring pattern (adapt names to the actual model):
#
# bridge = XronosBridge(["python", "xronos_controller.py"])
#
# def pre_step(tick: int, ctx: el.StepContext) -> None:
#     pos = np.asarray(ctx.read_component("vehicle.world_pos"), dtype=float)
#     vel = np.asarray(ctx.read_component("vehicle.world_vel"), dtype=float)
#     u = bridge.command(tick, position=float(pos[4]), velocity=float(vel[3]))
#     ctx.write_component("vehicle.control_command", np.asarray([u], dtype=np.float64))
#
# `vehicle.control_command` should be an Elodin component marked external_control=true.

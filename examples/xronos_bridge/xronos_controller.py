#!/usr/bin/env python3
"""Minimal Xronos controller process for an Elodin lockstep bridge.

stdin  -> {"tick": int, "position": float, "velocity": float}
stdout -> {"tick": int, "command": float}
"""

from __future__ import annotations

from collections.abc import Callable
import json

import xronos
import xronos.lib


class PDController(xronos.Reactor):
    state = xronos.InputPortDeclaration[str]()

    def __init__(self, kp: float, kd: float, limit: float) -> None:
        super().__init__()
        self.kp = float(kp)
        self.kd = float(kd)
        self.limit = abs(float(limit))

    @xronos.reaction
    def on_state(self, ctx: xronos.ReactionContext) -> Callable[[], None]:
        trigger = ctx.add_trigger(self.state)

        def handler() -> None:
            msg = json.loads(trigger.get())
            tick = int(msg["tick"])
            position = float(msg["position"])
            velocity = float(msg["velocity"])

            command = -(self.kp * position + self.kd * velocity)
            command = max(-self.limit, min(self.limit, command))
            print(
                json.dumps({"tick": tick, "command": command}, separators=(",", ":")),
                flush=True,
            )

        return handler


def main() -> None:
    env = xronos.Environment()
    source = env.create_reactor("stdin", xronos.lib.ConsoleInput, lambda line: line)
    controller = env.create_reactor("controller", PDController, 0.08, 0.25, 1.0)
    env.connect(source.output, controller.state)
    env.execute()


if __name__ == "__main__":
    main()

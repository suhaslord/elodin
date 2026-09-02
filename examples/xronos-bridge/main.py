import datetime
from collections.abc import Callable

import xronos

from elodin_ball import SIM_TIME_STEP, make_sim


class ElodinSimulator(xronos.Reactor):
    z = xronos.OutputPortDeclaration[float]()
    done_step = xronos.OutputPortDeclaration[int]()

    _step_timer = xronos.ProgrammableTimerDeclaration[None]()

    def __init__(self, steps: int = 600) -> None:
        super().__init__()
        self._sim = make_sim()
        self._steps = steps
        self._tick = 0

    @xronos.reaction
    def on_startup(self, ctx: xronos.ReactionContext) -> Callable[[], None]:
        _ = ctx.add_trigger(self.startup)
        timer = ctx.add_effect(self._step_timer)

        return lambda: timer.schedule(None, datetime.timedelta(0))

    @xronos.reaction
    def on_step(self, ctx: xronos.ReactionContext) -> Callable[[], None]:
        _ = ctx.add_trigger(self._step_timer)
        timer = ctx.add_effect(self._step_timer)
        shutdown = ctx.add_effect(self.shutdown)
        z_effect = ctx.add_effect(self.z)
        tick_effect = ctx.add_effect(self.done_step)

        def handler() -> None:
            self._sim.step(1)
            world_pos = self._sim.get_state(
                component_name="world_pos", entity_name="ball"
            )
            z_effect.set(float(world_pos[6]))
            tick_effect.set(self._tick)

            self._tick += 1
            if self._tick >= self._steps:
                shutdown.trigger_shutdown()
            else:
                timer.schedule(None, datetime.timedelta(seconds=SIM_TIME_STEP))

        return handler


class HeightObserver(xronos.Reactor):
    z = xronos.InputPortDeclaration[float]()
    tick = xronos.InputPortDeclaration[int]()

    height = xronos.MetricDeclaration("ball height")

    @xronos.reaction
    def observe(self, ctx: xronos.ReactionContext) -> Callable[[], None]:
        z_trigger = ctx.add_trigger(self.z)
        tick_trigger = ctx.add_trigger(self.tick)
        height = ctx.add_effect(self.height)

        def handler() -> None:
            z = z_trigger.get()
            height.record(z)
            if tick_trigger.is_present() and tick_trigger.get() % 60 == 0:
                print(f"tick={tick_trigger.get():4d} z={z:.3f} m")

        return handler


def main() -> None:
    env = xronos.Environment()

    simulator = env.create_reactor("elodin", ElodinSimulator)
    observer = env.create_reactor("observer", HeightObserver)

    env.connect(simulator.z, observer.z)
    env.connect(simulator.done_step, observer.tick)

    env.execute()


if __name__ == "__main__":
    main()

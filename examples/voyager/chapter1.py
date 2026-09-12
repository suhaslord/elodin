from pathlib import Path

import elodin as el
import jax.numpy as jnp
import spiceypy as spice
from jax.numpy import linalg as la


spice_dir = Path(__file__).resolve().parent / "nasa_spice_data"

for kernel in [
    spice_dir / "naif0012.tls",
    spice_dir / "de440.bsp",
    spice_dir / "Voyager_1.a54206u_V0.2_merged.bsp",
]:
    spice.furnsh(str(kernel))

start_time = "1979-02-22T00:00:00"
end_time = "1979-02-28T00:00:00"

start_et = spice.utc2et(start_time)

state, _ = spice.spkezr(
    "VOYAGER 1",
    start_et,
    "ECLIPJ2000",
    "NONE",
    "SUN",
)

start_pos = jnp.array(state[:3]) * 1000
start_vel = jnp.array(state[3:]) * 1000

G = 6.6743e-11
sun_mass = 1.9885e30
voyager_mass = 825.0
time_step = 3600.0
steps = 144

world = el.World()

sun = world.spawn(
    [
        el.Body(
            world_pos=el.WorldPos(linear=jnp.array([0.0, 0.0, 0.0])),
            world_vel=el.WorldVel(linear=jnp.array([0.0, 0.0, 0.0])),
            inertia=el.Inertia(sun_mass),
        )
    ],
    name="sun",
)

voyager = world.spawn(
    [
        el.Body(
            world_pos=el.WorldPos(linear=start_pos),
            world_vel=el.WorldVel(linear=start_vel),
            inertia=el.Inertia(voyager_mass),
        )
    ],
    name="voyager1",
)

GravityEdge = el.Annotated[
    el.Edge,
    el.Component("gravity_edge", el.ComponentType.Edge),
]


@el.dataclass
class GravityConstraint(el.Archetype):
    edge: GravityEdge

    def __init__(self, voyager_id, sun_id):
        self.edge = GravityEdge(voyager_id, sun_id)


world.spawn(
    GravityConstraint(voyager, sun),
    name="voyager_to_sun",
)


@el.system
def gravity(
    graph: el.GraphQuery[GravityEdge],
    query: el.Query[el.WorldPos, el.Inertia],
) -> el.Query[el.Force]:
    def gravity_force(force, voyager_pos, voyager_inertia, sun_pos, sun_inertia):
        r = voyager_pos.linear() - sun_pos.linear()
        distance = la.norm(r)

        force_amount = (
            G
            * sun_inertia.mass()
            * voyager_inertia.mass()
            * r
            / distance**3
        )

        return el.Force(linear=force.force() - force_amount)

    return graph.edge_fold(
        left_query=query,
        right_query=query,
        return_type=el.Force,
        init_value=el.Force(),
        fold_fn=gravity_force,
    )


def post_step(tick, ctx):
    if tick != steps - 1:
        return

    sim_pos = ctx.read_component("voyager1.world_pos")[4:7]

    end_et = spice.utc2et(end_time)
    truth, _ = spice.spkezr(
        "VOYAGER 1",
        end_et,
        "ECLIPJ2000",
        "NONE",
        "SUN",
    )

    truth_pos = jnp.array(truth[:3]) * 1000
    error_km = la.norm(sim_pos - truth_pos) / 1000

    print("Final position error:", float(error_km), "km")


system = el.six_dof(sys=gravity)

world.run(
    system,
    simulation_rate=1 / time_step,
    max_ticks=steps,
    post_step=post_step,
    db_path="dbs/chapter1",
    interactive=False,
)

print()
print("Chapter 1 finished")
print("Start:", start_time)
print("End:", end_time)
print("Steps:", steps)

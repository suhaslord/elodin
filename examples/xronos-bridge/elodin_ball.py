import elodin as el
import jax
import jax.numpy as jnp

SIM_TIME_STEP = 1.0 / 120.0
BOUNCINESS = 0.85


def make_world() -> el.World:
    world = el.World()
    world.spawn(
        el.Body(
            world_pos=el.SpatialTransform(
                linear=jnp.array([0.0, 0.0, 6.0])
            )
        ),
        name="ball",
    )
    return world


@el.map
def gravity(force: el.Force, inertia: el.Inertia) -> el.Force:
    return force + el.SpatialForce(
        linear=jnp.array([0.0, 0.0, -9.81]) * inertia.mass()
    )


@el.map
def bounce(pos: el.WorldPos, vel: el.WorldVel) -> el.WorldVel:
    return jax.lax.cond(
        jax.lax.max(pos.linear()[2], vel.linear()[2]) < 0.0,
        lambda _: el.SpatialMotion(
            linear=vel.linear()
            * jnp.array([1.0, 1.0, -1.0])
            * BOUNCINESS
        ),
        lambda _: vel,
        operand=None,
    )


def make_sim():
    world = make_world()
    system = bounce | el.six_dof(sys=gravity)
    return world.to_jax(system, simulation_rate=1.0 / SIM_TIME_STEP)

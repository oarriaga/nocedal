from functools import partial
from collections import namedtuple
import jax
import jax.numpy as jp


def _initialize_velocities(manifold, point):
    return point / jp.linalg.norm(point)


def _compute_inertia(manifold, points, old_points, velocities):
    return manifold.transport(old_points, points, velocities)


def compute_inertia_factor(iteration, total_iterations, min_value=0.4):
    # linearly decrease from 0.9 to 0.4 from 0 to total_iterations.
    return min_value + (0.5 * (1 - (iteration / total_iterations)))


def multiplicative_noise(key, factor, shape):
    return factor * jax.random.uniform(key, shape=shape)


def _compute_regrets(manifold, regret, key, points, best_points):
    regret = multiplicative_noise(key, regret, points.shape)
    return manifold.scale(regret, manifold.log(points, best_points))


def _compute_socials(manifold, social, key, points, best_global_point):
    social = multiplicative_noise(key, social, points.shape)
    return manifold.scale(social, manifold.log(points, best_global_point))


def compute_elite(losses, points):
    elite_arg = jp.argmin(losses)
    elite = points[elite_arg]
    return elite


def _move_point(manifold, point, velocity):
    return manifold.retract(point, velocity)


def compute_velocity(inertia, regrets, socials):
    return inertia + regrets + socials


def update(select_A, values_A, values_B):
    if len(values_A.shape) == 2:
        select_A = jp.expand_dims(select_A, axis=-1)
    return jp.where(select_A, values_A, values_B)


SwarmFields = ['key', 'points', 'velocities', 'losses', 'best_points', 'elite']
SwarmState = namedtuple('SwarmState', SwarmFields)


def kernel(manifold, num_steps, compute_loss, regret=1.4, social=1.4):
    # compute_inertia = partial(_compute_inertia, manifold)
    compute_regrets = partial(_compute_regrets, manifold, regret)
    compute_socials = partial(_compute_socials, manifold, social)
    move_points = jax.vmap(partial(_move_point, manifold))

    def compute_velocities(key, arg, state):
        key0, key1 = jax.random.split(key)
        w = compute_inertia_factor(arg, num_steps)
        inertia = w * state.velocities
        regrets = compute_regrets(key0, state.points, state.best_points)
        socials = compute_socials(key1, state.points, state.elite)
        new_velocities = compute_velocity(inertia, regrets, socials)
        return new_velocities

    @jax.jit
    def apply(state, arg):
        key, new_key = jax.random.split(state.key)
        new_velocities = compute_velocities(key, arg, state)
        new_points = move_points(state.points, new_velocities)
        new_losses = compute_loss(new_points)
        new_is_better = new_losses < state.losses
        new_best_points = update(new_is_better, new_points, state.best_points)
        new_losses = update(new_is_better, new_losses, state.losses)
        elite = compute_elite(new_losses, new_best_points)
        return SwarmState(new_key, new_points, new_velocities,
                          new_losses, new_best_points, elite), None
    return apply


def initialize(key, manifold, points, compute_loss):
    velocities = jax.vmap(partial(_initialize_velocities, manifold))(points)
    best_points = jp.copy(points)
    losses = compute_loss(points)
    elite = compute_elite(losses, points)
    return SwarmState(key, points, velocities, losses, best_points, elite)


def optimize(kernel, num_steps, manifold, compute_loss, key, points):
    initial_state = initialize(key, manifold, points, compute_loss)
    x = jp.arange(num_steps)
    state = jax.lax.scan(kernel, initial_state, x)[0]
    return state.best_points, state.elite


def SwarmOptimizer(manifold, compute_loss, num_steps, regret=1.4, social=1.4):
    one_step = kernel(manifold, num_steps, compute_loss, regret, social)
    return partial(optimize, one_step, num_steps, manifold, compute_loss)

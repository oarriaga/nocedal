from jax import grad
import jax.numpy as jp
import jax


def direct_search(loss, loss_with_barrier, weights,
                  slacks, multipliers, barrier, num_weights, num_inequalities):
    compute_grads = jax.grad(loss_with_barrier, [0, 1, 2])
    grads = compute_grads(weights, slacks, multipliers, barrier)
    grad_weights, grad_slacks, grad_multipliers = grads
    gradient = [grad_weights, grad_slacks, -grad_multipliers]
    gradient = jp.concatenate(gradient, axis=0)
    hessian = compute_hessian(loss, weights, slacks, multipliers, barrier)
    gradient_T = - gradient.reshape((gradient.size, 1))
    search_direction = jp.linalg.solve(hessian, gradient_T)
    search_direction = search_direction.reshape((gradient.size,))
    return search_direction.at[num_weights + num_inequalities:].set(
        -search_direction[num_weights + num_inequalities:])


def compute_hessian(loss, weights, slacks, multipliers, barrier):
    _compute_hessian = jax.hessian(loss, [0, 1, 2])
    hessian = _compute_hessian(weights, slacks, multipliers, barrier)
    # this can be fixed with set value API

    # change signs for upper row
    upper_L = hessian[0, 0]
    upper_M = hessian[0, 1]
    upper_R = hessian[0, 2]
    upper_part = jp.concatenate([upper_L, upper_M, -upper_R], axis=1)

    # change signs for middle row
    middle_L = hessian[1, 0]
    middle_M = hessian[1, 1]
    middle_R = hessian[1, 2]
    middle_part = jp.concatenate([middle_L, middle_M, -middle_R], axis=1)

    # change signs for lower row
    lower_L = hessian[2, 0]
    lower_M = hessian[2, 1]
    lower_R = hessian[2, 2]
    lower_part = jp.concatenate([-lower_L, -lower_M, lower_R], axis=1)
    return jp.concatenate([upper_part, middle_part, lower_part], axis=0)


def concatenate_constraints(weights, slacks, equality_constraints,
                            inequality_constraints, num_equalities,
                            num_inequalities):
    constraints = jp.zeros(num_equalities + num_inequalities)

    for arg in range(num_equalities):
        constraints = (constraints.at[arg].set(equality_constraints[arg](weights)))

    for arg in range(num_inequalities):
        inequality_val = inequality_constraints[arg](weights) - slacks[arg]
        inequality_position = arg + arg
        constraints = constraints.at[inequality_position].set(inequality_val)
    return constraints

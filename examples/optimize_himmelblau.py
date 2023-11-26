import jax
from nocedal.swarm import SwarmOptimizer
from nocedal.manifolds import euclidean

import numpy as np
from matplotlib import cm
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt


def plot(function, points=None, x_min=-5, y_min=-5,
         x_max=5, y_max=5, max_level=3.0):
    x = np.arange(x_min, x_max, 0.01)
    y = np.arange(y_min, y_max, 0.01)
    x, y = np.meshgrid(x, y)
    X = np.concatenate([np.reshape(x, (-1, 1)), np.reshape(y, (-1, 1))], 1)
    z = function(X).reshape(x.shape)
    levels = 10**np.arange(0.0, max_level, 0.1)
    figure, axis = plt.subplots()
    axis.contour(x, y, z, norm=LogNorm(), levels=levels, cmap='viridis')
    if points is not None:
        x = points[:, 0]
        y = points[:, 1]
        axis.plot(x, y, 'r*')
    figure.colorbar(cm.ScalarMappable(cmap='viridis'))
    plt.show()


def HIMMELBLAU(a=2, b=100):
    def apply(x):
        return (x[:, 0]**2 + x[:, 1] - 11)**2 + (x[:, 0] + x[:, 1]**2 - 7)**2
    return apply


num_points = 1000
key = jax.random.PRNGKey(888)
initial_points = jax.random.uniform(key, (num_points, 2), minval=-5, maxval=5)
compute_loss = HIMMELBLAU()
plot(compute_loss)
plot(compute_loss, initial_points)

num_steps = 500
regret = 1.4
social = 1.4
optimize = SwarmOptimizer(euclidean, compute_loss, num_steps, regret, social)
best_points, elite = optimize(key, initial_points)

print(f'elite point {elite}')
plot(compute_loss, best_points)

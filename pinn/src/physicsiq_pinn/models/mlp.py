"""mlp.py — the neural net. Deliberately the most boring net possible.

A PINN's net maps coordinates -> field values, e.g. (x, y, z) -> (ux, uy, uz).
For smooth solid-mechanics fields a small tanh MLP is the standard choice
(tanh = infinitely differentiable, which matters because the PDE loss
differentiates the net TWICE).

Everything is pure functions on pytrees — the JAX way. `params` is just a
list of (W, b) tuples you can pickle, inspect, or swap.
"""

import jax
import jax.numpy as jnp


def init_params(key, layer_sizes):
    """layer_sizes like [2, 64, 64, 64, 2]  (input dim ... output dim).

    Xavier/Glorot init: keeps signal variance stable through tanh layers.
    """
    params = []
    for n_in, n_out in zip(layer_sizes[:-1], layer_sizes[1:]):
        key, sub = jax.random.split(key)
        scale = jnp.sqrt(2.0 / (n_in + n_out))
        w = scale * jax.random.normal(sub, (n_in, n_out))
        b = jnp.zeros(n_out)
        params.append((w, b))
    return params


def apply(params, x):
    """Forward pass for ONE point x of shape (dim,). Vectorize over many
    points with jax.vmap(lambda p: apply(params, p)) — done in problems/."""
    h = x
    for w, b in params[:-1]:
        h = jnp.tanh(h @ w + b)
    w, b = params[-1]
    return h @ w + b            # last layer linear: fields aren't bounded

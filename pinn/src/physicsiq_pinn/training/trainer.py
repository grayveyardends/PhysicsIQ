"""trainer.py — the optimization loop. Problem-agnostic on purpose:
it only ever calls problem.loss_terms(params) and sums the terms.

The whole step is wrapped in jax.jit, so after the first (slow, compiling)
iteration the entire physics — residuals, boundary conditions, gradients —
runs as one fused GPU kernel. That is why this trains in seconds on the
4060 while classical FEM takes minutes.
"""

import pickle
import time
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import optax


@dataclass
class TrainConfig:
    epochs: int = 4000
    lr: float = 2e-3
    log_every: int = 200
    # Per-term loss weights. BCs get a head start over the PDE because a
    # net that ignores its boundary conditions converges to u=0 (which
    # solves the PDE perfectly and is perfectly useless).
    weights: dict = field(default_factory=lambda: {"pde": 1.0})

    def weight(self, name):
        if name in self.weights:
            return self.weights[name]
        return 10.0 if name.startswith("bc_") else 1.0


def train(problem, config: TrainConfig = None, seed=0, print_fn=print):
    """Returns (trained_params, history list of dicts)."""
    config = config or TrainConfig()
    params = problem.params_init(jax.random.PRNGKey(seed))

    # Cosine decay: start aggressive, land gentle. Standard PINN recipe.
    schedule = optax.cosine_decay_schedule(config.lr, config.epochs)
    opt = optax.adam(schedule)
    opt_state = opt.init(params)

    def total_loss(p):
        terms = problem.loss_terms(p)
        total = sum(config.weight(k) * v for k, v in terms.items())
        return total, terms

    @jax.jit
    def step(p, s):
        (loss, terms), grads = jax.value_and_grad(total_loss, has_aux=True)(p)
        updates, s = opt.update(grads, s)
        return optax.apply_updates(p, updates), s, loss, terms

    history = []
    t0 = time.time()
    for epoch in range(config.epochs):
        params, opt_state, loss, terms = step(params, opt_state)
        if epoch % config.log_every == 0 or epoch == config.epochs - 1:
            # .item() forces the async GPU value to materialize — fine at
            # log time, disastrous every iteration.
            entry = {"epoch": epoch, "total": float(loss),
                     **{k: float(v) for k, v in terms.items()}}
            history.append(entry)
            detail = " ".join(f"{k}={v:.2e}" for k, v in entry.items()
                              if k not in ("epoch", "total"))
            # flush=True: FreeCAD reads these lines live through a pipe.
            print_fn(f"epoch {epoch}/{config.epochs} "
                     f"total={entry['total']:.3e} {detail}", flush=True)
    print_fn(f"training done in {time.time() - t0:.1f}s", flush=True)
    return params, history


def save_params(params, path):
    """Checkpoint = plain pickle of the (W, b) list. Boring and portable."""
    with open(path, "wb") as fh:
        pickle.dump(jax.device_get(params), fh)


def load_params(path):
    with open(path, "rb") as fh:
        return [(jnp.asarray(w), jnp.asarray(b)) for w, b in pickle.load(fh)]

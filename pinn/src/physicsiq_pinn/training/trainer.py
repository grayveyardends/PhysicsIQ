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
    # Per-term STARTING weights. BCs get a head start over the PDE because
    # a net that ignores its boundary conditions converges to u=0 (which
    # solves the PDE perfectly and is perfectly useless).
    weights: dict = field(default_factory=lambda: {"pde": 1.0})
    # Self-adaptive balancing (SA-PINN style): each loss term gets a
    # trainable multiplier λ_k updated by gradient ASCENT — a term the
    # optimizer tries to abandon (classic failure: ignoring the traction
    # BC and outputting a zero field) sees its λ grow exponentially until
    # satisfying it becomes the cheapest option. This replaced a long
    # losing game of hand-tuned weights.
    adaptive: bool = True
    lam_lr: float = 3e-3      # ascent rate for log λ
    lam_max_boost: float = 12.0  # cap: log λ can rise this far above start
    # Stop when the total loss hasn't improved by rel_tol for `patience`
    # epochs. Real parts often converge well before the epoch budget;
    # burning the rest of it just delays the heatmap. 0 disables.
    patience: int = 0
    rel_tol: float = 0.01
    # Called as on_snapshot(epoch, params) right after the first step
    # (so the jit compile is already paid) and then every snapshot_every
    # epochs. The CLI uses it to stream a live stress field to FreeCAD.
    snapshot_every: int = 0
    on_snapshot: object = None

    def weight(self, name):
        if name in self.weights:
            return self.weights[name]
        return 10.0 if name.startswith("bc_") else 1.0


def train(problem, config: TrainConfig = None, seed=0, print_fn=print):
    """Returns (trained_params, history list of dicts)."""
    config = config or TrainConfig()
    params = problem.params_init(jax.random.PRNGKey(seed))

    # λ lives in log space so ascent multiplies instead of adds and can
    # never go negative. Keyed by term name, initialized at the config
    # weights (which encode the force-consistency priors).
    term_names = sorted(problem.loss_terms(params).keys())
    log_lam0 = {k: jnp.log(config.weight(k)) for k in term_names}
    log_lam = dict(log_lam0)

    # Cosine decay: start aggressive, land gentle. Standard PINN recipe.
    schedule = optax.cosine_decay_schedule(config.lr, config.epochs)
    opt = optax.adam(schedule)
    opt_state = opt.init(params)

    def total_loss(p, lam):
        terms = problem.loss_terms(p)
        total = sum(jnp.exp(lam[k]) * terms[k] for k in term_names)
        return total, terms

    @jax.jit
    def step(p, s, lam):
        (loss, terms), grads = jax.value_and_grad(
            total_loss, has_aux=True)(p, lam)
        updates, s = opt.update(grads, s)
        p = optax.apply_updates(p, updates)
        if config.adaptive:
            # Ascent on log λ: d(total)/d(log λ_k) = exp(log λ_k)·term_k,
            # normalized by the current total so the update is a RATE and
            # big absolute losses don't explode the multipliers.
            new_lam = {}
            for k in term_names:
                boost = config.lam_lr * jnp.exp(lam[k]) * terms[k] / (loss + 1e-12)
                new_lam[k] = jnp.clip(lam[k] + boost,
                                      log_lam0[k],
                                      log_lam0[k] + config.lam_max_boost)
            lam = new_lam
        return p, s, lam, loss, terms

    history = []
    t0 = time.time()
    best_loss, best_epoch = float("inf"), 0
    for epoch in range(config.epochs):
        params, opt_state, log_lam, loss, terms = step(
            params, opt_state, log_lam)
        if config.on_snapshot is not None and config.snapshot_every > 0 \
                and epoch % config.snapshot_every == 0:
            config.on_snapshot(epoch, params)
        if epoch % config.log_every == 0 or epoch == config.epochs - 1:
            # float() forces the async GPU value to materialize — fine at
            # log time, disastrous every iteration.
            entry = {"epoch": epoch, "total": float(loss),
                     **{k: float(v) for k, v in terms.items()}}
            history.append(entry)
            detail = " ".join(f"{k}={v:.2e}" for k, v in entry.items()
                              if k not in ("epoch", "total"))
            # flush=True: FreeCAD reads these lines live through a pipe.
            print_fn(f"epoch {epoch}/{config.epochs} "
                     f"total={entry['total']:.3e} {detail}", flush=True)
            # patience watches the RAW residual sum, not the weighted
            # total: the adaptive λ ascent inflates the total even while
            # every physical term is still improving.
            raw = sum(v for k, v in entry.items()
                      if k not in ("epoch", "total"))
            if raw < best_loss * (1.0 - config.rel_tol):
                best_loss, best_epoch = raw, epoch
            elif config.patience and epoch - best_epoch >= config.patience:
                print_fn(f"early stop at epoch {epoch}: loss flat since "
                         f"{best_epoch}", flush=True)
                break
    print_fn(f"training done in {time.time() - t0:.1f}s", flush=True)
    return params, history


def save_params(params, path):
    """Checkpoint = plain pickle of the (W, b) list. Boring and portable."""
    with open(path, "wb") as fh:
        pickle.dump(jax.device_get(params), fh)


def load_params(path):
    with open(path, "rb") as fh:
        loaded = pickle.load(fh)
    # Params are an arbitrary pytree (list of layers, dict with extra
    # scalars, ...) — rehydrate every leaf back onto the device.
    return jax.tree_util.tree_map(jnp.asarray, loaded)

"""Student ensemble, disagreement and calibration (SPEC 7.7).

* K students each estimate ẑ_k from the motion history; the car drives on the mean ẑ.
* Uncertainty u = mean over action dims of std_k(actor(obs, ẑ_k)) + 0.5 · mean over latent
  dims of std_k(ẑ_k).
* Calibration: isotonic regression from u to P(crash within the next 2 s), fit on validation
  rollouts and exported as a piecewise-linear table.
* Safety governor: when p_crash > τ, cap the throttle at max(0, 1 − (p − τ)/0.2); above τ + 0.2,
  brake at −0.4.

Honest limitation: all students share one teacher actor, so disagreement measures uncertainty
about the road conditions, not about the driving policy itself.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from trustlap.config import CONFIG
from trustlap.env import Observation
from trustlap.networks import ActorCritic
from trustlap.ppo import normalize
from trustlap.student import Adaptation, Teacher

GOVERNOR_RAMP = 0.2
GOVERNOR_BRAKE = -0.4


class EnsembleOutput(NamedTuple):
    action: jax.Array  # [2] executed (mean-ẑ) action, clipped
    uncertainty: jax.Array  # [] u
    z_mean: jax.Array  # [8]
    grip_mean: jax.Array  # [2] μ̂_front, μ̂_rear (ensemble mean)


def ensemble_act(teacher: Teacher, student_params: Any, obs: Observation) -> EnsembleOutput:
    """One car's student-ensemble decision. `student_params` has a leading K axis."""
    actor = ActorCritic(privileged_actor=True, cfg=CONFIG.network)
    adapt = Adaptation(CONFIG.network)
    z_k, grip_k = jax.vmap(lambda p: adapt.apply(p, obs.history))(student_params)  # [K, 8], [K, 2]
    obs_n = normalize(teacher.obs_stats, obs.policy)
    a_k = jax.vmap(lambda z: actor.apply(teacher.params, obs_n, z, method=ActorCritic.act))(z_k)
    a_k = jnp.clip(a_k, -1.0, 1.0)
    z_mean = jnp.mean(z_k, axis=0)
    action = jnp.clip(actor.apply(teacher.params, obs_n, z_mean, method=ActorCritic.act), -1.0, 1.0)
    u = jnp.mean(jnp.std(a_k, axis=0)) + 0.5 * jnp.mean(jnp.std(z_k, axis=0))
    return EnsembleOutput(action, u, z_mean, jnp.mean(grip_k, axis=0))


def governed_action(action: jax.Array, p_crash: jax.Array, tau: float) -> jax.Array:
    """Safety governor: cap the throttle as crash risk rises past τ; brake gently above τ + 0.2."""
    cap = jnp.clip(1.0 - (p_crash - tau) / GOVERNOR_RAMP, 0.0, 1.0)
    long = jnp.where(p_crash > tau, jnp.minimum(action[1], cap), action[1])
    long = jnp.where(p_crash > tau + GOVERNOR_RAMP, jnp.minimum(long, GOVERNOR_BRAKE), long)
    return jnp.stack([action[0], long])


# ---------------------------------------------------------------------------------------------
# Isotonic calibration (pool adjacent violators), numpy only
# ---------------------------------------------------------------------------------------------


def isotonic_fit(
    u: np.ndarray, y: np.ndarray, max_points: int = 64
) -> tuple[np.ndarray, np.ndarray]:
    """Monotone non-decreasing fit of y on u. Returns a piecewise-linear table (u_knots, p_knots)
    with at most `max_points` knots (block centers of the PAV solution)."""
    order = np.argsort(u, kind="stable")
    u_sorted = u[order].astype(np.float64)
    y_sorted = y[order].astype(np.float64)
    # Pool adjacent violators over blocks [value, weight, u_sum].
    values: list[float] = []
    weights: list[float] = []
    u_sums: list[float] = []
    for ui, yi in zip(u_sorted, y_sorted, strict=True):
        values.append(yi)
        weights.append(1.0)
        u_sums.append(ui)
        while len(values) > 1 and values[-2] > values[-1]:
            w = weights[-2] + weights[-1]
            v = (values[-2] * weights[-2] + values[-1] * weights[-1]) / w
            s = u_sums[-2] + u_sums[-1]
            values[-2:] = [v]
            weights[-2:] = [w]
            u_sums[-2:] = [s]
    centers = np.array(u_sums) / np.array(weights)
    vals = np.array(values)
    if len(centers) > max_points:  # thin while keeping the ends
        keep = np.unique(np.round(np.linspace(0, len(centers) - 1, max_points)).astype(int))
        centers, vals = centers[keep], vals[keep]
    return centers, vals


def isotonic_predict(table_u: np.ndarray, table_p: np.ndarray, u: Any) -> Any:
    """Piecewise-linear interpolation of the calibration table, clamped at the ends."""
    return np.interp(u, table_u, table_p)


def reliability(p: np.ndarray, y: np.ndarray, bins: int = 10) -> dict[str, Any]:
    """Equal-count reliability bins, ECE, Brier score and AUROC (SPEC 7.7)."""
    order = np.argsort(p, kind="stable")
    chunks = np.array_split(order, bins)
    rows = []
    ece = 0.0
    for c in chunks:
        if len(c) == 0:
            continue
        pp, yy = float(np.mean(p[c])), float(np.mean(y[c]))
        rows.append({"p_pred": pp, "p_obs": yy, "n": int(len(c))})
        ece += len(c) / len(p) * abs(pp - yy)
    brier = float(np.mean((p - y) ** 2))
    return {"bins": rows, "ece": float(ece), "brier": brier, "auroc": auroc(p, y)}


def auroc(score: np.ndarray, y: np.ndarray) -> float | None:
    """Area under the ROC curve via the rank statistic (ties averaged)."""
    pos = y > 0.5
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    order = np.argsort(score, kind="stable")
    ranks = np.empty(len(score))
    ranks[order] = np.arange(1, len(score) + 1)
    # average ranks over ties
    s_sorted = score[order]
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = (i + j + 2) / 2
        i = j + 1
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))

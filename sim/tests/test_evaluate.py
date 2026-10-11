"""Evaluation rollouts (regression tests for bugs found while evaluating)."""

import jax
import jax.numpy as jnp
import numpy as np

from trustlap.conditions import suite_conditions
from trustlap.env import Env
from trustlap.evaluate import _scripted, cluster_bootstrap, run
from trustlap.track import generate_track, make_bank


def _dry(bank, n):  # noqa: ANN001, ANN202
    tids = np.arange(n)
    keys = jax.random.split(jax.random.key(0), n)
    conds = jax.vmap(lambda k, length, width: suite_conditions(k, "dry", length, width))(
        keys, jnp.asarray(bank.length)[tids], jnp.asarray(bank.width)[tids]
    )
    return tids, conds


def test_one_driver_on_two_banks_uses_each_banks_geometry() -> None:
    """The compiled rollout once closed over the first bank it saw, so a second bank with the
    same lap count was simulated on the wrong tracks (every car crashed at once)."""
    act = _scripted()
    for seeds in ((400001, 400002), (400003, 400004)):
        bank = make_bank([generate_track(s) for s in seeds])
        tids, conds = _dry(bank, 2)
        laps = run(Env(bank), tids, conds, act)
        assert laps.completed.all(), seeds


def test_cluster_bootstrap_brackets_the_mean() -> None:
    rng = np.random.default_rng(0)
    grid = rng.uniform(size=(50, 3))
    lo, hi = cluster_bootstrap(grid, rng)
    assert lo < grid.mean() < hi

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


def test_notice_latencies_window_and_control() -> None:
    """A prompt response counts; a dip seconds later doesn't; control dips are counted."""
    from trustlap.evaluate import notice_latencies

    t = 100
    mu = np.full((t, 3), 1.0)
    mu[10:16] = 0.6  # true grip low for 6 steps (300 ms at 50 ms)
    mu[:, 2] = 1.0  # trial 2 never drops: skipped
    hat = np.full((t, 3), 1.0)
    hat[12:, 0] = 0.75  # trial 0: falls by ≥ half the 0.4 drop 2 steps in → 100 ms
    hat[60:, 1] = 0.7  # trial 1: dips 2.5 s later, outside the window → not noticed
    ctl = np.full((t, 3), 1.0)
    ctl[14:, 1] = 0.7  # control dips inside trial 1's window → one false alarm
    lat, dropped, false_alarms = notice_latencies(mu, hat, ctl, dt_ms=50.0)
    assert dropped.tolist() == [True, True, False]
    assert lat[0] == 100.0
    assert np.isnan(lat[1]) and np.isnan(lat[2])
    assert false_alarms == 1

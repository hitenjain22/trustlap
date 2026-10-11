"""PBT (SPEC 12.1): exploit copies params + optimizer + obs stats; lineage ids are consistent;
hyperparameters stay in bounds."""

import jax
import jax.numpy as jnp
import numpy as np

from trustlap.config import CONFIG
from trustlap.env import Env
from trustlap.pbt import Lineage, exploit_explore, initial_hyperparams
from trustlap.ppo import PPO
from trustlap.track import generate_track, make_bank


def test_exploit_copies_and_lineage_is_consistent() -> None:
    env = Env(make_bank([generate_track(5), generate_track(6)]))
    ppo = PPO(env, n_envs=8, rollout_len=8)
    p = 4
    ts = jax.vmap(ppo.init, in_axes=(0, None, None))(
        jax.random.split(jax.random.PRNGKey(0), p), jnp.zeros(8), jnp.zeros(8, bool)
    )
    # Make member stats distinguishable.
    ts = ts._replace(
        obs_stats=ts.obs_stats._replace(mean=ts.obs_stats.mean + jnp.arange(p)[:, None])
    )
    rng = np.random.default_rng(0)
    hp = initial_hyperparams(p, rng)
    fitness = np.array([1.0, 4.0, 2.0, 3.0])  # member 0 worst, member 1 best
    lineage = Lineage(p)
    new, hp2, events = exploit_explore(ts, hp, fitness, lineage, 25, 1000, rng)
    assert len(events) == 1 and events[0]["slot"] == 0 and events[0]["parent_id"] == 1
    leaves_new = jax.tree.leaves(new.params)
    leaves_old = jax.tree.leaves(ts.params)
    for a, b in zip(leaves_new, leaves_old, strict=True):
        np.testing.assert_array_equal(a[0], b[1])  # copied
        np.testing.assert_array_equal(a[2], b[2])  # untouched
    np.testing.assert_array_equal(new.obs_stats.mean[0], ts.obs_stats.mean[1])
    for a, b in zip(jax.tree.leaves(new.opt_state), jax.tree.leaves(ts.opt_state), strict=True):
        np.testing.assert_array_equal(np.asarray(a)[0], np.asarray(b)[1])
    assert lineage.ids.tolist() == [4, 1, 2, 3] and lineage.next_id == 5
    lo, hi = CONFIG.pbt.lr_range
    assert lo <= hp2["lr"][0] <= hi
    lo, hi = CONFIG.pbt.ent_range
    assert lo <= hp2["ent"][0] <= hi
    assert len(lineage.snapshots) == 1 and len(lineage.snapshots[0]["members"]) == p

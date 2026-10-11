"""The scripted baseline completes laps (SPEC M1 acceptance, small CI-sized version).

The full check runs `scripts/eval_scripted.py` on the 2,000-track training bank (see PLAN.md).
"""

import jax
import jax.numpy as jnp
import numpy as np

from trustlap.conditions import suite_conditions
from trustlap.env import Env
from trustlap.rollout import run_laps
from trustlap.scripted import scripted_action
from trustlap.track import generate_track, make_bank


def test_scripted_driver_completes_dry_and_wet_laps() -> None:
    bank = make_bank([generate_track(s) for s in range(300000, 300008)])
    env = Env(bank)
    track_ids = np.arange(8)
    policy = lambda state, obs, bank: scripted_action(state, bank, xp=jnp)  # noqa: E731
    for suite, minimum in (("dry", 1.0), ("wet", 0.75)):
        keys = jax.random.split(jax.random.key(0), len(track_ids))
        conds = jax.vmap(
            lambda k, length, width, suite=suite: suite_conditions(k, suite, length, width)
        )(keys, jnp.asarray(bank.length), jnp.asarray(bank.width))
        result, _ = run_laps(env, track_ids, conds, policy)
        assert result.completed.mean() >= minimum, (suite, result)
        assert np.all(result.lap_time_s[result.completed] < 120.0)

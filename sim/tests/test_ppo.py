"""PPO checks (SPEC 12.1): GAE by hand (terminated vs truncated) and a learning smoke test."""

import jax
import jax.numpy as jnp
import numpy as np

from trustlap.ppo import Transition, gae, stats_init, stats_update


def _traj(rewards, values, v_next, terminated, done):  # noqa: ANN001, ANN202
    t = len(rewards)
    col = lambda x: jnp.asarray(x, jnp.float32)[:, None]  # noqa: E731
    z = jnp.zeros((t, 1))
    return Transition(
        z, z, z, z, z, z, col(values), col(v_next), col(rewards), col(terminated), col(done)
    )


def test_gae_terminated_does_not_bootstrap() -> None:
    g, lam = 0.9, 0.5
    traj = _traj([1, 1, 1], [0.5, 0.5, 0.5], [0.5, 0.5, 2.0], [0, 0, 1], [0, 0, 1])
    adv, _ = gae(traj, g, lam)
    d2 = 1 - 0.5  # terminated: no bootstrap from V_next = 2.0
    d1 = 1 + g * 0.5 - 0.5
    d0 = 1 + g * 0.5 - 0.5
    a2 = d2
    a1 = d1 + g * lam * a2
    a0 = d0 + g * lam * a1
    np.testing.assert_allclose(np.asarray(adv[:, 0]), [a0, a1, a2], rtol=1e-6)


def test_gae_truncated_bootstraps_from_final_value() -> None:
    g, lam = 0.9, 0.5
    traj = _traj([1, 1, 1], [0.5, 0.5, 0.5], [0.5, 0.5, 2.0], [0, 0, 0], [0, 0, 1])
    adv, targets = gae(traj, g, lam)
    a2 = 1 + g * 2.0 - 0.5  # truncated: bootstrap from V(s_final) = 2.0
    a1 = (1 + g * 0.5 - 0.5) + g * lam * a2
    a0 = (1 + g * 0.5 - 0.5) + g * lam * a1
    np.testing.assert_allclose(np.asarray(adv[:, 0]), [a0, a1, a2], rtol=1e-6)
    np.testing.assert_allclose(np.asarray(targets[:, 0]), np.array([a0, a1, a2]) + 0.5, rtol=1e-6)


def test_gae_episode_boundary_cuts_the_trace() -> None:
    traj = _traj([0, 0, 5], [0, 0, 0], [0, 0, 0], [0, 1, 0], [0, 1, 0])
    adv, _ = gae(traj, 0.99, 0.95)
    assert float(adv[1, 0]) == 0.0  # reward after the episode ended doesn't leak back


def test_running_stats_match_numpy() -> None:
    rng = np.random.default_rng(0)
    data = rng.normal(3.0, 2.0, (1000, 4)).astype(np.float32)
    s = stats_init(4)
    for chunk in np.split(data, 10):
        s = stats_update(s, jnp.asarray(chunk))
    np.testing.assert_allclose(np.asarray(s.mean), data.mean(0), rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(np.asarray(s.var), data.var(0), rtol=1e-3)


def test_ppo_learns_on_the_real_env_quickly() -> None:
    """Smoke test: a few updates on a tiny batch raise the mean speed/return (learning works)."""
    from trustlap.env import Env
    from trustlap.ppo import PPO
    from trustlap.track import generate_track, make_bank

    env = Env(make_bank([generate_track(s) for s in (21, 22)]))
    ppo = PPO(env, n_envs=64, rollout_len=32)
    d, st = jnp.zeros(64), jnp.zeros(64, bool)
    ts = ppo.init(jax.random.PRNGKey(0), d, st)
    upd = jax.jit(ppo.update)
    speeds = []
    for _ in range(25):
        ts, m = upd(ts, d, st, jnp.float32(1e-3), jnp.float32(0.003), env.bank)
        speeds.append(float(m["speed_sum"]) / (64 * 32))
    assert all(np.isfinite(speeds))
    assert np.mean(speeds[-5:]) > np.mean(speeds[:5])

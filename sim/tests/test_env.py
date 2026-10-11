"""Environment tests (SPEC 12.1): observation layout, reward terms, termination rules,
auto-reset, history buffer, progress tracking."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from trustlap.conditions import nominal_conditions
from trustlap.config import CONFIG, OBS_LAYOUT
from trustlap.env import Env, core_step, lateral_and_heading_error, observe, point_at, start_state
from trustlap.scripted import scripted_action
from trustlap.track import TrackBank, generate_track, make_bank

NO_GUST = np.zeros(2, np.float32)
NOISE_FREE = np.zeros(CONFIG.obs.policy_size - 1, np.float32)


@pytest.fixture(scope="module")
def bank() -> TrackBank:
    return make_bank([generate_track(s) for s in (11, 12, 13, 14)])


def on_track(bank: TrackBank, tid: int = 0, s: float = 50.0, speed: float = 10.0):  # noqa: ANN201
    return start_state(
        bank, np.int32(tid), np.float32(s), np.float32(speed), nominal_conditions(), 0.0
    )


def step(state, bank, action=(0.0, 0.0)):  # noqa: ANN001, ANN201
    return core_step(state, np.asarray(action, np.float32), NO_GUST, bank)


# Observations ----------------------------------------------------------------------------------
def test_observation_sizes_match_layout(bank: TrackBank) -> None:
    obs, sensed = observe(on_track(bank), bank, NOISE_FREE)
    assert obs.policy.shape == (CONFIG.obs.policy_size,) == (len(OBS_LAYOUT["policy"]),) == (33,)
    assert (
        obs.privileged.shape
        == (CONFIG.obs.privileged_size,)
        == (len(OBS_LAYOUT["privileged"]),)
        == (14,)
    )
    assert obs.history.shape == (CONFIG.obs.history_size,) == (200,)
    assert len(OBS_LAYOUT["history_row"]) == CONFIG.obs.history_row_size == 10
    assert sensed.shape == (8,)


def test_observation_values_for_a_car_on_the_centerline(bank: TrackBank) -> None:
    state = on_track(bank, speed=15.0)
    obs, _ = observe(state, bank, NOISE_FREE)
    p = dict(zip(OBS_LAYOUT["policy"], obs.policy.tolist(), strict=True))
    assert p["vx"] == pytest.approx(15.0 / 30.0)
    assert p["vy"] == p["yaw_rate"] == p["steer"] == 0.0
    assert abs(p["e_y"]) < 1e-3 and abs(p["e_psi"]) < 1e-3
    assert p["width"] == pytest.approx(float(bank.width[0]) / 14.0)
    # The 5 m lookahead point is ~5 m straight ahead (scaled by 1/100).
    assert p["look_5m_x"] == pytest.approx(0.05, abs=2e-3)
    assert abs(p["look_5m_y"]) < 2e-3
    # Lookahead x grows monotonically along a track with no hairpin in the first 160 m... at least
    # the first few points do.
    xs = [p[f"look_{int(d)}m_x"] for d in CONFIG.obs.lookahead_m[:4]]
    assert xs == sorted(xs)


def test_noise_only_touches_proprioception_and_lookahead(bank: TrackBank) -> None:
    state = on_track(bank)._replace(cond=nominal_conditions()._replace(obs_noise=np.float32(0.1)))
    clean, _ = observe(state, bank, NOISE_FREE)
    noisy, sensed = observe(state, bank, np.ones_like(NOISE_FREE))
    np.testing.assert_allclose(noisy.policy[:-1] - clean.policy[:-1], 0.1, atol=1e-6)
    assert noisy.policy[-1] == clean.policy[-1]
    np.testing.assert_allclose(sensed, noisy.policy[:8])


def test_privileged_reflects_conditions(bank: TrackBank) -> None:
    dry, _ = observe(on_track(bank), bank, NOISE_FREE)
    wet_cond = nominal_conditions()._replace(mu_base=np.float32(0.6))
    wet, _ = observe(on_track(bank)._replace(cond=wet_cond), bank, NOISE_FREE)
    idx = OBS_LAYOUT["privileged"].index("mu_front")
    assert wet.privileged[idx] < dry.privileged[idx]
    assert np.all(np.abs(dry.privileged) <= 1.0 + 1e-6)


# Reward ----------------------------------------------------------------------------------------
def test_reward_is_progress_on_a_clean_straight_step(bank: TrackBank) -> None:
    state = on_track(bank, speed=20.0)
    new, reward, info = step(state, bank)
    width = float(bank.width[0])
    e_y, _ = lateral_and_heading_error(
        bank, new.track_id, new.s_m, new.car.x_m, new.car.y_m, new.car.yaw_rad
    )
    expected = 0.1 * info.delta_s_m - 0.02 * (e_y / (width / 2)) ** 2
    assert reward == pytest.approx(expected, abs=1e-6)
    assert 0.9 < info.delta_s_m < 1.1  # ~20 m/s × 0.05 s


def test_reward_smoothness_penalty(bank: TrackBank) -> None:
    state = on_track(bank)
    _, r_same, _ = step(state, bank, (0.0, 0.0))
    _, r_jerk, info = step(state, bank, (0.5, -0.5))
    # Different action changes the physics slightly too, so compare with the penalty term alone.
    assert r_jerk < r_same
    assert r_same - r_jerk == pytest.approx(0.01 * 0.5, abs=0.02)


# Termination -----------------------------------------------------------------------------------
def _offset(state, bank, lateral_m: float, yaw_offset: float = 0.0):  # noqa: ANN001, ANN201
    point, tangent = point_at(bank, state.track_id, state.s_m)
    normal = np.array([-tangent[1], tangent[0]])
    xy = point + lateral_m * normal
    car = state.car._replace(
        x_m=np.float32(xy[0]),
        y_m=np.float32(xy[1]),
        yaw_rad=np.float32(math.atan2(tangent[1], tangent[0]) + yaw_offset),
    )
    return state._replace(car=car)


def test_off_track_is_a_crash(bank: TrackBank) -> None:
    state = _offset(on_track(bank), bank, float(bank.width[0]) / 2 + 1.5)
    _, reward, info = step(state, bank)
    assert info.crashed and info.terminated and not info.truncated
    assert reward < -9.0


def test_spinning_for_over_one_second_is_a_crash(bank: TrackBank) -> None:
    state = _offset(on_track(bank, speed=0.0), bank, 0.0, yaw_offset=math.radians(120))
    steps = 0
    while steps < 25:
        state, _, info = step(state, bank)
        steps += 1
        if info.terminated:
            break
    assert info.crashed and not info.stalled
    assert steps == 21  # > 1 s = the 21st consecutive step


def test_stall_terminates_after_three_seconds(bank: TrackBank) -> None:
    state = on_track(bank, speed=0.0)
    steps = 0
    while steps < 70:
        state, reward, info = step(state, bank)
        steps += 1
        if info.terminated:
            break
    assert info.stalled and not info.crashed
    assert steps == 61  # > 3 s = the 61st consecutive step
    assert reward == pytest.approx(-2.0, abs=0.05)


def test_wrong_way_terminates(bank: TrackBank) -> None:
    state = _offset(on_track(bank, s=200.0, speed=20.0), bank, 0.0, yaw_offset=math.pi)
    for _ in range(25):
        state, _, info = step(state, bank)
        if info.terminated:
            break
    assert info.wrong_way and not info.crashed


def test_lap_completion(bank: TrackBank) -> None:
    state = on_track(bank, speed=20.0)._replace(progress_m=np.float32(float(bank.length[0]) - 0.5))
    _, reward, info = step(state, bank)
    assert info.lap_completed and info.terminated
    assert reward > 9.0


def test_time_limit_truncates(bank: TrackBank) -> None:
    state = on_track(bank, speed=20.0)._replace(steps=np.int32(CONFIG.episode.max_steps - 1))
    _, _, info = step(state, bank)
    assert info.truncated and not info.terminated


# History ---------------------------------------------------------------------------------------
def test_history_buffer_shifts_oldest_first(bank: TrackBank) -> None:
    state = on_track(bank)
    rows = []
    for k in range(3):
        proprio = np.full(8, k + 1, np.float32)
        action = np.array([0.1 * (k + 1), -0.1 * (k + 1)], np.float32)
        state = state._replace(last_proprio=proprio)
        state, _, _ = core_step(state, action, NO_GUST, bank)
        rows.append(np.concatenate([proprio, action]))
    np.testing.assert_allclose(state.history[-3:], np.stack(rows), atol=1e-6)
    assert np.all(state.history[:-3] == 0.0)
    assert state.history_len == 3


# Progress tracking -----------------------------------------------------------------------------
def test_progress_never_jumps_over_a_full_scripted_lap(bank: TrackBank) -> None:
    for tid in range(len(bank.width)):
        state = start_state(
            bank, np.int32(tid), np.float32(0.0), np.float32(5.0), nominal_conditions(), 0.0
        )
        for _ in range(CONFIG.episode.max_steps):
            prev_odo = state.car.odometer_m
            state, _, info = core_step(state, scripted_action(state, bank), NO_GUST, bank)
            moved = state.car.odometer_m - prev_odo
            assert (
                abs(info.delta_s_m) <= moved + 0.5
            )  # along-track progress never exceeds distance driven
            if info.terminated:
                break
        assert info.lap_completed, f"track {tid}"
        # Distance driven ≈ track length (cars cut corners or swing wide a little).
        assert abs(state.car.odometer_m - bank.length[tid]) / bank.length[tid] < 0.05


# Batched env -----------------------------------------------------------------------------------
def test_different_keys_give_different_conditions(bank: TrackBank) -> None:
    env = Env(bank)
    state, _ = env.reset(jax.random.split(jax.random.key(0), 64), jnp.ones(64))
    assert len(np.unique(np.asarray(state.cond.mu_dry))) == 64
    assert len(np.unique(np.asarray(state.track_id))) > 1


def test_auto_reset_produces_a_fresh_episode(bank: TrackBank) -> None:
    env = Env(bank)
    n = 32
    state, _ = env.reset(jax.random.split(jax.random.key(1), n), jnp.ones(n))
    state = state._replace(steps=jnp.full(n, CONFIG.episode.max_steps - 1, jnp.int32))
    new, obs, _, info, final_obs = env.step(
        state, jnp.zeros((n, 2)), jax.random.split(jax.random.key(2), n), jnp.ones(n)
    )
    assert bool(jnp.all(info.truncated))
    assert bool(jnp.all(new.steps == 0))
    assert bool(jnp.any(new.cond.mu_dry != state.cond.mu_dry))
    assert not bool(jnp.allclose(obs.policy, final_obs.policy))  # final obs is pre-reset
    assert bool(jnp.all(jnp.isfinite(obs.policy)))

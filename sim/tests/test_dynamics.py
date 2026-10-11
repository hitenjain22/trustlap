"""Dynamics sanity tests: all ten from SPEC 6.8, plus NumPy/JAX agreement and a jitter check."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from trustlap.conditions import Conditions, nominal_conditions, sample_conditions, with_rain
from trustlap.config import CONFIG
from trustlap.dynamics import CarState, axle_positions, control_step, initial_car_state, substep

CAR = CONFIG.car
G = CAR.gravity_mps2
NO_GUST = np.zeros(2, np.float32)


def act(steer: float, long: float) -> np.ndarray:
    return np.array([steer, long], np.float32)


def drive(state: CarState, cond: Conditions, action: np.ndarray, steps: int) -> list[CarState]:
    """Run `steps` control steps with a constant action; returns every state after the start."""
    out = []
    for _ in range(steps):
        state = control_step(state, cond, action, NO_GUST)
        out.append(state)
    return out


def hold_speed(
    state: CarState, cond: Conditions, steer: float, speed: float, steps: int
) -> list[CarState]:
    """Constant steering with a P-controller on throttle to hold `speed`."""
    out = []
    for _ in range(steps):
        a_long = float(np.clip(0.5 * (speed - state.vx_mps), -1, 1))
        state = control_step(state, cond, act(steer, a_long), NO_GUST)
        out.append(state)
    return out


def ice() -> Conditions:
    c = nominal_conditions()
    return c._replace(mu_dry=np.float32(0.25), mu_base=np.float32(0.25))


# 1 ---------------------------------------------------------------------------------------------
def test_coasting_decelerates_and_never_reverses() -> None:
    states = drive(initial_car_state(0, 0, 0, 20.0), nominal_conditions(), act(0, 0), 1200)
    vx = np.array([s.vx_mps for s in states])
    assert np.all(np.diff(vx) <= 1e-6)
    assert np.all(vx >= 0.0)
    assert vx[-1] < 0.5 * 20.0  # drag + rolling resistance only: ~5 m/s after 60 s


# 2 ---------------------------------------------------------------------------------------------
def test_full_throttle_reaches_analytic_terminal_speed() -> None:
    cond = nominal_conditions()
    mass = CAR.mass_kg

    def net_force(v: float) -> float:
        drag = 0.5 * CAR.air_density_kgpm3 * float(cond.drag_area_m2) * v * v
        return CAR.power_w / v - drag - float(cond.rolling_coef) * mass * G

    lo, hi = 1.0, 200.0
    for _ in range(100):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if net_force(mid) > 0 else (lo, mid)
    final = drive(initial_car_state(0, 0, 0, 20.0), cond, act(0, 1), 2400)[-1]
    assert abs(final.vx_mps - lo) / lo < 0.05


# 3 ---------------------------------------------------------------------------------------------
def test_low_speed_turning_radius_is_kinematic() -> None:
    states = hold_speed(initial_car_state(0, 0, 0, 4.0), nominal_conditions(), 0.1, 4.0, 200)
    last = states[-1]
    radius = last.vx_mps / last.yaw_rate_radps
    kinematic = CAR.wheelbase_m / np.tan(last.steer_rad)
    assert abs(radius - kinematic) / kinematic < 0.10


# 4 ---------------------------------------------------------------------------------------------
def test_low_grip_high_speed_understeers() -> None:
    # Coasting (no throttle: on a rear-drive car, throttle at 4x the grip limit spins it instead).
    cond = nominal_conditions()._replace(mu_dry=np.float32(0.4), mu_base=np.float32(0.4))
    states = drive(initial_car_state(0, 0, 0, 20.0), cond, act(0.25, 0.0), 30)
    # Path radius = speed / rate of change of the direction of travel (not v / yaw rate,
    # which is the radius of the body's rotation and ignores the slide).
    course = np.array([s.yaw_rad + np.arctan2(s.vy_mps, s.vx_mps) for s in states])
    speed = np.array([np.hypot(s.vx_mps, s.vy_mps) for s in states])
    dt = CONFIG.time.control_dt_s
    path_radius = speed[10:20].mean() / ((course[20] - course[10]) / (10 * dt))
    kinematic = CAR.wheelbase_m / np.tan(states[-1].steer_rad)
    assert path_radius > 1.5 * kinematic


def _puddle_at(cond: Conditions, xy: np.ndarray, radius_m: float = 2.0) -> Conditions:
    """A single small puddle (σ = 1 m) so it sits under one axle only (they are 2.6 m apart)."""
    p = cond.puddle_xy_m.copy()
    p[0] = xy
    active = cond.puddle_active.copy()
    active[0] = 1.0
    mult = cond.puddle_mult.copy()
    mult[0] = 0.35
    radius = cond.puddle_radius_m.copy()
    radius[0] = radius_m
    return cond._replace(
        puddle_xy_m=p, puddle_active=active, puddle_mult=mult, puddle_radius_m=radius
    )


def test_puddle_under_rear_axle_oversteers_and_under_front_understeers() -> None:
    dry = nominal_conditions()
    # Settle into a steady left turn at 15 m/s, then add moderate throttle (0.3: within the rear
    # tires' dry grip, so the dry baseline doesn't spin; heavier throttle spins the car anyway).
    state = hold_speed(initial_car_state(0, 0, 0, 15.0), dry, 0.15, 15.0, 80)[-1]
    front, rear = axle_positions(state)

    def yaw_rates(cond: Conditions) -> np.ndarray:
        states = drive(state, cond, act(0.15, 0.3), 2)  # 0.1 s
        return np.array([s.yaw_rate_radps for s in states])

    baseline = yaw_rates(dry)
    rear_puddle = yaw_rates(_puddle_at(dry, rear))
    front_puddle = yaw_rates(_puddle_at(dry, front))
    assert rear_puddle[-1] > 1.1 * baseline[-1]  # rear lets go: yaw rate overshoots
    assert front_puddle[-1] < 0.9 * baseline[-1]  # front washes out: yaw rate drops


# 5 ---------------------------------------------------------------------------------------------
def _braking_distance(cond: Conditions) -> float:
    state = initial_car_state(0, 0, 0, 30.0)
    for _ in range(1000):
        state = control_step(state, cond, act(0, -1), NO_GUST)
        if state.vx_mps < 0.01:
            break
    return float(state.odometer_m)


def test_braking_distance_orders_and_matches_analytic_on_dry() -> None:
    dry = nominal_conditions()
    d_dry = _braking_distance(dry)
    d_wet = _braking_distance(with_rain(dry, 1.0))
    d_ice = _braking_distance(ice())
    assert d_dry < d_wet < d_ice
    analytic = 30.0**2 / (2 * float(dry.mu_base) * G)
    assert abs(d_dry - analytic) / analytic < 0.15


# 6 ---------------------------------------------------------------------------------------------
def test_steering_lag_reaches_63_percent_at_tau() -> None:
    tau = 0.1
    cond = nominal_conditions()._replace(steer_tau_s=np.float32(tau))
    state = initial_car_state(0, 0, 0, 10.0)
    target = 0.5 * CAR.steer_max_rad  # small enough that the rate limit never binds
    dt = CONFIG.time.physics_dt_s
    t = 0.0
    while state.steer_rad < 0.632 * target:
        state = substep(state, cond, np.float32(0.5), np.float32(0.0))
        t += dt
    assert abs(t - tau) <= dt + 1e-9


# 7 ---------------------------------------------------------------------------------------------
def test_action_delay_takes_effect_exactly_d_steps_later() -> None:
    for delay in range(CONFIG.conditions.delay_ood_steps + 1):
        cond = nominal_conditions()._replace(delay_steps=np.int32(delay))
        state = initial_car_state(0, 0, 0, 10.0)
        steer = []
        for k in range(delay + 2):
            a = act(1.0, 0.0) if k == 0 else act(0.0, 0.0)
            state = control_step(state, cond, a, NO_GUST)
            steer.append(float(state.steer_rad))
        assert all(s == 0.0 for s in steer[:delay]), (delay, steer)
        assert steer[delay] > 0.0, (delay, steer)


# 8 ---------------------------------------------------------------------------------------------
def test_headwind_lowers_top_speed_and_crosswind_drifts() -> None:
    calm = nominal_conditions()
    headwind = calm._replace(wind_mps=np.array([-10.0, 0.0], np.float32))
    v_calm = drive(initial_car_state(0, 0, 0, 60.0), calm, act(0, 1), 1200)[-1].vx_mps
    v_head = drive(initial_car_state(0, 0, 0, 60.0), headwind, act(0, 1), 1200)[-1].vx_mps
    assert v_head < v_calm - 3.0

    # Wind blowing toward +y. It first pushes the car sideways (downwind); because the center
    # of pressure is behind the center of gravity, it then turns the nose into the wind, like a
    # weathervane. Either way, a car with straight wheels no longer drives straight.
    crosswind = calm._replace(wind_mps=np.array([0.0, 10.0], np.float32))
    calm_states = hold_speed(initial_car_state(0, 0, 0, 20.0), calm, 0.0, 20.0, 60)
    cross_states = hold_speed(initial_car_state(0, 0, 0, 20.0), crosswind, 0.0, 20.0, 200)
    assert abs(calm_states[-1].y_m) < 1e-3
    assert cross_states[0].vy_mps > 0.0  # initial push is downwind
    assert cross_states[-1].yaw_rad < 0.0  # nose turned toward where the wind comes from
    assert abs(cross_states[-1].y_m) > 0.5  # 10 s later the path has drifted off the line


# 9 ---------------------------------------------------------------------------------------------
def _jax_batch_step(
    states: CarState, conds: Conditions, actions: jax.Array, noise: jax.Array
) -> CarState:
    return jax.vmap(lambda s, c, a, n: control_step(s, c, a, n, CONFIG, jnp))(
        states, conds, actions, noise
    )


def _random_world(key: jax.Array, n: int) -> tuple[CarState, Conditions]:
    k_cond, k_pos, k_speed, k_yaw, k_puddle = jax.random.split(key, 5)
    conds = jax.vmap(lambda k: sample_conditions(k, 1.0, 1200.0, 12.0))(jax.random.split(k_cond, n))
    xy = jax.random.uniform(k_pos, (n, 2), minval=-50, maxval=50)
    speed = jax.random.uniform(k_speed, (n,), minval=0.0, maxval=60.0)
    yaw = jax.random.uniform(k_yaw, (n,), minval=-np.pi, maxval=np.pi)
    puddles = jax.random.uniform(
        k_puddle, (n, CONFIG.conditions.puddle_slots, 2), minval=-60, maxval=60
    )
    conds = conds._replace(puddle_xy_m=puddles)
    states = jax.vmap(lambda x, y, p, v: initial_car_state(x, y, p, v, CONFIG, jnp))(
        xy[:, 0], xy[:, 1], yaw, speed
    )
    return states, conds


def test_random_actions_full_ranges_stay_finite() -> None:
    n_cars, n_steps = 100, 100  # 10,000 car-steps
    states, conds = _random_world(jax.random.key(0), n_cars)

    @jax.jit
    def rollout(states: CarState, key: jax.Array) -> tuple[CarState, jax.Array]:
        def body(s: CarState, k: jax.Array) -> tuple[CarState, jax.Array]:
            ka, kn = jax.random.split(k)
            actions = jax.random.uniform(ka, (n_cars, 2), minval=-1.0, maxval=1.0)
            s = _jax_batch_step(s, conds, actions, jax.random.normal(kn, (n_cars, 2)))
            finite = jnp.all(
                jnp.stack([jnp.all(jnp.isfinite(leaf)) for leaf in jax.tree.leaves(s)])
            )
            return s, finite

        return jax.lax.scan(body, states, jax.random.split(key, n_steps))

    final, finite = rollout(states, jax.random.key(1))
    assert bool(jnp.all(finite))
    assert bool(jnp.all(final.vx_mps >= 0.0))


# 10 --------------------------------------------------------------------------------------------
def test_vmap_over_4096_cars_matches_individual_cars() -> None:
    n = 4096
    states, conds = _random_world(jax.random.key(2), n)
    actions = jax.random.uniform(jax.random.key(3), (n, 2), minval=-1.0, maxval=1.0)
    noise = jax.random.normal(jax.random.key(4), (n, 2))
    batched = jax.jit(_jax_batch_step)(states, conds, actions, noise)
    for i in range(4):
        one = jax.tree.map(lambda leaf, i=i: leaf[i], (states, conds, actions, noise))
        single = control_step(*one, CONFIG, jnp)
        for a, b in zip(
            jax.tree.leaves(single),
            jax.tree.leaves(jax.tree.map(lambda leaf, i=i: leaf[i], batched)),
            strict=True,
        ):
            np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-5, atol=1e-5)


# Extras ----------------------------------------------------------------------------------------
def test_numpy_and_jax_implementations_agree() -> None:
    states, conds = _random_world(jax.random.key(5), 8)
    actions = jax.random.uniform(jax.random.key(6), (8, 2), minval=-1.0, maxval=1.0)
    noise = jax.random.normal(jax.random.key(7), (8, 2))
    for i in range(8):
        one = jax.tree.map(lambda leaf, i=i: np.asarray(leaf[i]), (states, conds, actions, noise))
        out_np = control_step(*one, CONFIG, np)
        out_jax = control_step(*jax.tree.map(jnp.asarray, one), CONFIG, jnp)
        for a, b in zip(jax.tree.leaves(out_np), jax.tree.leaves(out_jax), strict=True):
            np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-4, atol=1e-4)


def test_numpy_path_stays_float32() -> None:
    cond = nominal_conditions()._replace(gust_sigma_mps=np.float32(2.0))
    state = control_step(
        initial_car_state(0, 0, 0, 10.0), cond, act(0.3, 0.5), np.ones(2, np.float32)
    )
    for leaf in state:
        assert np.asarray(leaf).dtype == np.float32


def test_parked_car_does_not_jitter() -> None:
    states = drive(initial_car_state(0, 0, 0.3, 0.0), nominal_conditions(), act(0.5, 0), 100)
    last = states[-1]
    assert last.vx_mps == 0.0
    assert abs(last.vy_mps) < 1e-6 and abs(last.yaw_rate_radps) < 1e-6
    assert abs(last.x_m) < 1e-6 and abs(last.y_m) < 1e-6

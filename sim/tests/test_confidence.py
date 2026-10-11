"""Calibration math and the safety governor (SPEC 7.7)."""

import jax.numpy as jnp
import numpy as np
import pytest

from trustlap.confidence import auroc, governed_action, isotonic_fit, isotonic_predict, reliability


def test_isotonic_is_monotone_and_fits_a_step() -> None:
    rng = np.random.default_rng(0)
    u = rng.uniform(0, 1, 5000)
    y = (rng.uniform(0, 1, 5000) < np.where(u > 0.5, 0.8, 0.1)).astype(float)
    tu, tp = isotonic_fit(u, y)
    assert np.all(np.diff(tp) >= -1e-12)
    assert np.all(np.diff(tu) > 0)
    assert isotonic_predict(tu, tp, 0.2) == pytest.approx(0.1, abs=0.05)
    assert isotonic_predict(tu, tp, 0.8) == pytest.approx(0.8, abs=0.05)


def test_isotonic_pools_violators_by_hand() -> None:
    tu, tp = isotonic_fit(np.array([0.0, 1.0, 2.0, 3.0]), np.array([0.0, 1.0, 0.0, 1.0]))
    # The violating pair (1, 0) pools to 0.5 at u = 1.5.
    np.testing.assert_allclose(tp, [0.0, 0.5, 1.0])
    np.testing.assert_allclose(tu, [0.0, 1.5, 3.0])


def test_auroc_matches_pair_counting() -> None:
    rng = np.random.default_rng(1)
    s = rng.normal(size=300)
    y = (rng.uniform(size=300) < 1 / (1 + np.exp(-2 * s))).astype(float)
    pos, neg = s[y > 0.5], s[y < 0.5]
    brute = np.mean([(p > n) + 0.5 * (p == n) for p in pos for n in neg])
    assert auroc(s, y) == pytest.approx(brute, abs=1e-9)


def test_reliability_of_perfect_calibration_is_near_zero_ece() -> None:
    rng = np.random.default_rng(2)
    p = rng.uniform(size=200_000)
    y = (rng.uniform(size=200_000) < p).astype(float)
    rel = reliability(p, y)
    assert rel["ece"] < 0.01
    assert len(rel["bins"]) == 10


def test_governor_caps_then_brakes() -> None:
    a = jnp.array([0.2, 1.0])
    np.testing.assert_allclose(governed_action(a, jnp.float32(0.1), 0.2), a)  # below τ: untouched
    np.testing.assert_allclose(
        governed_action(a, jnp.float32(0.3), 0.2), [0.2, 0.5], atol=1e-6
    )  # capped
    np.testing.assert_allclose(
        governed_action(a, jnp.float32(0.5), 0.2), [0.2, -0.4], atol=1e-6
    )  # brake

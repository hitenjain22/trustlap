"""Track generation and validation (SPEC 6.1, 12.1)."""

import math

import numpy as np
import pytest

from trustlap.config import CONFIG
from trustlap.track import (
    REASON_PINCH,
    REASON_SELF_INTERSECTION,
    REASON_TOO_SHORT,
    REASON_TOO_TIGHT,
    _closed_polyline_self_intersects,
    build_track,
    generate_track,
    make_bank,
    validate,
)

CFG = CONFIG.track


def _ring(radii: list[float]) -> np.ndarray:
    n = len(radii)
    return np.array(
        [
            [r * math.cos(2 * math.pi * i / n), r * math.sin(2 * math.pi * i / n)]
            for i, r in enumerate(radii)
        ]
    )


@pytest.mark.parametrize("chunk", range(10))
def test_500_seeds_are_valid(chunk: int) -> None:
    for seed in range(chunk * 50, (chunk + 1) * 50):
        track = generate_track(seed + 7_000_000)
        assert validate(track) is None
        assert CFG.length_min_m <= track.length_m <= CFG.length_max_m
        assert 1.0 / np.max(np.abs(track.curvature)) >= CFG.min_radius_m
        assert track.n_points <= CFG.n_max


def test_closed_loop_and_uniform_spacing() -> None:
    track = generate_track(1234)
    c = track.centerline
    steps = np.linalg.norm(np.roll(c, -1, axis=0) - c, axis=1)  # includes the closing step
    assert abs(track.ds_m - CFG.ds_target_m) < 0.01 * CFG.ds_target_m
    # Chords are slightly shorter than arcs on curves, never longer.
    assert np.all(steps <= track.ds_m + 1e-9)
    assert np.all(steps > 0.99 * track.ds_m)


def test_frames_are_orthonormal_and_left_handed() -> None:
    track = generate_track(99)
    t, n = track.tangent, track.normal
    np.testing.assert_allclose(np.linalg.norm(t, axis=1), 1.0, atol=1e-12)
    np.testing.assert_allclose(np.sum(t * n, axis=1), 0.0, atol=1e-12)
    np.testing.assert_allclose(t[:, 0] * n[:, 1] - t[:, 1] * n[:, 0], 1.0, atol=1e-12)


def test_curvature_on_a_counterclockwise_circle() -> None:
    track = build_track(_ring([150.0] * 24), 12.0, 0, 0)
    assert abs(track.length_m - 2 * math.pi * 150) < 1.0
    # Positive = turning left. A spline through 24 points is only nearly circular (±4%).
    assert abs(np.mean(track.curvature) - 1 / 150) < 0.005 / 150
    np.testing.assert_allclose(track.curvature, 1 / 150, rtol=0.05)


def test_generation_is_deterministic() -> None:
    a, b = generate_track(42), generate_track(42)
    np.testing.assert_array_equal(a.centerline, b.centerline)
    assert (a.width_m, a.attempt, a.smooth_iters) == (b.width_m, b.attempt, b.smooth_iters)


def test_rejects_figure_eight() -> None:
    t = np.linspace(0, 2 * math.pi, 16, endpoint=False)
    pts = np.stack([300 * np.sin(t), 150 * np.sin(t) * np.cos(t)], axis=1)
    assert validate(build_track(pts, 12.0, 0, 0)) in (REASON_PINCH, REASON_SELF_INTERSECTION)


def test_rejects_pinch() -> None:
    pts = []
    for i in range(16):
        a = 2 * math.pi * i / 16
        y = 160 * math.sin(a) * (0.04 if abs(math.cos(a)) < 0.3 else 1.0)
        pts.append([320 * math.cos(a), y])
    assert validate(build_track(np.array(pts), 12.0, 0, 0)) == REASON_PINCH


def test_rejects_too_tight() -> None:
    star = _ring([250.0 if i % 2 == 0 else 120.0 for i in range(14)])
    assert validate(build_track(star, 12.0, 0, 0)) == REASON_TOO_TIGHT


def test_rejects_too_short() -> None:
    assert validate(build_track(_ring([100.0] * 10), 12.0, 0, 0)) == REASON_TOO_SHORT


def test_self_intersection_detector() -> None:
    bowtie = np.array([[0, 0], [10, 10], [10, 0], [0, 10]], float)
    square = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], float)
    assert _closed_polyline_self_intersects(bowtie)
    assert not _closed_polyline_self_intersects(square)


def test_bank_padding() -> None:
    tracks = [generate_track(s) for s in (1, 2, 3)]
    bank = make_bank(tracks)
    assert bank.centerline.shape == (3, CFG.n_max, 2)
    assert bank.centerline.dtype == np.float32
    for i, tr in enumerate(tracks):
        np.testing.assert_allclose(bank.centerline[i, : tr.n_points], tr.centerline, atol=1e-3)
        assert np.all(bank.centerline[i, tr.n_points :] == 0)

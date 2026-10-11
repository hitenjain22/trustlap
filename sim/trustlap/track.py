"""Procedural race tracks (SPEC 6.1), ported line by line to `web/src/sim/track.ts`.

A track is a closed centerline loop resampled at ~2 m spacing, with tangents, left normals,
curvature and a constant width. Generation runs once, offline, in float64 NumPy; the JAX env
indexes into a padded float32 bank built from many tracks.

The order of floating-point operations here is part of the contract with the TypeScript port
(both are float64, compared to 1e-6 m): keep explicit formulas instead of library helpers
whose internals may differ (e.g. `np.interp`, `np.hypot`).
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path
from typing import NamedTuple

import numpy as np

from trustlap.config import CONFIG, TrackConfig
from trustlap.rng import PCG32, splitmix32

# Validation failure reasons (stable strings; the web maps them to user-facing messages).
REASON_TOO_SHORT = "too_short"
REASON_TOO_LONG = "too_long"
REASON_TOO_TIGHT = "too_tight"
REASON_SELF_INTERSECTION = "self_intersection"
REASON_PINCH = "pinch"


@dataclass(frozen=True)
class Track:
    """One generated track, float64. Arrays have n_points rows."""

    seed: int  # requested seed
    attempt: int  # 0 if the first try was valid, else the retry number that succeeded
    smooth_iters: int  # Laplacian smoothing passes applied to the control points
    control_points: np.ndarray  # [n_ctrl, 2] m
    centerline: np.ndarray  # [N, 2] m
    tangent: np.ndarray  # [N, 2] unit
    normal: np.ndarray  # [N, 2] unit, tangent rotated +90° (points left)
    curvature: np.ndarray  # [N] 1/m, positive = turning left
    width_m: float
    length_m: float

    @property
    def n_points(self) -> int:
        return int(self.centerline.shape[0])

    @property
    def ds_m(self) -> float:
        """Exact point spacing (length / n_points, within 0.1% of 2 m)."""
        return self.length_m / self.n_points


class TrackBank(NamedTuple):
    """Padded track arrays for the JAX env. Index modulo n_points so padding is never read."""

    centerline: np.ndarray  # [T, N_MAX, 2] float32
    tangent: np.ndarray  # [T, N_MAX, 2] float32
    normal: np.ndarray  # [T, N_MAX, 2] float32
    curvature: np.ndarray  # [T, N_MAX] float32
    width: np.ndarray  # [T] float32, m
    length: np.ndarray  # [T] float32, m
    n_points: np.ndarray  # [T] int32
    seeds: np.ndarray  # [T] int64
    attempts: np.ndarray  # [T] int32
    smooth_iters: np.ndarray  # [T] int32


# ---------------------------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------------------------


def control_points_from_seed(
    seed: int, cfg: TrackConfig = CONFIG.track
) -> tuple[np.ndarray, float]:
    """Random star-shaped control polygon and track width for one seed (steps 1–4 and 8).

    RNG draw order: n; n angle jitters; n radii; per point a pull coin (and a pull amount if
    it lands); width.
    """
    rng = PCG32(seed)
    n = rng.next_int(cfg.n_ctrl_min, cfg.n_ctrl_max)
    spacing = 2.0 * math.pi / n
    angles = [
        2.0 * math.pi * i / n + rng.next_range(-cfg.angle_jitter, cfg.angle_jitter) * spacing
        for i in range(n)
    ]
    angles.sort()
    radii = [rng.next_range(cfg.radius_min_m, cfg.radius_max_m) for _ in range(n)]
    for i in range(n):
        if rng.next_float01() < cfg.pull_prob:
            radii[i] = radii[i] - rng.next_range(cfg.pull_min, cfg.pull_max) * radii[i]
    width = rng.next_range(cfg.width_min_m, cfg.width_max_m)
    pts = np.array(
        [[radii[i] * math.cos(angles[i]), radii[i] * math.sin(angles[i])] for i in range(n)]
    )
    return pts, width


def catmull_rom_closed(points: np.ndarray, samples_per_segment: int, alpha: float) -> np.ndarray:
    """Dense samples of a closed centripetal Catmull–Rom spline through `points` [n, 2].

    Barry–Goldman pyramid form. Segment i runs from points[i] to points[i+1]; each segment
    contributes `samples_per_segment` samples starting at its first point. Returns [n·S, 2].
    """
    n = points.shape[0]
    out = np.empty((n * samples_per_segment, 2))
    for i in range(n):
        p0 = points[(i - 1) % n]
        p1 = points[i]
        p2 = points[(i + 1) % n]
        p3 = points[(i + 2) % n]
        t0 = 0.0
        t1 = t0 + _knot(p0, p1, alpha)
        t2 = t1 + _knot(p1, p2, alpha)
        t3 = t2 + _knot(p2, p3, alpha)
        for k in range(samples_per_segment):
            t = t1 + (t2 - t1) * (k / samples_per_segment)
            a1 = (t1 - t) / (t1 - t0) * p0 + (t - t0) / (t1 - t0) * p1
            a2 = (t2 - t) / (t2 - t1) * p1 + (t - t1) / (t2 - t1) * p2
            a3 = (t3 - t) / (t3 - t2) * p2 + (t - t2) / (t3 - t2) * p3
            b1 = (t2 - t) / (t2 - t0) * a1 + (t - t0) / (t2 - t0) * a2
            b2 = (t3 - t) / (t3 - t1) * a2 + (t - t1) / (t3 - t1) * a3
            out[i * samples_per_segment + k] = (t2 - t) / (t2 - t1) * b1 + (t - t1) / (t2 - t1) * b2
    return out


def _knot(a: np.ndarray, b: np.ndarray, alpha: float) -> float:
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    return math.sqrt(dx * dx + dy * dy) ** alpha


def resample_closed(dense: np.ndarray, ds_target: float) -> tuple[np.ndarray, float]:
    """Resample a closed polyline at uniform arclength spacing (step 6).

    N = round(L / ds_target) points at s_k = k · L / N. Returns (points [N, 2], length L).
    """
    m = dense.shape[0]
    cum = np.empty(m + 1)  # cumulative arclength including the closing segment
    cum[0] = 0.0
    for j in range(m):
        a = dense[j]
        b = dense[(j + 1) % m]
        dx = b[0] - a[0]
        dy = b[1] - a[1]
        cum[j + 1] = cum[j] + math.sqrt(dx * dx + dy * dy)
    length = float(cum[m])
    n = int(round(length / ds_target))
    ds = length / n
    out = np.empty((n, 2))
    j = 0
    for k in range(n):
        s = k * ds
        while cum[j + 1] < s:
            j += 1
        a = dense[j]
        b = dense[(j + 1) % m]
        f = (s - cum[j]) / (cum[j + 1] - cum[j])
        out[k, 0] = a[0] + f * (b[0] - a[0])
        out[k, 1] = a[1] + f * (b[1] - a[1])
    return out, length


def frames(
    centerline: np.ndarray, ds: float, smooth_window: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Tangents (central differences), left normals and smoothed curvature (step 7).

    Raw curvature at i is the signed turning angle from tangent i to tangent i+1, divided by
    ds; then a circular moving average over `smooth_window` points.
    """
    n = centerline.shape[0]
    tangent = np.empty((n, 2))
    for i in range(n):
        a = centerline[(i - 1) % n]
        b = centerline[(i + 1) % n]
        dx = b[0] - a[0]
        dy = b[1] - a[1]
        norm = math.sqrt(dx * dx + dy * dy)
        tangent[i, 0] = dx / norm
        tangent[i, 1] = dy / norm
    normal = np.empty((n, 2))
    normal[:, 0] = -tangent[:, 1]
    normal[:, 1] = tangent[:, 0]
    raw = np.empty(n)
    for i in range(n):
        t0 = tangent[i]
        t1 = tangent[(i + 1) % n]
        cross = t0[0] * t1[1] - t0[1] * t1[0]
        dot = t0[0] * t1[0] + t0[1] * t1[1]
        raw[i] = math.atan2(cross, dot) / ds
    half = smooth_window // 2
    curvature = np.empty(n)
    for i in range(n):
        acc = 0.0
        for k in range(-half, half + 1):
            acc += raw[(i + k) % n]
        curvature[i] = acc / smooth_window
    return tangent, normal, curvature


def laplacian_smooth(points: np.ndarray, lam: float) -> np.ndarray:
    """One Laplacian smoothing pass over a closed polygon: p_i += λ · (mean of neighbours − p_i)."""
    n = points.shape[0]
    out = np.empty_like(points)
    for i in range(n):
        a = points[(i - 1) % n]
        b = points[(i + 1) % n]
        p = points[i]
        out[i, 0] = p[0] + lam * ((a[0] + b[0]) / 2.0 - p[0])
        out[i, 1] = p[1] + lam * ((a[1] + b[1]) / 2.0 - p[1])
    return out


def build_track(
    control_points: np.ndarray,
    width_m: float,
    seed: int,
    attempt: int,
    smooth_iters: int = 0,
    cfg: TrackConfig = CONFIG.track,
) -> Track:
    """Spline, resample and compute frames for a control polygon (steps 5–7). No validation."""
    dense = catmull_rom_closed(control_points, cfg.samples_per_segment, cfg.catmull_rom_alpha)
    centerline, length = resample_closed(dense, cfg.ds_target_m)
    ds = length / centerline.shape[0]
    tangent, normal, curvature = frames(centerline, ds, cfg.curvature_smooth_window)
    return Track(
        seed=seed,
        attempt=attempt,
        smooth_iters=smooth_iters,
        control_points=control_points,
        centerline=centerline,
        tangent=tangent,
        normal=normal,
        curvature=curvature,
        width_m=width_m,
        length_m=length,
    )


# ---------------------------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------------------------


def validate(track: Track, cfg: TrackConfig = CONFIG.track) -> str | None:
    """Return None if the track is valid, else the first failing reason (step 9).

    Checks run cheapest first: length, minimum turning radius, non-adjacent clearance
    ("pinch"), then boundary self-intersection.
    """
    if track.length_m < cfg.length_min_m:
        return REASON_TOO_SHORT
    if track.length_m > cfg.length_max_m or track.n_points > cfg.n_max:
        return REASON_TOO_LONG
    if float(np.max(np.abs(track.curvature))) > 1.0 / cfg.min_radius_m:
        return REASON_TOO_TIGHT
    if not _clearance_ok(track, cfg):
        return REASON_PINCH
    if _boundaries_intersect(track, cfg):
        return REASON_SELF_INTERSECTION
    return None


def _clearance_ok(track: Track, cfg: TrackConfig) -> bool:
    """Every pair of points more than `clearance_index_gap` apart (circularly) is at least
    width + clearance_extra_m apart in space."""
    c = track.centerline
    n = c.shape[0]
    idx = np.arange(n)
    gap = np.abs(idx[:, None] - idx[None, :])
    gap = np.minimum(gap, n - gap)
    dx = c[:, None, 0] - c[None, :, 0]
    dy = c[:, None, 1] - c[None, :, 1]
    d2 = dx * dx + dy * dy
    min_d = track.width_m + cfg.clearance_extra_m
    return not bool(np.any((gap > cfg.clearance_index_gap) & (d2 < min_d * min_d)))


def boundaries(track: Track) -> tuple[np.ndarray, np.ndarray]:
    """Left and right boundary polylines: centerline ± W/2 · normal."""
    half = track.width_m / 2.0
    return track.centerline + half * track.normal, track.centerline - half * track.normal


def _boundaries_intersect(track: Track, cfg: TrackConfig) -> bool:
    left, right = boundaries(track)
    left = left[:: cfg.validate_stride]
    right = right[:: cfg.validate_stride]
    return (
        _closed_polyline_self_intersects(left)
        or _closed_polyline_self_intersects(right)
        or _closed_polylines_cross(left, right)
    )


def _segments(poly: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return poly, np.roll(poly, -1, axis=0)


def _cross_pairs(a0: np.ndarray, a1: np.ndarray, b0: np.ndarray, b1: np.ndarray) -> np.ndarray:
    """Proper intersection matrix between segment sets A [m] and B [k] (touching excluded)."""

    def orient(p: np.ndarray, q: np.ndarray, r: np.ndarray) -> np.ndarray:
        return (q[..., 0] - p[..., 0]) * (r[..., 1] - p[..., 1]) - (q[..., 1] - p[..., 1]) * (
            r[..., 0] - p[..., 0]
        )

    p0, p1 = a0[:, None, :], a1[:, None, :]
    q0, q1 = b0[None, :, :], b1[None, :, :]
    d1 = orient(p0, p1, q0)
    d2 = orient(p0, p1, q1)
    d3 = orient(q0, q1, p0)
    d4 = orient(q0, q1, p1)
    return (d1 * d2 < 0) & (d3 * d4 < 0)


def _closed_polyline_self_intersects(poly: np.ndarray) -> bool:
    a0, a1 = _segments(poly)
    hits = _cross_pairs(a0, a1, a0, a1)
    m = poly.shape[0]
    idx = np.arange(m)
    gap = np.abs(idx[:, None] - idx[None, :])
    gap = np.minimum(gap, m - gap)
    return bool(np.any(hits & (gap > 1)))


def _closed_polylines_cross(p: np.ndarray, q: np.ndarray) -> bool:
    a0, a1 = _segments(p)
    b0, b1 = _segments(q)
    return bool(np.any(_cross_pairs(a0, a1, b0, b1)))


# ---------------------------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------------------------


class TrackGenerationError(RuntimeError):
    """Raised when no valid track is found in max_attempts tries (means the parameters are off)."""


def build_valid_track(
    control_points: np.ndarray,
    width_m: float,
    seed: int,
    attempt: int,
    cfg: TrackConfig = CONFIG.track,
) -> tuple[Track, str | None]:
    """Build a track, auto-fixing corners that are too tight by Laplacian-smoothing the control
    points (λ = 0.3, up to 10 passes). Shared by generated and hand-drawn tracks.

    Returns the last track built and its validation result (None = valid).
    """
    pts = control_points
    for it in range(cfg.autofix_max_iters + 1):
        track = build_track(pts, width_m, seed, attempt, it, cfg)
        reason = validate(track, cfg)
        if reason != REASON_TOO_TIGHT or it == cfg.autofix_max_iters:
            return track, reason
        pts = laplacian_smooth(pts, cfg.autofix_lambda)
    raise AssertionError("unreachable")


def generate_track(seed: int, cfg: TrackConfig = CONFIG.track) -> Track:
    """Generate a valid track for `seed` (step 10).

    Each attempt draws a control polygon from a sub-seed (the seed itself, then
    splitmix32(seed ^ attempt)) and auto-fixes tight corners; see DECISIONS.md (2026-10-11).
    """
    seed &= 0xFFFFFFFF
    for attempt in range(cfg.max_attempts + 1):
        sub_seed = seed if attempt == 0 else splitmix32(seed ^ attempt)
        pts, width = control_points_from_seed(sub_seed, cfg)
        track, reason = build_valid_track(pts, width, seed, attempt, cfg)
        if reason is None:
            return track
    raise TrackGenerationError(f"no valid track for seed {seed} in {cfg.max_attempts} attempts")


def make_bank(tracks: list[Track], n_max: int = CONFIG.track.n_max) -> TrackBank:
    """Pack tracks into padded float32 arrays."""
    t = len(tracks)
    bank = TrackBank(
        centerline=np.zeros((t, n_max, 2), np.float32),
        tangent=np.zeros((t, n_max, 2), np.float32),
        normal=np.zeros((t, n_max, 2), np.float32),
        curvature=np.zeros((t, n_max), np.float32),
        width=np.array([tr.width_m for tr in tracks], np.float32),
        length=np.array([tr.length_m for tr in tracks], np.float32),
        n_points=np.array([tr.n_points for tr in tracks], np.int32),
        seeds=np.array([tr.seed for tr in tracks], np.int64),
        attempts=np.array([tr.attempt for tr in tracks], np.int32),
        smooth_iters=np.array([tr.smooth_iters for tr in tracks], np.int32),
    )
    for i, tr in enumerate(tracks):
        n = tr.n_points
        bank.centerline[i, :n] = tr.centerline
        bank.tangent[i, :n] = tr.tangent
        bank.normal[i, :n] = tr.normal
        bank.curvature[i, :n] = tr.curvature
    return bank


def bank_from_seeds(seeds: list[int], workers: int | None = None) -> TrackBank:
    """Generate tracks for `seeds` (in parallel processes) and pack them into a bank."""
    if workers == 1 or len(seeds) < 16:
        tracks = [generate_track(s) for s in seeds]
    else:
        with Pool(workers) as pool:
            tracks = pool.map(generate_track, seeds, chunksize=8)
    return make_bank(tracks)


def save_bank(bank: TrackBank, path: Path) -> None:
    """Save a bank as a compressed .npz."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **bank._asdict())


def load_bank(path: Path) -> TrackBank:
    """Load a bank saved by `save_bank`."""
    with np.load(path) as data:
        return TrackBank(**{k: data[k] for k in TrackBank._fields})


def main() -> None:
    """CLI: build the train / validation / test banks into an output directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("runs/banks"))
    parser.add_argument("--banks", nargs="+", default=["train", "validation", "test"])
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()
    for name in args.banks:
        start, stop = getattr(CONFIG.banks, name)
        t0 = time.perf_counter()
        bank = bank_from_seeds(list(range(start, stop)), args.workers)
        save_bank(bank, args.out / f"{name}.npz")
        attempts = np.bincount(bank.attempts)
        print(
            f"{name}: {len(bank.seeds)} tracks in {time.perf_counter() - t0:.1f}s, "
            f"length {bank.length.min():.0f}–{bank.length.max():.0f} m, "
            f"attempt histogram {attempts.tolist()}"
        )


if __name__ == "__main__":
    main()

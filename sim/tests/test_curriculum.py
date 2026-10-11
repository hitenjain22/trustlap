"""Curriculum controller (SPEC 12.1)."""

from trustlap.curriculum import Curriculum


def run_checks(cur: Curriculum, success: float, checks: int) -> list[float]:
    out = []
    for _ in range(checks):
        for _ in range(cur.cfg.check_interval_updates):
            cur.record(100, 100 * success)
        cur.step()
        out.append(cur.difficulty)
    return out


def test_rises_only_after_two_consecutive_good_checks() -> None:
    cur = Curriculum()
    assert run_checks(cur, 0.9, 4) == [0.0, 0.05, 0.05, 0.1]


def test_hysteresis_resets_on_a_middling_check() -> None:
    cur = Curriculum()
    run_checks(cur, 0.9, 1)
    run_checks(cur, 0.7, 1)  # between thresholds: no move, counter reset
    assert run_checks(cur, 0.9, 1) == [0.0]


def test_falls_with_low_success_and_clamps() -> None:
    cur = Curriculum(difficulty=0.1)
    assert run_checks(cur, 0.2, 3) == [0.05, 0.0, 0.0]
    cur = Curriculum(difficulty=1.0)
    assert run_checks(cur, 0.95, 4)[-1] == 1.0


def test_checks_only_every_interval() -> None:
    cur = Curriculum()
    for _ in range(cur.cfg.check_interval_updates - 1):
        cur.record(100, 100)
        assert not cur.step()


def test_env_arrays_easy_and_stretch_split() -> None:
    cur = Curriculum(difficulty=0.5)
    d, s = cur.env_arrays(100)
    assert (d[:20] == 0).all() and (d[20:] == 0.5).all()
    assert s[20:30].all() and not s[:20].any() and not s[30:].any()

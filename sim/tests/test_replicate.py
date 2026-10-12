"""Combining training seeds: the primary seed's numbers stay; agreement is counted honestly."""

from trustlap.replicate import combine


def _metrics(dry_ci: list[float], wet_ci: list[float]) -> dict:
    return {
        "headline": {"student_crash_reduction_vs_blind_combined": {"value": 0.05, "ci": [0, 0.1]}},
        "calibration": {"auroc": 0.7, "baseline": {"auroc": 0.58}},
        "paired_deltas": [
            {
                "suite": "dry",
                "a": "student",
                "b": "blind",
                "metric": "completion",
                "value": -0.3,
                "ci": dry_ci,
            },
            {
                "suite": "wet",
                "a": "student",
                "b": "blind",
                "metric": "completion",
                "value": 0.3,
                "ci": wet_ci,
            },
            {
                "suite": "wet",
                "a": "student",
                "b": "blind",
                "metric": "crash",
                "value": -0.3,
                "ci": [-0.4, -0.2],
            },
        ],
    }


def test_combine_keeps_primary_and_counts_agreement() -> None:
    per_seed = {
        1: _metrics([-0.4, -0.2], [0.2, 0.4]),
        2: _metrics([-0.1, 0.05], [0.1, 0.3]),
        3: _metrics([-0.5, -0.3], [0.25, 0.35]),
    }
    out = combine(per_seed, primary=1)
    assert out["paired_deltas"] == per_seed[1]["paired_deltas"]
    rep = out["replication"]
    assert rep["primary_seed"] == 1 and rep["seeds"] == [1, 2, 3]
    assert rep["agreement"]["dry"] == {"primary": "worse", "same_in": 2, "of": 3}
    assert rep["agreement"]["wet"] == {"primary": "better", "same_in": 3, "of": 3}
    assert rep["per_seed"]["1"]["risk_auroc_disagreement_only"] == 0.58

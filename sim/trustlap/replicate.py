"""Combine independent training seeds into one metrics file.

The published networks and every chart come from the primary seed (fixed before any result).
The `replication` block adds, for each training seed, the numbers a reader needs to judge whether a
finding is robust: the Adaptive-vs-Basic verdict per condition, the combined-conditions crash
change, and the risk model's AUROC. A verdict "holds" in a seed when its paired 95% interval
excludes zero in the same direction.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _verdict(ci: list[float | None]) -> str:
    lo, hi = ci
    if lo is not None and lo > 0:
        return "better"
    if hi is not None and hi < 0:
        return "worse"
    return "same"


def seed_summary(metrics: dict[str, Any]) -> dict[str, Any]:
    """The robustness-relevant numbers from one seed's evaluation."""
    suites: dict[str, Any] = {}
    for d in metrics.get("paired_deltas", []):
        if d["a"] == "student" and d["b"] == "blind" and d["metric"] == "completion":
            suites[d["suite"]] = {
                "delta": d["value"],
                "ci": d["ci"],
                "verdict": _verdict(d["ci"]),
            }
    head = metrics.get("headline", {})
    cal = metrics.get("calibration", {})
    return {
        "suites": suites,
        "crash_reduction_combined": head.get("student_crash_reduction_vs_blind_combined"),
        "risk_auroc": cal.get("auroc"),
        "risk_auroc_disagreement_only": (cal.get("baseline") or {}).get("auroc"),
        "puddles_noticed": (head.get("puddle_latency_ms") or {}).get("noticed"),
    }


def combine(per_seed: dict[int, dict[str, Any]], primary: int) -> dict[str, Any]:
    """Primary seed's metrics plus a replication block over every seed."""
    out = json.loads(json.dumps(per_seed[primary]))
    summaries = {s: seed_summary(m) for s, m in per_seed.items()}
    suites = sorted({k for s in summaries.values() for k in s["suites"]})
    agreement = {}
    for suite in suites:
        verdicts = [summaries[s]["suites"].get(suite, {}).get("verdict") for s in summaries]
        primary_v = summaries[primary]["suites"].get(suite, {}).get("verdict")
        agreement[suite] = {
            "primary": primary_v,
            "same_in": sum(v == primary_v for v in verdicts),
            "of": len(verdicts),
        }
    out["replication"] = {
        "seeds": sorted(per_seed),
        "primary_seed": primary,
        "per_seed": {str(s): summaries[s] for s in sorted(summaries)},
        "agreement": agreement,
    }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--primary", type=int, default=1)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--eval-dir", default="eval", help="per-seed evaluation folder (eval, or eval_ft)"
    )
    args = parser.parse_args()
    per_seed = {
        s: json.loads((args.root / f"seed{s}" / args.eval_dir / "metrics_partial.json").read_text())
        for s in args.seeds
    }
    args.out.write_text(json.dumps(combine(per_seed, args.primary), indent=1))
    print(f"wrote {args.out} (primary seed {args.primary}, {len(per_seed)} seeds)")


if __name__ == "__main__":
    main()

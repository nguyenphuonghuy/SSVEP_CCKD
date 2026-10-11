"""Quantify pure run-to-run nondeterminism (GPU/cuDNN), isolated from any
RNG/config difference: 5 independent process runs of the IDENTICAL config
(benchmark, split_seed=2026, seed=2026, lambda_sp=100, relational, FB-tCNN).

Any spread across the 5 repeats here cannot come from seed, split, data, or
code differences -- it is the noise floor that deltas elsewhere in the paper
should be compared against.
"""
import argparse
import glob
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1] / "results" / "module2"
KEYS = ["teacher_9C", "3C_baseline", "3C_relational", "1C_baseline", "1C_relational"]


def load(root, i):
    files = glob.glob(str(root / f"rep{i}" / "*.json"))
    assert len(files) == 1, (i, files)
    with open(files[0]) as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="benchmark", choices=["benchmark", "beta"])
    args = ap.parse_args()
    root = ROOT / ("tier7_determinism" if args.dataset == "benchmark" else "tier7_determinism_beta")
    runs = [load(root, i) for i in range(1, 6)]
    print(f"Repeat-to-repeat spread, 5 identical runs ({args.dataset}, seed=2026, split=2026, lambda_sp=100):")
    for key in KEYS:
        accs = np.array([d["runs"][key]["mean_acc"] * 100 for d in runs])
        print(f"  {key:16s} " + " ".join(f"{a:.4f}" for a in accs) +
              f"   range={accs.max()-accs.min():.4f}pp  SD={accs.std(ddof=1):.4f}pp")
    for key in ["3C_relational", "1C_relational"]:
        deltas = np.array([d["runs"][key]["paired_vs_baseline"]["mean_delta_pp"] for d in runs])
        print(f"  delta[{key}] " + " ".join(f"{x:+.4f}" for x in deltas) +
              f"   range={deltas.max()-deltas.min():.4f}pp  SD={deltas.std(ddof=1):.4f}pp")


if __name__ == "__main__":
    main()

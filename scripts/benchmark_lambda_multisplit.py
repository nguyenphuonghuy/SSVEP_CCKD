"""Robustness of the Benchmark lambda_sp comparison across 5 subject splits.

Compares lambda_sp = 1 (selected by validation on split 2026) with lambda_sp = 100
(headline setting). Both use FB-tCNN, window 1.0 s, relational only, 8 seeds per split.
Per split, delta = mean relational - mean baseline (pp), taken from the same run.
"""
import glob
import json
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1] / "results" / "module2"
SPLITS = [2026, 3026, 4026, 5026, 6026]
LAMBDAS = [1, 100]
CONFIGS = ["3C", "1C"]


def run_dir(lam, split):
    if split == 2026:
        return ROOT / "tier6_lambda_val" / f"benchmark_lam{lam}_s2026_8seed"
    return ROOT / "tier6_lambda_ms" / f"benchmark_lam{lam}_s{split}_8seed"


def load(lam, split):
    files = glob.glob(str(run_dir(lam, split) / "*.json"))
    assert len(files) == 1, (lam, split, files)
    with open(files[0]) as f:
        return json.load(f)


def main():
    for lam in LAMBDAS:
        print(f"== lambda_sp = {lam}")
        for cfg in CONFIGS:
            key = f"{cfg}_relational"
            deltas = []
            for split in SPLITS:
                d = load(lam, split)
                assert d["config"]["split_seed"] == split and d["config"]["lambda_sp"] == float(lam)
                deltas.append(d["runs"][key]["paired_vs_baseline"]["mean_delta_pp"])
            deltas = np.array(deltas)
            _, p = stats.ttest_1samp(deltas, 0.0)
            print(f"  {cfg}: " + " ".join(f"{x:+.2f}" for x in deltas) +
                f"  mean {deltas.mean():+.2f} ± {deltas.std(ddof=1):.2f}  "
                f"pos {(deltas > 0).sum()}/5  p_t={p:.4f}")


if __name__ == "__main__":
    main()

"""Multi-split window-length analysis for CCKD (FB-tCNN, relational, lambda_sp=100).

For each (dataset, window_s, configuration) collects the per-split paired
delta (relational vs baseline, mean over 8 seeds) across 5 subject splits,
runs a one-sample t-test over the 5 split-level deltas, and applies
Holm-Bonferroni over the full family of 16 tests.

Split 2026 comes from results/module2/tier2_window (single canonical split),
splits 3026-6026 from results/module2/tier2_window_ms, and the 1.0 s window
from results/module2/tier0_multisplit (same configuration).
"""
import glob
import json
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1] / "results" / "module2"
SPLITS = [2026, 3026, 4026, 5026, 6026]
WINDOWS = [0.5, 1.0, 1.5, 2.0]
DATASETS = ["benchmark", "beta"]
CONFIGS = ["3C_relational", "1C_relational"]
WTAG = {0.5: "w05", 1.5: "w15", 2.0: "w20"}


def load_single(pattern):
    files = glob.glob(str(pattern))
    assert len(files) == 1, (pattern, files)
    with open(files[0]) as f:
        return json.load(f)


def load_run(dataset, window, split):
    if window == 1.0:
        return load_single(ROOT / "tier0_multisplit" / f"{dataset}_s{split}_8seed" / "*.json")
    if split == 2026:
        return load_single(ROOT / "tier2_window" / f"{dataset}_{WTAG[window]}_8seed" / "*.json")
    return load_single(ROOT / "tier2_window_ms" / f"{dataset}_{WTAG[window]}_s{split}_8seed" / "*.json")


def holm(pvals):
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adjusted = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvals[i]))
        adjusted[i] = running
    return adjusted


def main():
    rows = []
    for dataset in DATASETS:
        for window in WINDOWS:
            for cfg in CONFIGS:
                deltas = []
                for split in SPLITS:
                    d = load_run(dataset, window, split)
                    assert d["config"]["window_s"] == window and d["config"]["split_seed"] == split
                    assert len(d["config"]["seeds"]) == 8
                    deltas.append(d["runs"][cfg]["paired_vs_baseline"]["mean_delta_pp"])
                deltas = np.array(deltas)
                _, p = stats.ttest_1samp(deltas, 0.0)
                rows.append({
                    "dataset": dataset, "window": window, "cfg": cfg,
                    "deltas": deltas.tolist(), "mean": float(deltas.mean()),
                    "sd": float(deltas.std(ddof=1)), "p_t": float(p),
                    "n_pos": int((deltas > 0).sum()),
                })

    for row, p_holm in zip(rows, holm([r["p_t"] for r in rows])):
        row["p_holm"] = p_holm
        print(f"{row['dataset']:9s} {row['window']:.1f}s {row['cfg']:14s} "
              + " ".join(f"{x:+.2f}" for x in row["deltas"])
              + f"  mean {row['mean']:+.2f} ± {row['sd']:.2f}  pos {row['n_pos']}/5"
              + f"  p_t={row['p_t']:.4f}  p_Holm={row['p_holm']:.4f}")


if __name__ == "__main__":
    main()

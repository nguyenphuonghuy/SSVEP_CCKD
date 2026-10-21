"""Validation-based lambda_sp selection for Module 3 N2b runs (several lambdas per job).

    python scripts/select_lambda_module3.py results/module3/n2b [--out results/module3/n2b_selected.csv]

For every dataset x backbone x student x split, picks the lambda_sp with the highest mean
validation accuracy over the 8 seeds (test set not used), then reports the five-split gain at
the selected lambda and, for comparison, at each fixed lambda. Ties are broken toward the
smaller lambda.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import ttest_1samp

SPLITS = [2026, 3026, 4026, 5026, 6026]


def stats(d):
    return {"mean": float(np.mean(d)), "sd": float(np.std(d, ddof=1)) if len(d) > 1 else float("nan"),
            "pos": int(sum(x > 0 for x in d)), "p": float(ttest_1samp(d, 0).pvalue) if len(d) > 2 else float("nan")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--mode", default="relational")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    # cell[(ds, bb, student)][split][lambda] = (mean_val, delta_pp)
    cell = defaultdict(lambda: defaultdict(dict))
    for f in glob.glob(f"{a.root}/**/cckd_*.json", recursive=True):
        r = json.load(open(f))
        c = r["config"]
        ds = Path(f).name.split("_")[2]
        for k, v in r["runs"].items():
            if "@lsp" not in k or f"_{a.mode}" not in k:
                continue
            student = k.split(f"_{a.mode}")[0]
            lsp = float(k.split("@lsp")[1])
            parts = k.split("@")
            kl = parts[1] if len(parts) == 3 else "+".join(c.get("kd_layers", ["legacy"]))
            val = float(np.mean(r["val_best_per_seed"][k]))
            cell[(ds, c["backbone"], kl, student)][c["split_seed"]][lsp] = (val, v["paired_vs_baseline"]["mean_delta_pp"])
    rows = []
    for (ds, bb, kl, st), per in sorted(cell.items()):
        splits = [s for s in SPLITS if s in per]
        lambdas = sorted({l for s in splits for l in per[s]})
        chosen = {s: max(sorted(per[s]), key=lambda l: per[s][l][0]) for s in splits}
        sel = stats([per[s][chosen[s]][1] for s in splits])
        row = {"dataset": ds, "backbone": bb, "kd_layers": kl, "student": st, "n_splits": len(splits),
               "chosen_lambda": " ".join(f"{chosen[s]:g}" for s in splits),
               "sel_mean_pp": round(sel["mean"], 2), "sel_sd_pp": round(sel["sd"], 2),
               "sel_pos": sel["pos"], "sel_p": round(sel["p"], 4)}
        for l in lambdas:
            d = [per[s][l][1] for s in splits if l in per[s]]
            st_l = stats(d)
            row[f"lsp{l:g}_mean_pp"] = round(st_l["mean"], 2)
            row[f"lsp{l:g}_p"] = round(st_l["p"], 4)
        rows.append(row)
        print(f"{ds:9s} {bb:14s} {kl:7s} {st:7s} selected [{row['chosen_lambda']}] -> {sel['mean']:+.2f} +/- {sel['sd']:.2f} "
              f"({sel['pos']}/{len(splits)}, p={sel['p']:.4f}) | fixed: " +
              " ".join(f"{l:g}:{row[f'lsp{l:g}_mean_pp']:+.2f}" for l in lambdas))
    if a.out and rows:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        keys = sorted({k for r in rows for k in r}, key=lambda k: list(rows[0]).index(k) if k in rows[0] else 99)
        with open(a.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        print(f"written {a.out}")


if __name__ == "__main__":
    main()

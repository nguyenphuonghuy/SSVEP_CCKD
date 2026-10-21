"""Aggregate Module 3 runs: five-split CCKD gain and teacher-student CKA per setting.

    python scripts/aggregate_module3.py results/module3/n2a [--out results/module3/n2a_summary.csv]

Reads every cckd_*.json under the given directory. Groups by (dataset, backbone, kd_layers,
lambda_sp, run key) and reports the per-split paired gain, the mean +/- SD over splits, the
number of positive splits, the one-sample t-test over splits, and the mean CKA (mid, last) of
baseline and distilled students. Incomplete groups (fewer than 5 splits) are reported as such.
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    groups = defaultdict(dict)
    for f in glob.glob(f"{a.root}/**/cckd_*.json", recursive=True):
        r = json.load(open(f))
        c = r["config"]
        kl = "+".join(c.get("kd_layers", ["legacy"]))
        ds = Path(f).name.split("_")[2]
        for k, v in r["runs"].items():
            if "paired_vs_baseline" not in v:
                continue
            student = k.split("_")[0] if "@" not in k else k.split("_")[0]
            base = r["runs"][f"{student}_baseline"] if f"{student}_baseline" in r["runs"] else \
                r["runs"][k.rsplit("_", 1)[0].split("@")[0].rsplit("_", 1)[0] + "_baseline"]
            g = (ds, c["backbone"], kl, c["lambda_sp"], k)
            groups[g][c["split_seed"]] = {
                "delta": v["paired_vs_baseline"]["mean_delta_pp"],
                "acc": v["mean_acc"] * 100, "base_acc": base["mean_acc"] * 100,
                "cka": v.get("cka_mean", {}), "cka_base": base.get("cka_mean", {}),
            }
    rows = []
    for (ds, bb, kl, lsp, k), per in sorted(groups.items()):
        d = [per[s]["delta"] for s in SPLITS if s in per]
        row = {"dataset": ds, "backbone": bb, "kd_layers": kl, "lambda_sp": lsp, "run": k,
               "n_splits": len(d), "deltas_pp": " ".join(f"{x:+.2f}" for x in d),
               "mean_pp": round(float(np.mean(d)), 2),
               "sd_pp": round(float(np.std(d, ddof=1)), 2) if len(d) > 1 else "",
               "pos_splits": sum(x > 0 for x in d),
               "t_p": round(float(ttest_1samp(d, 0).pvalue), 4) if len(d) > 2 else "",
               "base_acc": round(float(np.mean([per[s]["base_acc"] for s in per])), 2),
               "kd_acc": round(float(np.mean([per[s]["acc"] for s in per])), 2)}
        for lvl in ("mid", "last"):
            cb = [per[s]["cka_base"].get(lvl) for s in per if per[s]["cka_base"].get(lvl) is not None]
            ck = [per[s]["cka"].get(lvl) for s in per if per[s]["cka"].get(lvl) is not None]
            row[f"cka_base_{lvl}"] = round(float(np.mean(cb)), 3) if cb else ""
            row[f"cka_kd_{lvl}"] = round(float(np.mean(ck)), 3) if ck else ""
            row[f"dcka_{lvl}"] = round(float(np.mean(ck) - np.mean(cb)), 3) if cb and ck else ""
        rows.append(row)
    for r in rows:
        flag = "" if r["n_splits"] == 5 else f"  (only {r['n_splits']} splits)"
        print(f"{r['dataset']:9s} {r['backbone']:14s} {r['kd_layers']:9s} lsp={r['lambda_sp']:<6g} {r['run']:22s} "
              f"{r['mean_pp']:+.2f} +/- {r['sd_pp']} ({r['pos_splits']}/{r['n_splits']}, p={r['t_p']}) "
              f"[{r['deltas_pp']}] dCKA last={r['dcka_last']} mid={r['dcka_mid']}{flag}")
    if a.out and rows:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        with open(a.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"written {a.out}")


if __name__ == "__main__":
    main()

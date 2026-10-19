"""Single-split vs five-split verdicts for CCKD (relational, lambda_sp=100, 1.0 s), Module 2 data.

    python scripts/single_vs_multisplit.py [--results results/module2] [--out results/module3/n6_single_vs_multi.csv]

For every backbone x dataset x student setting, compares the verdict a study would draw from
split 2026 alone (Wilcoxon over test subjects, 8 seeds) with the verdict from five subject
splits (one-sample t-test over the five split-level gains). Verdict: "+" (p<0.05, gain>0),
"-" (p<0.05, gain<0), "0" (p>=0.05).
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
from pathlib import Path

import numpy as np
from scipy.stats import ttest_1samp

ROOT = Path(__file__).resolve().parents[1]
SPLITS = [2026, 3026, 4026, 5026, 6026]
# backbone -> list of (tier dir glob pattern) searched for each split
SOURCES = {
    "fbtcnn": ["tier0_multisplit/{ds}_s{s}_8seed"],
    "eegnet": ["tier1b_eegnet_multisplit/{ds}_s{s}_8seed"],
    "deepconvnet": ["deepconvnet_confirm_{ds}_8seed", "tier1_multisplit/deepconvnet_{ds}_s{s}_8seed"],
    "shallowconvnet": ["shallowconvnet_confirm_{ds}_8seed", "tier1_multisplit/shallowconvnet_{ds}_s{s}_8seed"],
    "compactcnn": ["compactcnn_confirm_{ds}_8seed", "tier1_multisplit/compactcnn_{ds}_s{s}_8seed"],
}


def find(results, backbone, ds, s):
    for pat in SOURCES[backbone]:
        for f in glob.glob(f"{results}/{pat.format(ds=ds, s=s)}/cckd_*.json"):
            r = json.load(open(f))
            c = r["config"]
            if c["split_seed"] == s and c["backbone"] == backbone and abs(c["window_s"] - 1.0) < 1e-9 \
                    and c["lambda_sp"] == 100.0 and len(c["seeds"]) == 8:
                return r, f
    raise SystemExit(f"missing {backbone} {ds} split {s}")


def verdict(gain, p):
    return "0" if p >= 0.05 else ("+" if gain > 0 else "-")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(ROOT / "results" / "module2"))
    ap.add_argument("--out", default=str(ROOT / "results" / "module3" / "n6_single_vs_multi.csv"))
    a = ap.parse_args()
    rows = []
    for bb in SOURCES:
        for ds in ("benchmark", "beta"):
            for c in ("3C", "1C"):
                runs = [find(a.results, bb, ds, s) for s in SPLITS]
                st0 = runs[0][0]["runs"][f"{c}_relational"]["paired_vs_baseline"]
                d = [r["runs"][f"{c}_relational"]["paired_vs_baseline"]["mean_delta_pp"] for r, _ in runs]
                pt = float(ttest_1samp(d, 0.0).pvalue)
                v1, v5 = verdict(st0["mean_delta_pp"], st0["wilcoxon_p"]), verdict(np.mean(d), pt)
                rows.append({"backbone": bb, "dataset": ds, "student": c,
                             "single_gain_pp": round(st0["mean_delta_pp"], 2), "single_wilcoxon_p": round(st0["wilcoxon_p"], 4),
                             "single_verdict": v1, "multi_mean_pp": round(float(np.mean(d)), 2),
                             "multi_sd_pp": round(float(np.std(d, ddof=1)), 2), "multi_pos_splits": sum(x > 0 for x in d),
                             "multi_t_p": round(pt, 4), "multi_verdict": v5, "agree": v1 == v5,
                             "sources": ";".join(str(Path(f).relative_to(ROOT)) for _, f in runs)})
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(f"{r['backbone']:15s} {r['dataset']:9s} {r['student']}: single {r['single_gain_pp']:+.2f} "
              f"(p={r['single_wilcoxon_p']:.3f}) [{r['single_verdict']}] | 5 splits {r['multi_mean_pp']:+.2f}"
              f"+/-{r['multi_sd_pp']:.2f} ({r['multi_pos_splits']}/5, p={r['multi_t_p']:.4f}) [{r['multi_verdict']}]"
              f"{'' if r['agree'] else '  <- differs'}")
    k = sum(not r["agree"] for r in rows)
    print(f"\nverdict differs in {k} of {len(rows)} settings; written to {a.out}")


if __name__ == "__main__":
    main()

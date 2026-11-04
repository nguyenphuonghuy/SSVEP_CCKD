"""Module 3 RQ1/RQ1b/RQ2/RQ3 analysis, computed entirely from committed result JSONs
(N2a, N2b, N5, N3). No GPU needed. Run:

    python scripts/module3_rq_analysis.py

Writes CSVs to results/module3/analysis/ and prints a short summary per RQ.

Lambda-per-backbone table below is the one actually used to generate the N5/N3 jobs
(majority vote of the per-split validation-selected lambda from N2b; see
results/module3/n2b_selected.csv for the per-split detail).
"""
from __future__ import annotations

import csv
import glob
import json
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr, ttest_1samp, wilcoxon

ROOT = Path(__file__).resolve().parents[1] / "results" / "module3"
OUT = ROOT / "analysis"
OUT.mkdir(parents=True, exist_ok=True)

SPLITS = [2026, 3026, 4026, 5026, 6026]
DATASETS = ["benchmark", "beta"]
BACKBONES = ["fbtcnn", "deepconvnet", "eegnet", "shallowconvnet", "compactcnn"]
STUDENTS = ["3C", "1C"]
LAMBDA = {
    ("benchmark", "fbtcnn"): 100, ("benchmark", "deepconvnet"): 100, ("benchmark", "eegnet"): 10,
    ("benchmark", "shallowconvnet"): 10, ("benchmark", "compactcnn"): 10,
    ("beta", "fbtcnn"): 100, ("beta", "deepconvnet"): 1000, ("beta", "eegnet"): 10,
    ("beta", "shallowconvnet"): 100, ("beta", "compactcnn"): 10,
}


def load(path_glob):
    files = glob.glob(str(path_glob))
    assert len(files) == 1, (path_glob, files)
    with open(files[0]) as f:
        return json.load(f)


def n2_json(tier, dataset, backbone, split):
    if backbone == "fbtcnn" and tier == "n2":
        return load(ROOT / "n2a" / f"{dataset}_fbtcnn_last_lsp100_s{split}" / "cckd_*.json")
    return load(ROOT / "n2b" / f"{dataset}_{backbone}_last_lsp1-10-100-1000_s{split}" / "cckd_*.json")


def relational_key(backbone, student, lam):
    if backbone == "fbtcnn":
        return f"{student}_relational"
    return f"{student}_relational@lsp{lam}"


def holm(pvals):
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvals[i]))
        adj[i] = running
    return adj


# ---------------------------------------------------------------------------
# RQ1: backbone effect at a FIXED lambda_sp=100 (fair cross-backbone comparison)
# ---------------------------------------------------------------------------
def rq1():
    rows = []
    for dataset in DATASETS:
        for backbone in BACKBONES:
            for student in STUDENTS:
                deltas = []
                for split in SPLITS:
                    d = n2_json("n2", dataset, backbone, split)
                    assert d["config"]["split_seed"] == split
                    key = relational_key(backbone, student, 100)
                    deltas.append(d["runs"][key]["paired_vs_baseline"]["mean_delta_pp"])
                deltas = np.array(deltas)
                _, p = ttest_1samp(deltas, 0.0)
                rows.append({"dataset": dataset, "backbone": backbone, "student": student,
                             "mean_delta_pp": float(deltas.mean()), "sd_pp": float(deltas.std(ddof=1)),
                             "n_pos": int((deltas > 0).sum()), "p_t": float(p)})
    p_holm = holm([r["p_t"] for r in rows])
    for r, ph in zip(rows, p_holm):
        r["p_holm"] = ph
    with open(OUT / "rq1_backbone_effect_lsp100.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    n_sig_pos = sum(1 for r in rows if r["p_t"] < 0.05 and r["mean_delta_pp"] > 0)
    n_sig_neg = sum(1 for r in rows if r["p_t"] < 0.05 and r["mean_delta_pp"] < 0)
    n_holm_pos = sum(1 for r in rows if r["p_holm"] < 0.05 and r["mean_delta_pp"] > 0)
    n_holm_neg = sum(1 for r in rows if r["p_holm"] < 0.05 and r["mean_delta_pp"] < 0)
    print(f"RQ1 (lambda_sp=100, {len(rows)} cells): raw p<0.05 -> {n_sig_pos} up / {n_sig_neg} down / "
          f"{len(rows)-n_sig_pos-n_sig_neg} none. After Holm ({len(rows)} tests): {n_holm_pos} up / {n_holm_neg} down.")
    return rows


# ---------------------------------------------------------------------------
# RQ1b: montage sensitivity (N3), compared at the SAME validation-selected lambda
# ---------------------------------------------------------------------------
ALT_MAP = {"3C": ["3C_PPO", "3C_POO"], "1C": ["1C_POz", "1C_O1"]}


def rq1b():
    rows = []
    for dataset in DATASETS:
        for backbone in BACKBONES:
            lam = LAMBDA[(dataset, backbone)]
            for student in STUDENTS:
                orig_deltas = []
                for split in SPLITS:
                    d = n2_json("n2", dataset, backbone, split)
                    key = relational_key(backbone, student, lam)
                    orig_deltas.append(d["runs"][key]["paired_vs_baseline"]["mean_delta_pp"])
                orig_mean = float(np.mean(orig_deltas))
                for alt in ALT_MAP[student]:
                    alt_deltas = []
                    for split in SPLITS:
                        d = load(ROOT / "n3" / f"{dataset}_{backbone}_altmontage_lsp{lam}_s{split}" / "cckd_*.json")
                        alt_deltas.append(d["runs"][f"{alt}_relational"]["paired_vs_baseline"]["mean_delta_pp"])
                    alt_deltas = np.array(alt_deltas)
                    _, p = ttest_1samp(alt_deltas, 0.0)
                    rows.append({"dataset": dataset, "backbone": backbone, "lambda_sp": lam,
                                 "orig_student": student, "orig_mean_delta_pp": orig_mean,
                                 "alt_montage": alt, "alt_mean_delta_pp": float(alt_deltas.mean()),
                                 "alt_sd_pp": float(alt_deltas.std(ddof=1)),
                                 "alt_n_pos": int((alt_deltas > 0).sum()), "alt_p_t": float(p),
                                 "sign_flip": (orig_mean > 0) != (float(alt_deltas.mean()) > 0)})
    with open(OUT / "rq1b_montage.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    n_flip = sum(1 for r in rows if r["sign_flip"])
    print(f"RQ1b (montage, {len(rows)} alt-montage cells vs original): {n_flip} sign flips vs original montage.")
    return rows


# ---------------------------------------------------------------------------
# RQ2a: CKA (last layer) vs accuracy gain, at each backbone's selected lambda
# ---------------------------------------------------------------------------
def mean_cka_delta(d, baseline_key, relational_key_, level):
    b = np.array(d["cka_per_seed"][baseline_key][level])
    r = np.array(d["cka_per_seed"][relational_key_][level])
    return float((r - b).mean()), float(b.mean())


def rq2a():
    rows = []
    for dataset in DATASETS:
        for backbone in BACKBONES:
            lam = LAMBDA[(dataset, backbone)]
            for student in STUDENTS:
                dcka_list, dacc_list, base_cka_list = [], [], []
                for split in SPLITS:
                    d = n2_json("n2", dataset, backbone, split)
                    key = relational_key(backbone, student, lam)
                    dacc_list.append(d["runs"][key]["paired_vs_baseline"]["mean_delta_pp"])
                    dcka, base_cka = mean_cka_delta(d, f"{student}_baseline", key, "last")
                    dcka_list.append(dcka); base_cka_list.append(base_cka)
                rows.append({"dataset": dataset, "backbone": backbone, "student": student,
                             "lambda_sp": lam, "mean_delta_acc_pp": float(np.mean(dacc_list)),
                             "mean_delta_cka_last": float(np.mean(dcka_list)),
                             "mean_baseline_cka_last": float(np.mean(base_cka_list))})
    with open(OUT / "rq2a_cka_correlation.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    dacc = [r["mean_delta_acc_pp"] for r in rows]
    dcka = [r["mean_delta_cka_last"] for r in rows]
    base_cka = [r["mean_baseline_cka_last"] for r in rows]
    r1, p1 = pearsonr(dcka, dacc); rho1, pr1 = spearmanr(dcka, dacc)
    r2, p2 = pearsonr(base_cka, dacc); rho2, pr2 = spearmanr(base_cka, dacc)
    print(f"RQ2a (n={len(rows)}): corr(delta_CKA_last, delta_Acc): Pearson r={r1:.3f} p={p1:.4f}, "
          f"Spearman rho={rho1:.3f} p={pr1:.4f}")
    print(f"       corr(baseline_CKA_last, delta_Acc): Pearson r={r2:.3f} p={p2:.4f}, "
          f"Spearman rho={rho2:.3f} p={pr2:.4f}")
    return rows


# ---------------------------------------------------------------------------
# RQ2b: layer intervention. Compare distilling LAST only (N2) vs MID only vs BOTH (N5),
# at each backbone's selected lambda, and whether delta_CKA at the distilled level
# tracks delta_Acc better than at the other level.
# ---------------------------------------------------------------------------
def rq2b():
    rows = []
    for dataset in DATASETS:
        for backbone in BACKBONES:
            lam = LAMBDA[(dataset, backbone)]
            for student in STUDENTS:
                last_acc, mid_acc, both_acc = [], [], []
                last_cka_last, mid_cka_mid, both_cka_last, both_cka_mid = [], [], [], []
                for split in SPLITS:
                    d2 = n2_json("n2", dataset, backbone, split)
                    k_last = relational_key(backbone, student, lam)
                    last_acc.append(d2["runs"][k_last]["paired_vs_baseline"]["mean_delta_pp"])
                    dcka, _ = mean_cka_delta(d2, f"{student}_baseline", k_last, "last")
                    last_cka_last.append(dcka)

                    d5 = load(ROOT / "n5" / f"{dataset}_{backbone}_mid-both_lsp{lam}_s{split}" / "cckd_*.json")
                    k_mid = f"{student}_relational@mid"; k_both = f"{student}_relational@both"
                    mid_acc.append(d5["runs"][k_mid]["paired_vs_baseline"]["mean_delta_pp"])
                    both_acc.append(d5["runs"][k_both]["paired_vs_baseline"]["mean_delta_pp"])
                    dcka_mid, _ = mean_cka_delta(d5, f"{student}_baseline", k_mid, "mid")
                    mid_cka_mid.append(dcka_mid)
                    dcka_both_last, _ = mean_cka_delta(d5, f"{student}_baseline", k_both, "last")
                    dcka_both_mid, _ = mean_cka_delta(d5, f"{student}_baseline", k_both, "mid")
                    both_cka_last.append(dcka_both_last); both_cka_mid.append(dcka_both_mid)
                rows.append({
                    "dataset": dataset, "backbone": backbone, "student": student, "lambda_sp": lam,
                    "acc_last_only": float(np.mean(last_acc)), "acc_mid_only": float(np.mean(mid_acc)),
                    "acc_both": float(np.mean(both_acc)),
                    "cka_last_at_last_distill": float(np.mean(last_cka_last)),
                    "cka_mid_at_mid_distill": float(np.mean(mid_cka_mid)),
                    "cka_last_at_both_distill": float(np.mean(both_cka_last)),
                    "cka_mid_at_both_distill": float(np.mean(both_cka_mid)),
                })
    with open(OUT / "rq2b_layer_intervention.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    last_better = sum(1 for r in rows if r["acc_last_only"] > r["acc_mid_only"])
    both_best = sum(1 for r in rows if r["acc_both"] >= max(r["acc_last_only"], r["acc_mid_only"]))
    print(f"RQ2b (n={len(rows)}): last-only beats mid-only accuracy in {last_better}/{len(rows)} cells; "
          f"distilling both layers is the best (or tied) of the three in {both_best}/{len(rows)} cells.")
    return rows


# ---------------------------------------------------------------------------
# RQ3: single-split (2026) vs 5-split verdict mismatch, at lambda_sp=100 (RQ1 protocol)
# ---------------------------------------------------------------------------
def rq3_single_vs_multi(rq1_rows):
    rows = []
    for dataset in DATASETS:
        for backbone in BACKBONES:
            for student in STUDENTS:
                d = n2_json("n2", dataset, backbone, 2026)
                key = relational_key(backbone, student, 100)
                st = d["runs"][key]["paired_vs_baseline"]
                single_delta = st["mean_delta_pp"]; single_p = st["wilcoxon_p"]
                multi = next(r for r in rq1_rows if r["dataset"] == dataset and r["backbone"] == backbone
                             and r["student"] == student)
                single_sig = single_p < 0.05
                multi_sig = multi["p_t"] < 0.05
                sign_mismatch = (single_delta > 0) != (multi["mean_delta_pp"] > 0)
                sig_mismatch = single_sig != multi_sig
                rows.append({"dataset": dataset, "backbone": backbone, "student": student,
                             "single_split_delta_pp": single_delta, "single_split_p": single_p,
                             "multi_split_mean_delta_pp": multi["mean_delta_pp"], "multi_split_p_t": multi["p_t"],
                             "sign_mismatch": sign_mismatch, "significance_mismatch": sig_mismatch,
                             "any_mismatch": sign_mismatch or sig_mismatch})
    with open(OUT / "rq3_single_vs_multisplit.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    k = sum(1 for r in rows if r["any_mismatch"])
    print(f"RQ3 single-vs-5-split (lambda_sp=100, n={len(rows)}): mismatch (sign or significance) in {k}/{len(rows)} cells.")
    return rows


if __name__ == "__main__":
    print("=" * 78)
    rq1_rows = rq1()
    print("=" * 78)
    rq1b()
    print("=" * 78)
    rq2a()
    print("=" * 78)
    rq2b()
    print("=" * 78)
    rq3_single_vs_multi(rq1_rows)
    print("=" * 78)
    print(f"CSVs written to {OUT}")

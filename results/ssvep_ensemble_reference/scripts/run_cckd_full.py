"""
run_cckd_full.py — Chay TOAN BO thi nghiem CCKD trong 1 process (an toan cho PBS).

Giong tinh than run_beta_full.py: khong input(), khong tuong tac, load data 1 lan,
chay tuan tu, tu thoat. Ket qua + log ghi ra file.

Ma tran (mac dinh = "smoke/quyet dinh"): Benchmark, cua so 1.0s, backbone fbtcnn,
  teacher 9C -> student {1C, 3C} x {baseline, hybrid KD}, lap qua nhieu seed khoi tao,
  tren MOT phep chia subject-independent CO DINH (test 10 subject).

Chay HPC:      python run_cckd_full.py
Test pipeline: python run_cckd_full.py --use_dummy
"""
import os
import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import torch

sys.path.insert(0, str(Path.cwd()))

from data.benchmark_loader import BenchmarkDataset, create_dummy_data
from models.backbones import build_backbone, count_params
import cckd_pipeline as P

# ── CAU HINH (sua o day, giong run_beta_full.py) ──
DATASET     = "benchmark"
T_START     = 0.14
WINDOW_S    = 0.5           # cua so vat ly
BACKBONE    = "fbtcnn"      # 'fbtcnn' (chinh) | 'tcnn' | 'eegnet'
N_BANDS     = 3
SPLIT_SEED  = 2026
SEEDS       = [2026, 2027, 2028]     # 3 seed khoi tao (phep chia van co dinh)
MODES       = ["sp"]                 # SP da tang; do luong sach vs baseline
STUDENTS    = ["3C", "1C"]

# Cau hinh rieng theo dataset. LUU Y sfreq loader KHAC NHAU:
#   Benchmark loader sfreq=1000 -> mau = t*1000 -> 1.0s can t_end=0.39
#   BETA      loader sfreq=250  -> mau = t*250  -> 1.0s can t_end=1.14
if DATASET == "beta":
    DATA_PATH, N_SUBJECTS, SPLIT = "data/raw/beta", 70, (42, 8, 20)      # test 20 subject
    T_END = 0.14 + WINDOW_S           # 1.14 -> 250 mau @ sfreq=250 = 1.0s
else:
    DATA_PATH, N_SUBJECTS, SPLIT = "data/raw/benchmark", 35, (21, 4, 10)  # test 10 subject
    T_END = 0.14 + WINDOW_S / 4.0     # 0.39 -> 250 mau @ sfreq=1000 = 1.0s
SAVE_DIR = f"results/cckd_{DATASET}_{WINDOW_S}s"

CONFIG = {
    "epochs": 100, "batch_size": 64, "lr": 1e-3, "weight_decay": 1e-4,
    "tau": 4.0,
    "lambda_kd": 1.0,      # chi dung khi mode='hybrid'
    "lambda_sp": 100.0,    # lam_sp*SP ~ ngang CE (da kiem tra tren data that)
}


def log(msg=""):
    print(msg, flush=True)


def build(cfg_key, n_samples, n_classes):
    C = len(P.CH_POS[cfg_key])
    return build_backbone(BACKBONE, C, n_samples, n_classes, n_bands=N_BANDS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--use_dummy", action="store_true")
    args = ap.parse_args()
    global CONFIG, SEEDS

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()
    log("=" * 64); log("  CCKD FULL RUN")
    log(f"  Device: {device}" + (f" | {torch.cuda.get_device_name(0)}" if device.type == "cuda" else ""))
    log(f"  Dataset={DATASET} | Backbone={BACKBONE} bands={N_BANDS} window={WINDOW_S}s split={SPLIT} seeds={SEEDS}")
    log("=" * 64)

    # ── Load data 1 lan ──
    if args.use_dummy:
        log("  [DUMMY] create_dummy_data + rut gon (test pipeline)")
        X, y, s = create_dummy_data(n_subjects=12, n_channels=9, n_samples=250)
        CONFIG = {**CONFIG, "epochs": 4}
        SEEDS = [2026]
        split = (7, 2, 3)
    else:
        if DATASET == "beta":
            from data.beta_loader import BETADataset
            loader = BETADataset(DATA_PATH, t_start=T_START, t_end=T_END, verbose=False)
        else:
            loader = BenchmarkDataset(DATA_PATH, t_start=T_START, t_end=T_END, verbose=False)
        X, y, s = loader.load_all_subjects(n_subjects=N_SUBJECTS)
        split = SPLIT
    n_samples, n_classes = X.shape[2], len(np.unique(y))
    log(f"  Data: X={X.shape} | n_samples={n_samples} ({n_samples/P.FS_PHYS:.2f}s vat ly) | classes={n_classes}")

    tr_s, va_s, te_s = P.fixed_split(s, *split, split_seed=SPLIT_SEED)
    log(f"  Split CO DINH: train={list(tr_s)} val={list(va_s)} test={list(te_s)}\n")
    m_tr, m_va, m_te = P.masks_for(s, tr_s), P.masks_for(s, va_s), P.masks_for(s, te_s)

    # ── Tien tinh input (filter bank) 1 lan cho moi cau hinh kenh ──
    log("  Chuan bi input (filter bank neu fbtcnn)...")
    inp = {}
    for k in ["9C"] + STUDENTS:
        Xk = P.subset(X, k)
        inp[k] = {
            "tr": (P.prep_input(Xk[m_tr], BACKBONE, N_BANDS), y[m_tr]),
            "va": (P.prep_input(Xk[m_va], BACKBONE, N_BANDS), y[m_va], s[m_va]),
            "te": (P.prep_input(Xk[m_te], BACKBONE, N_BANDS), y[m_te], s[m_te]),
        }
    log(f"    9C train input: {inp['9C']['tr'][0].shape}\n")

    Path(SAVE_DIR).mkdir(parents=True, exist_ok=True)
    results = {"config": {"backbone": BACKBONE, "window_s": WINDOW_S, "split": split,
                          "seeds": SEEDS, "n_samples": int(n_samples)}, "runs": {}}
    # per-config per-seed per-subject acc
    store = {f"{c}_baseline": [] for c in STUDENTS}
    for c in STUDENTS:
        for md in MODES:
            store[f"{c}_{md}"] = []
    teacher_te = []

    for seed in SEEDS:
        log(f"───── SEED {seed} ─────")
        np.random.seed(seed); torch.manual_seed(seed)

        # Teacher 9C
        Xtr, ytr = inp["9C"]["tr"]; Xva, yva, sva = inp["9C"]["va"]
        teacher = build("9C", n_samples, n_classes)
        teacher, tva = P.train_ce(teacher, Xtr, ytr, Xva, yva, sva, CONFIG, device)
        t_acc = P.eval_per_subject(teacher, *inp["9C"]["te"], device)
        teacher_te.append(t_acc)
        log(f"  teacher-9C: val={tva:.4f} test={np.mean(list(t_acc.values())):.4f} "
            f"params={count_params(teacher)}")

        for c in STUDENTS:
            Xtr_c, ytr_c = inp[c]["tr"]; Xva_c, yva_c, sva_c = inp[c]["va"]
            # baseline
            base = build(c, n_samples, n_classes)
            base, bva = P.train_ce(base, Xtr_c, ytr_c, Xva_c, yva_c, sva_c, CONFIG, device)
            b_acc = P.eval_per_subject(base, *inp[c]["te"], device)
            store[f"{c}_baseline"].append(b_acc)
            log(f"  {c} baseline: test={np.mean(list(b_acc.values())):.4f}")
            # KD modes
            for md in MODES:
                stud = build(c, n_samples, n_classes)
                stud, kva = P.train_kd(stud, teacher, Xtr_c, inp["9C"]["tr"][0], ytr_c,
                                       Xva_c, yva_c, sva_c, CONFIG, device, mode=md)
                k_acc = P.eval_per_subject(stud, *inp[c]["te"], device)
                store[f"{c}_{md}"].append(k_acc)
                log(f"  {c} {md}: test={np.mean(list(k_acc.values())):.4f}")
        log("")

    # ── Tong hop qua seed ──
    log("=" * 64); log("  KET QUA (mean±std qua seed, tren test set co dinh)"); log("=" * 64)
    results["runs"]["teacher_9C"] = P.summarize(teacher_te, WINDOW_S)
    log(f"  teacher-9C: {results['runs']['teacher_9C']['mean_acc']*100:.2f}% "
        f"(ITR={results['runs']['teacher_9C']['itr']:.1f})")
    for c in STUDENTS:
        base_sum = P.summarize(store[f"{c}_baseline"], WINDOW_S)
        results["runs"][f"{c}_baseline"] = base_sum
        log(f"\n  {c} baseline: {base_sum['mean_acc']*100:.2f}% ± {base_sum['std_acc']*100:.2f} "
            f"(ITR={base_sum['itr']:.1f})")
        for md in MODES:
            kd_sum = P.summarize(store[f"{c}_{md}"], WINDOW_S)
            results["runs"][f"{c}_{md}"] = kd_sum
            gain = (kd_sum["mean_acc"] - base_sum["mean_acc"]) * 100
            st = P.paired_stats(store[f"{c}_baseline"], store[f"{c}_{md}"])
            results["runs"][f"{c}_{md}"]["paired_vs_baseline"] = st
            sig = "***" if st["wilcoxon_p"] < 0.05 else ""
            log(f"  {c} {md}:    {kd_sum['mean_acc']*100:.2f}% ± {kd_sum['std_acc']*100:.2f} "
                f"(ITR={kd_sum['itr']:.1f}) | gain {gain:+.2f}pp")
            log(f"        paired: delta={st['mean_delta_pp']:+.2f}pp "
                f"| cai thien {st['n_improved']}/{st['n_subjects']} subj "
                f"| Wilcoxon p={st['wilcoxon_p']:.4f} {sig}")

    # heuristic quyet dinh
    log("\n--- Doc ket qua ---")
    g = {c: (P.summarize(store[f"{c}_{MODES[0]}"], WINDOW_S)["mean_acc"]
             - P.summarize(store[f"{c}_baseline"], WINDOW_S)["mean_acc"]) for c in STUDENTS}
    if all(v <= 0 for v in g.values()):
        log("  - KD khong giup -> check teacher, tau, lambda_sp, cach lay feature.")
    else:
        log("  + KD duong -> chay tiep ma tran (BETA, cua so khac, EEGNet).")
    if "1C" in g and "3C" in g and g["1C"] > g["3C"] + 1e-4:
        log("  ++ 1C huong loi > 3C -> cau chuyen 'thieu kenh -> KD cang gia tri'.")

    ts = datetime.now().strftime("%Y%m%d_%H%M")
    out = Path(SAVE_DIR) / f"cckd_{BACKBONE}_{WINDOW_S}s_{ts}.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    log(f"\n  Ket qua: {out}")
    log(f"  Tong thoi gian: {(time.time()-t0)/60:.1f} phut")


if __name__ == "__main__":
    main()

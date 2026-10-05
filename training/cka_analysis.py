"""Module 2 addendum -- Centered Kernel Alignment (CKA) diagnostic.

Trains teacher(9C), student(baseline CE), student(CCKD relational) for ONE
backbone/dataset/student-config combination, then measures linear CKA
similarity between teacher and student penultimate features on the held-out
test set. Goal: check whether CCKD's accuracy benefit (or lack thereof)
correlates with how much it increases teacher-student representational
similarity -- a quantitative, continuous diagnostic to complement the
discrete per-backbone accuracy comparison in the main results.

Single seed by design: this is an exploratory diagnostic over 5 backbones x
2 datasets x 2 student configs (20 points), not a confirmatory replication
of any single accuracy claim (those already exist elsewhere with 8 seeds x
5 splits). CKA values themselves are near-deterministic given a converged
model, so seed variance is a minor concern here relative to accuracy.

Usage:
    python training/cka_analysis.py --dataset benchmark --data_root ... \
        --backbone fbtcnn --output_dir results/module2/cka/fbtcnn_benchmark
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets.public_ssvep import PublicSSVEPDataset
from models.cckd_backbones import build_backbone, count_params
from utils.config import load_yaml, deep_get
import training.cckd_pipeline as P

DEFAULT_SPLIT = {"benchmark": (21, 4, 10), "beta": (42, 8, 20)}
DEFAULT_N_SUBJECTS = {"benchmark": 35, "beta": 70}


def log(msg: str = "") -> None:
    print(msg, flush=True)


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Linear CKA (Kornblith et al., 2019). X: [n, d1], Y: [n, d2]."""
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)
    hsic = np.linalg.norm(X.T @ Y, ord="fro") ** 2
    var1 = np.linalg.norm(X.T @ X, ord="fro")
    var2 = np.linalg.norm(Y.T @ Y, ord="fro")
    if var1 < 1e-12 or var2 < 1e-12:
        return float("nan")
    return float(hsic / (var1 * var2))


@torch.no_grad()
def extract_features(model, X: np.ndarray, device, batch: int = 512) -> np.ndarray:
    """Final penultimate feature level (last element if list, else the tensor)."""
    model.eval()
    xb = torch.from_numpy(X)
    feats = []
    for i in range(0, len(xb), batch):
        _, f = model(xb[i:i + batch].to(device), return_feat=True)
        if isinstance(f, (list, tuple)):
            f = f[-1]
        feats.append(f.flatten(1).cpu().numpy())
    return np.concatenate(feats, axis=0)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["benchmark", "beta"])
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--backbone", required=True,
                     choices=["fbtcnn", "eegnet", "deepconvnet", "compactcnn", "shallowconvnet"])
    ap.add_argument("--n_bands", type=int, default=3)
    ap.add_argument("--window_s", type=float, default=1.0)
    ap.add_argument("--t_start", type=float, default=0.14)
    ap.add_argument("--split_seed", type=int, default=2026)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--students", nargs="+", default=["3C", "1C"])
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--tau", type=float, default=4.0)
    ap.add_argument("--lambda_kd", type=float, default=1.0)
    ap.add_argument("--lambda_sp", type=float, default=100.0)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    return ap.parse_args()


def main():
    a = parse_args()
    device = torch.device("cuda" if (a.device in ("auto", "cuda") and torch.cuda.is_available()) else "cpu")

    cfg_path = a.config or str(ROOT / "configs" / f"{a.dataset}.yaml")
    cfg = load_yaml(cfg_path)
    ch_indices = deep_get(cfg, "dataset.teacher_channel_indices")
    mat_key = deep_get(cfg, "dataset.mat_key", "data")
    n_subjects = DEFAULT_N_SUBJECTS[a.dataset]
    split = DEFAULT_SPLIT[a.dataset]

    CONFIG = {"epochs": a.epochs, "batch_size": a.batch_size, "lr": a.lr,
              "weight_decay": a.weight_decay, "tau": a.tau,
              "lambda_kd": a.lambda_kd, "lambda_sp": a.lambda_sp}

    t0 = time.time()
    log("=" * 72)
    log("  CCKD-SSVEP -- CKA diagnostic (Tier 3)")
    log(f"  Device: {device}" + (f" | {torch.cuda.get_device_name(0)}" if device.type == "cuda" else ""))
    log(f"  Dataset={a.dataset} | Backbone={a.backbone} | seed={a.seed} split_seed={a.split_seed}")
    log("=" * 72)

    loader = PublicSSVEPDataset(a.data_root, a.dataset, sfreq=250.0, t_start=a.t_start,
                                 window_seconds=a.window_s, mat_key=mat_key, verbose=False)
    X64, y, s = loader.load_all_subjects(n_subjects)
    X = X64[:, ch_indices, :]
    n_samples, n_classes = X.shape[2], len(np.unique(y))

    tr_s, va_s, te_s = P.fixed_split(s, *split, split_seed=a.split_seed)
    m_tr, m_va, m_te = P.masks_for(s, tr_s), P.masks_for(s, va_s), P.masks_for(s, te_s)

    inp = {}
    for k in ["9C"] + a.students:
        Xk = P.subset(X, k)
        inp[k] = {
            "tr": (P.prep_input(Xk[m_tr], a.backbone, a.n_bands), y[m_tr]),
            "va": (P.prep_input(Xk[m_va], a.backbone, a.n_bands), y[m_va], s[m_va]),
            "te": (P.prep_input(Xk[m_te], a.backbone, a.n_bands), y[m_te], s[m_te]),
        }

    def build(cfg_key):
        C = len(P.CH_POS[cfg_key])
        return build_backbone(a.backbone, C, n_samples, n_classes, n_bands=a.n_bands)

    np.random.seed(a.seed); torch.manual_seed(a.seed)

    log("  Training teacher (9C)...")
    Xtr, ytr = inp["9C"]["tr"]; Xva, yva, sva = inp["9C"]["va"]
    teacher = build("9C")
    teacher, tva = P.train_ce(teacher, Xtr, ytr, Xva, yva, sva, CONFIG, device)
    t_acc = P.eval_per_subject(teacher, *inp["9C"]["te"], device)
    t_mean = float(np.mean(list(t_acc.values())))
    log(f"    teacher-9C: val={tva:.4f} test={t_mean:.4f} params={count_params(teacher)}")

    Xte_teacher = inp["9C"]["te"][0]
    feat_teacher = extract_features(teacher, Xte_teacher, device)

    out_dir = Path(a.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {
        "config": {"backbone": a.backbone, "dataset": a.dataset, "seed": a.seed,
                    "split_seed": a.split_seed, "epochs": a.epochs},
        "teacher_acc": t_mean,
        "students": {},
    }

    for c in a.students:
        log(f"  --- Student {c} ---")
        Xtr_c, ytr_c = inp[c]["tr"]; Xva_c, yva_c, sva_c = inp[c]["va"]

        base = build(c)
        base, bva = P.train_ce(base, Xtr_c, ytr_c, Xva_c, yva_c, sva_c, CONFIG, device)
        b_acc = P.eval_per_subject(base, *inp[c]["te"], device)
        b_mean = float(np.mean(list(b_acc.values())))
        feat_base = extract_features(base, inp[c]["te"][0], device)
        cka_base = linear_cka(feat_teacher, feat_base)
        log(f"    baseline: test={b_mean:.4f}  CKA(teacher, baseline)={cka_base:.4f}")

        stud = build(c)
        stud, kva = P.train_kd(stud, teacher, Xtr_c, inp["9C"]["tr"][0], ytr_c,
                                Xva_c, yva_c, sva_c, CONFIG, device, mode="relational")
        k_acc = P.eval_per_subject(stud, *inp[c]["te"], device)
        k_mean = float(np.mean(list(k_acc.values())))
        feat_cckd = extract_features(stud, inp[c]["te"][0], device)
        cka_cckd = linear_cka(feat_teacher, feat_cckd)
        log(f"    CCKD:     test={k_mean:.4f}  CKA(teacher, CCKD)={cka_cckd:.4f}")

        delta_acc = (k_mean - b_mean) * 100
        delta_cka = cka_cckd - cka_base
        log(f"    delta_acc={delta_acc:+.3f}pp  delta_CKA={delta_cka:+.4f}")

        results["students"][c] = {
            "acc_baseline": b_mean, "acc_cckd": k_mean, "delta_acc_pp": delta_acc,
            "cka_baseline": cka_base, "cka_cckd": cka_cckd, "delta_cka": delta_cka,
        }

    out_path = out_dir / f"cka_{a.backbone}_{a.dataset}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    log(f"\n  Saved: {out_path}")
    log(f"  Total time: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()

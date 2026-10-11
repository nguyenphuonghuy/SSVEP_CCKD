"""In-process counterpart to the cross-process nondeterminism check.

Repeats the IDENTICAL training (benchmark, split_seed=2026, seed=2026,
lambda_sp=100, relational, FB-tCNN) 5 times inside a single Python process /
single CUDA context, re-seeding before each repeat. Data loading and
preprocessing are done once and reused (they do not depend on the model seed).

If the spread here is much smaller than the 5-separate-process measurement
(scripts/measure_nondeterminism.py), the noise is likely tied to something
fixed once per process (e.g. a cuDNN algorithm/workspace choice). If the
spread is comparable, the noise is per-training-call (e.g. nondeterministic
backward-pass reduction order), and the original 8-seeds-in-one-process runs
do NOT average out this noise any more than independent processes would.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets.public_ssvep import PublicSSVEPDataset
from models.cckd_backbones import build_backbone
from utils.config import load_yaml, deep_get
import training.cckd_pipeline as P

DATA_ROOT = {
    "benchmark": "/gpfs0/users/gv_nphuy/projects/ssvep_ensemble/data/raw/benchmark",
    "beta": "/gpfs0/users/gv_nphuy/projects/ssvep_ensemble/data/raw/beta",
}
SPLIT = {"benchmark": (21, 4, 10), "beta": (42, 8, 20)}
N_SUBJECTS = {"benchmark": 35, "beta": 70}
SPLIT_SEED = 2026
SEED = 2026
WINDOW_S = 1.0
T_START = 0.14
CONFIG = {"epochs": 100, "batch_size": 64, "lr": 1e-3, "weight_decay": 1e-4,
          "tau": 4.0, "lambda_kd": 1.0, "lambda_sp": 100.0}
N_REPEATS = 5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="benchmark", choices=["benchmark", "beta"])
    args = ap.parse_args()
    DATASET = args.dataset
    OUT = ROOT / "results" / "module2" / "tier7_determinism" / f"inprocess_{DATASET}.json"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_yaml(str(ROOT / "configs" / f"{DATASET}.yaml"))
    ch_indices = deep_get(cfg, "dataset.teacher_channel_indices")
    mat_key = deep_get(cfg, "dataset.mat_key", "data")

    loader = PublicSSVEPDataset(DATA_ROOT[DATASET], DATASET, sfreq=250.0, t_start=T_START,
                                 window_seconds=WINDOW_S, mat_key=mat_key, verbose=False)
    X64, y, s = loader.load_all_subjects(N_SUBJECTS[DATASET])
    X = X64[:, ch_indices, :]
    n_samples, n_classes = X.shape[2], len(np.unique(y))

    tr_s, va_s, te_s = P.fixed_split(s, *SPLIT[DATASET], split_seed=SPLIT_SEED)
    m_tr, m_va, m_te = P.masks_for(s, tr_s), P.masks_for(s, va_s), P.masks_for(s, te_s)

    inp = {}
    for k in ["9C", "3C", "1C"]:
        Xk = P.subset(X, k)
        inp[k] = {
            "tr": (P.prep_input(Xk[m_tr], "fbtcnn", 3), y[m_tr]),
            "va": (P.prep_input(Xk[m_va], "fbtcnn", 3), y[m_va], s[m_va]),
            "te": (P.prep_input(Xk[m_te], "fbtcnn", 3), y[m_te], s[m_te]),
        }

    def build(cfg_key):
        C = len(P.CH_POS[cfg_key])
        return build_backbone("fbtcnn", C, n_samples, n_classes, n_bands=3)

    repeats = []
    for rep in range(1, N_REPEATS + 1):
        print(f"--- repeat {rep} ---", flush=True)
        np.random.seed(SEED); torch.manual_seed(SEED)

        Xtr, ytr = inp["9C"]["tr"]; Xva, yva, sva = inp["9C"]["va"]
        teacher = build("9C")
        teacher, _ = P.train_ce(teacher, Xtr, ytr, Xva, yva, sva, CONFIG, device)
        t_acc = P.eval_per_subject(teacher, *inp["9C"]["te"], device)

        row = {"teacher_9C": float(np.mean(list(t_acc.values())))}
        for c in ["3C", "1C"]:
            Xtr_c, ytr_c = inp[c]["tr"]; Xva_c, yva_c, sva_c = inp[c]["va"]
            base = build(c)
            base, _ = P.train_ce(base, Xtr_c, ytr_c, Xva_c, yva_c, sva_c, CONFIG, device)
            b_acc = P.eval_per_subject(base, *inp[c]["te"], device)
            row[f"{c}_baseline"] = float(np.mean(list(b_acc.values())))

            stud = build(c)
            stud, _ = P.train_kd(stud, teacher, Xtr_c, inp["9C"]["tr"][0], ytr_c,
                                  Xva_c, yva_c, sva_c, CONFIG, device, mode="relational")
            k_acc = P.eval_per_subject(stud, *inp[c]["te"], device)
            row[f"{c}_relational"] = float(np.mean(list(k_acc.values())))
            row[f"delta_{c}"] = (row[f"{c}_relational"] - row[f"{c}_baseline"]) * 100

        print(row, flush=True)
        repeats.append(row)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(repeats, f, indent=2)
    print(f"\nWrote {OUT}")

    print("\nIn-process spread (5 repeats, same CUDA context):")
    for key in ["teacher_9C", "3C_baseline", "3C_relational", "1C_baseline", "1C_relational"]:
        accs = np.array([r[key] for r in repeats]) * 100
        print(f"  {key:16s} " + " ".join(f"{a:.4f}" for a in accs) +
              f"   range={accs.max()-accs.min():.4f}pp  SD={accs.std(ddof=1):.4f}pp")
    for c in ["3C", "1C"]:
        deltas = np.array([r[f"delta_{c}"] for r in repeats])
        print(f"  delta_{c:16s} " + " ".join(f"{x:+.4f}" for x in deltas) +
              f"   range={deltas.max()-deltas.min():.4f}pp  SD={deltas.std(ddof=1):.4f}pp")


if __name__ == "__main__":
    main()

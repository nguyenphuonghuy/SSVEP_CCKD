"""Module 2 -- Cross-Channel Relational Knowledge Distillation (CCKD).

Runs the FULL experimental matrix described in the paper's Experimental Setup
in a single process (safe for PBS): teacher(9C) -> student{3C,1C} x
{baseline CE, response-based KD, response+relational KD, proposed CCKD
(relational only)}, averaged over multiple init seeds on ONE fixed
subject-independent split.

Usage:
    python training/train_cckd.py --dataset benchmark --data_root /path/to/Benchmark \
        --output_dir results/module2/cckd_benchmark_1s
    python training/train_cckd.py --dataset beta --data_root /path/to/BETA \
        --output_dir results/module2/cckd_beta_1s
    python training/train_cckd.py --dataset benchmark --backbone eegnet \
        --data_root /path/to/Benchmark --output_dir results/module2/eegnet_benchmark_1s

Smoke test:
    python training/train_cckd.py --dataset benchmark --data_root ... \
        --epochs 3 --seeds 2026 --output_dir results/module2/smoke
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
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


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["benchmark", "beta"])
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--config", default=None, help="YAML with dataset.teacher_channel_indices; "
                                                      "defaults to configs/<dataset>.yaml")
    ap.add_argument("--backbone", default="fbtcnn",
                     choices=["fbtcnn", "tcnn", "eegnet", "deepconvnet", "compactcnn", "shallowconvnet"])
    ap.add_argument("--n_bands", type=int, default=3)
    ap.add_argument("--window_s", type=float, default=1.0)
    ap.add_argument("--t_start", type=float, default=0.14)
    ap.add_argument("--split", type=int, nargs=3, default=None, metavar=("N_TRAIN", "N_VAL", "N_TEST"))
    ap.add_argument("--split_seed", type=int, default=2026)
    ap.add_argument("--seeds", type=int, nargs="+", default=[2026, 2027, 2028])
    ap.add_argument("--students", nargs="+", default=["3C", "1C"])
    ap.add_argument("--modes", nargs="+", default=["relational"],
                     choices=["logit", "relational", "hybrid", "fitnets", "attention"])
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
    split = tuple(a.split) if a.split else DEFAULT_SPLIT[a.dataset]

    CONFIG = {
        "epochs": a.epochs, "batch_size": a.batch_size, "lr": a.lr,
        "weight_decay": a.weight_decay, "tau": a.tau,
        "lambda_kd": a.lambda_kd, "lambda_sp": a.lambda_sp,
    }

    t0 = time.time()
    log("=" * 72)
    log("  CCKD-SSVEP -- Module 2 -- FULL RUN")
    log(f"  Device: {device}" + (f" | {torch.cuda.get_device_name(0)}" if device.type == "cuda" else ""))
    log(f"  Dataset={a.dataset} | Backbone={a.backbone} bands={a.n_bands} window={a.window_s}s "
        f"split={split} seeds={a.seeds} modes={a.modes}")
    log("=" * 72)

    log("  Loading dataset...")
    loader = PublicSSVEPDataset(a.data_root, a.dataset, sfreq=250.0, t_start=a.t_start,
                                 window_seconds=a.window_s, mat_key=mat_key, verbose=False)
    X64, y, s = loader.load_all_subjects(n_subjects)
    X = X64[:, ch_indices, :]  # -> 9-channel teacher array, fixed order
    n_samples, n_classes = X.shape[2], len(np.unique(y))
    log(f"  Data: X={X.shape} (9C selected) | n_samples={n_samples} "
        f"({n_samples / P.FS_PHYS:.2f}s @ {P.FS_PHYS}Hz) | classes={n_classes}")

    tr_s, va_s, te_s = P.fixed_split(s, *split, split_seed=a.split_seed)
    log(f"  Fixed split: train={list(tr_s)}\n              val={list(va_s)}\n              test={list(te_s)}\n")
    m_tr, m_va, m_te = P.masks_for(s, tr_s), P.masks_for(s, va_s), P.masks_for(s, te_s)

    log("  Preparing input (filter bank if fbtcnn)...")
    inp = {}
    for k in ["9C"] + a.students:
        Xk = P.subset(X, k)
        inp[k] = {
            "tr": (P.prep_input(Xk[m_tr], a.backbone, a.n_bands), y[m_tr]),
            "va": (P.prep_input(Xk[m_va], a.backbone, a.n_bands), y[m_va], s[m_va]),
            "te": (P.prep_input(Xk[m_te], a.backbone, a.n_bands), y[m_te], s[m_te]),
        }
    log(f"    9C train input: {inp['9C']['tr'][0].shape}\n")

    out_dir = Path(a.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {
        "config": {"backbone": a.backbone, "n_bands": a.n_bands, "window_s": a.window_s,
                   "t_start": a.t_start, "split": list(split), "split_seed": a.split_seed,
                   "seeds": a.seeds, "n_samples": int(n_samples), "tau": a.tau,
                   "lambda_kd": a.lambda_kd, "lambda_sp": a.lambda_sp, "epochs": a.epochs},
        "runs": {},
    }
    store_keys = [f"{c}_baseline" for c in a.students] + \
                 [f"{c}_{md}" for c in a.students for md in a.modes]
    store = {k: [] for k in store_keys}
    teacher_te = []
    teacher_params = None

    # --- Resume support: a seed is only checkpointed once it has fully finished
    # (teacher + every student/mode for that seed), so a checkpoint is always
    # safe to trust even if the process was killed (e.g. by the PBS hang-watchdog)
    # right after it was written.
    ckpt_path = out_dir / "checkpoint.json"

    def _load_ckpt():
        if not ckpt_path.exists():
            return {"fingerprint": None, "completed_seeds": [], "teacher_te": {},
                    "store": {k: {} for k in store_keys}, "val": {}, "teacher_params": None}
        with open(ckpt_path) as f:
            c = json.load(f)
        # JSON round-trips subject ids (dict keys) and seeds (dict keys) as strings.
        c["teacher_te"] = {int(s): {int(sub): acc for sub, acc in d.items()}
                            for s, d in c["teacher_te"].items()}
        c["store"] = {k: {int(s): {int(sub): acc for sub, acc in d.items()} for s, d in v.items()}
                      for k, v in c["store"].items()}
        c["val"] = {k: {int(s): v for s, v in d.items()} for k, d in c["val"].items()}
        return c

    def _save_ckpt(ckpt):
        tmp = ckpt_path.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump(ckpt, f)
        tmp.replace(ckpt_path)  # atomic on the same filesystem

    fingerprint = {
        "dataset": a.dataset, "data_root": str(a.data_root), "config_path": cfg_path,
        "teacher_channel_indices": [int(i) for i in ch_indices], "mat_key": mat_key,
        "backbone": a.backbone, "n_bands": a.n_bands, "window_s": a.window_s,
        "t_start": a.t_start, "split": list(split), "split_seed": a.split_seed,
        "n_samples": int(n_samples), "students": list(a.students), "modes": list(a.modes),
        "train": CONFIG,
    }
    ckpt = _load_ckpt()
    if ckpt["completed_seeds"] and ckpt["fingerprint"] != fingerprint:
        raise SystemExit(f"{ckpt_path} was written by a different configuration; refusing to resume. "
                         "Delete it or use a new --output_dir.")
    ckpt["fingerprint"] = fingerprint
    if ckpt["completed_seeds"]:
        log(f"  Resuming from checkpoint: {len(ckpt['completed_seeds'])}/{len(a.seeds)} "
            f"seed(s) already done ({ckpt['completed_seeds']})\n")

    def build(cfg_key):
        C = len(P.CH_POS[cfg_key])
        return build_backbone(a.backbone, C, n_samples, n_classes, n_bands=a.n_bands)

    for seed in a.seeds:
        if seed in ckpt["completed_seeds"]:
            log(f"----- SEED {seed} (from checkpoint) -----\n")
            continue
        log(f"----- SEED {seed} -----")
        np.random.seed(seed); torch.manual_seed(seed)

        Xtr, ytr = inp["9C"]["tr"]; Xva, yva, sva = inp["9C"]["va"]
        teacher = build("9C")
        teacher, tva = P.train_ce(teacher, Xtr, ytr, Xva, yva, sva, CONFIG, device)
        t_acc = P.eval_per_subject(teacher, *inp["9C"]["te"], device)
        teacher_params = count_params(teacher)
        log(f"  teacher-9C: val={tva:.4f} test={np.mean(list(t_acc.values())):.4f} "
            f"params={teacher_params}")

        seed_store = {}
        seed_val = {"teacher_9C": float(tva)}
        for c in a.students:
            Xtr_c, ytr_c = inp[c]["tr"]; Xva_c, yva_c, sva_c = inp[c]["va"]
            base = build(c)
            base, bva = P.train_ce(base, Xtr_c, ytr_c, Xva_c, yva_c, sva_c, CONFIG, device)
            b_acc = P.eval_per_subject(base, *inp[c]["te"], device)
            seed_store[f"{c}_baseline"] = b_acc
            seed_val[f"{c}_baseline"] = float(bva)
            log(f"  {c} baseline: test={np.mean(list(b_acc.values())):.4f}")
            for md in a.modes:
                stud = build(c)
                stud, kva = P.train_kd(stud, teacher, Xtr_c, inp["9C"]["tr"][0], ytr_c,
                                        Xva_c, yva_c, sva_c, CONFIG, device, mode=md)
                k_acc = P.eval_per_subject(stud, *inp[c]["te"], device)
                seed_store[f"{c}_{md}"] = k_acc
                seed_val[f"{c}_{md}"] = float(kva)
                log(f"  {c} {md}: test={np.mean(list(k_acc.values())):.4f}")
        log("")

        # Seed fully finished -- checkpoint it now so a kill/hang after this point
        # never has to redo it.
        ckpt["completed_seeds"].append(seed)
        ckpt["teacher_te"][seed] = t_acc
        ckpt["teacher_params"] = teacher_params
        for k, v in seed_store.items():
            ckpt["store"][k][seed] = v
        for k, v in seed_val.items():
            ckpt["val"].setdefault(k, {})[seed] = v
        _save_ckpt(ckpt)

    teacher_te = [ckpt["teacher_te"][s] for s in a.seeds]
    teacher_params = ckpt["teacher_params"]
    for k in store_keys:
        store[k] = [ckpt["store"][k][s] for s in a.seeds]

    log("=" * 72); log("  RESULTS (test set; +/- = SD across subjects after seed averaging)"); log("=" * 72)
    results["runs"]["teacher_9C"] = P.summarize(teacher_te, a.window_s)
    results["runs"]["teacher_9C"]["params"] = teacher_params
    log(f"  teacher-9C: {results['runs']['teacher_9C']['mean_acc']*100:.2f}% "
        f"(ITR={results['runs']['teacher_9C']['itr']:.1f})")
    for c in a.students:
        base_sum = P.summarize(store[f"{c}_baseline"], a.window_s)
        results["runs"][f"{c}_baseline"] = base_sum
        log(f"\n  {c} baseline: {base_sum['mean_acc']*100:.2f}% +/- {base_sum['std_acc']*100:.2f} SD(subj) "
            f"(ITR={base_sum['itr']:.1f})")
        for md in a.modes:
            kd_sum = P.summarize(store[f"{c}_{md}"], a.window_s)
            results["runs"][f"{c}_{md}"] = kd_sum
            gain = (kd_sum["mean_acc"] - base_sum["mean_acc"]) * 100
            st = P.paired_stats(store[f"{c}_baseline"], store[f"{c}_{md}"])
            results["runs"][f"{c}_{md}"]["paired_vs_baseline"] = st
            sig = "***" if st["wilcoxon_p"] < 0.05 else ""
            log(f"  {c} {md}: {kd_sum['mean_acc']*100:.2f}% +/- {kd_sum['std_acc']*100:.2f} SD(subj) "
                f"(ITR={kd_sum['itr']:.1f}) | gain {gain:+.2f}pp")
            log(f"        paired: delta={st['mean_delta_pp']:+.2f}pp "
                f"| improved {st['n_improved']}/{st['n_subjects']} subj "
                f"| Wilcoxon p={st['wilcoxon_p']:.4f} {sig}")

    results["val_best_per_seed"] = {k: [ckpt["val"][k][s] for s in a.seeds]
                                    for k in ["teacher_9C"] + store_keys}
    results["config"]["model_selection"] = "best validation accuracy per run; test set not used"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = out_dir / f"cckd_{a.backbone}_{a.dataset}_{a.window_s}s_{ts}.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    log(f"\n  Results: {out}")
    log(f"  Total time: {(time.time()-t0)/60:.1f} min")
    ckpt_path.unlink(missing_ok=True)  # final result written successfully; checkpoint no longer needed


if __name__ == "__main__":
    main()

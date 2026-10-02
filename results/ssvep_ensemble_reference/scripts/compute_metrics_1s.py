"""
compute_metrics_1s.py — Tinh acc + std + ITR + kiem dinh thong ke cho CUA SO 1.0s.

Tu dong do file collaborative moi nhat (co timestamp) trong results/ensemble_1s/.
ITR tinh voi T = 1.0 + 0.5 (gaze) = 1.5s.

Chay:  python compute_metrics_1s.py
"""
import json
import math
import glob
import numpy as np
from pathlib import Path
from scipy import stats

N_CLASSES = 40
GAZE = 0.5
WINDOW = 1.0
T = WINDOW + GAZE   # 1.5s cho ITR @1s

BASE_DIR = 'results/baselines_1s'
ENS_DIR = 'results/ensemble_1s'


def itr(N, P, T):
    if P >= 1.0:
        bits = math.log2(N)
    elif P <= 1.0 / N:
        bits = 0.0
    else:
        bits = (math.log2(N) + P * math.log2(P)
                + (1 - P) * math.log2((1 - P) / (N - 1)))
    return (60.0 / T) * bits


def load_folds(path, key):
    """Doc accuracy per-subject tu file json."""
    with open(path) as f:
        data = json.load(f)
    folds = data['folds']
    if key is None:
        return np.array([float(x) for x in folds])
    return np.array([fd[key] for fd in folds])


def find_latest(pattern):
    """Tim file moi nhat khop pattern (theo timestamp trong ten)."""
    files = sorted(glob.glob(pattern))
    return files[-1] if files else None


print("=" * 60)
print(f"  METRICS CUA SO 1.0s  (T = {T}s cho ITR)")
print("=" * 60)

# ── 1. Single best (tim extractor tot nhat trong 3 baseline) ──
print("\n[Baseline extractors]")
single_accs = {}
for ext in ['temporal_cnn', 'freq_domain_cnn', 'spatial_temporal_cnn']:
    p = f'{BASE_DIR}/{ext}_loso_results.json'
    if Path(p).exists():
        a = load_folds(p, 'best_acc')
        single_accs[ext] = a
        print(f"  {ext:22s}: {a.mean()*100:6.2f}% +/- {a.std()*100:5.2f}%")
    else:
        print(f"  {ext:22s}: [!] khong thay {p}")

if single_accs:
    best_ext = max(single_accs, key=lambda k: single_accs[k].mean())
    single_best = single_accs[best_ext]
    print(f"  -> Single best: {best_ext} ({single_best.mean()*100:.2f}%)")
else:
    single_best = None

# ── 2. Naive ensemble ──
print("\n[Naive ensemble]")
naive_file = find_latest(f'{ENS_DIR}/naive_ensemble_loso_*.json')
if naive_file:
    naive = load_folds(naive_file, 'ensemble_acc')
    print(f"  File: {Path(naive_file).name}")
    print(f"  Acc : {naive.mean()*100:6.2f}% +/- {naive.std()*100:5.2f}% | ITR={itr(N_CLASSES, naive.mean(), T):.2f}")
else:
    naive = None
    print(f"  [!] Khong thay file naive trong {ENS_DIR}")

# ── 3. Collaborative ──
print("\n[Collaborative ensemble]")
collab_file = find_latest(f'{ENS_DIR}/collaborative_loso_*.json')
if collab_file:
    collab = load_folds(collab_file, 'best_acc')
    print(f"  File: {Path(collab_file).name}")
    print(f"  Acc : {collab.mean()*100:6.2f}% +/- {collab.std()*100:5.2f}% | ITR={itr(N_CLASSES, collab.mean(), T):.2f}")
else:
    collab = None
    print(f"  [!] Khong thay file collaborative trong {ENS_DIR}")

# ── 4. Bang tong hop ITR ──
if single_best is not None and naive is not None and collab is not None:
    print("\n" + "=" * 60)
    print("  BANG TONG HOP (cua so 1.0s)")
    print("=" * 60)
    print(f"  {'Phuong phap':24s} {'Acc (%)':>14s} {'ITR':>8s}")
    print(f"  {'Single best':24s} {single_best.mean()*100:>7.2f}+/-{single_best.std()*100:<5.2f} {itr(N_CLASSES, single_best.mean(), T):>8.2f}")
    print(f"  {'Naive ensemble':24s} {naive.mean()*100:>7.2f}+/-{naive.std()*100:<5.2f} {itr(N_CLASSES, naive.mean(), T):>8.2f}")
    print(f"  {'Collaborative':24s} {collab.mean()*100:>7.2f}+/-{collab.std()*100:<5.2f} {itr(N_CLASSES, collab.mean(), T):>8.2f}")

    # ── 5. Kiem dinh thong ke ──
    if len(collab) == len(naive):
        print("\n  --- Kiem dinh Collaborative vs Naive ---")
        t_stat, t_p = stats.ttest_rel(collab, naive)
        print(f"    Chenh lech    : {(collab.mean()-naive.mean())*100:+.2f}%")
        print(f"    Paired t-test : t={t_stat:.3f}, p={t_p:.4e}")
        try:
            w_stat, w_p = stats.wilcoxon(collab, naive)
            print(f"    Wilcoxon      : W={w_stat:.1f}, p={w_p:.4e}")
        except ValueError as e:
            print(f"    Wilcoxon: {e}")
        print(f"    So subject cai thien: {(collab>naive).sum()}/{len(collab)}")

    # ── 6. So sanh voi SSVEPformer ──
    print("\n" + "=" * 60)
    print("  SO SANH VOI SSVEPformer (cung cross-subject, cung 1.0s)")
    print("=" * 60)
    print(f"    SSVEPformer      : 80.40%")
    print(f"    FB-SSVEPformer   : 83.19%")
    print(f"    Collaborative (ban): {collab.mean()*100:.2f}%")
    diff = collab.mean()*100 - 80.40
    print(f"    => Chenh lech vs SSVEPformer: {diff:+.2f}%")

print("\n" + "=" * 60)

"""
compute_metrics_4s.py — Tinh acc + std + ITR + kiem dinh cho CUA SO 4.0s.

Tu dong do file collaborative/naive moi nhat trong results/*_4s_real/.
ITR tinh voi T = 4.0 + 0.5 = 4.5s.

Chay:  python compute_metrics_4s.py
"""
import json
import math
import glob
import numpy as np
from pathlib import Path
from scipy import stats

N_CLASSES = 40
GAZE = 0.5
WINDOW = 4.0
T = WINDOW + GAZE   # 4.5s

BASE_DIR = 'results/baselines_4s_real'
ENS_DIR = 'results/ensemble_4s_real'


def itr(N, P, T):
    if P >= 1.0:
        b = math.log2(N)
    elif P <= 1.0 / N:
        b = 0.0
    else:
        b = math.log2(N) + P*math.log2(P) + (1-P)*math.log2((1-P)/(N-1))
    return (60.0/T) * b


def load_folds(path, key):
    d = json.load(open(path))
    folds = d['folds']
    if key is None:
        return np.array([float(x) for x in folds])
    return np.array([f[key] for f in folds])


def find_latest(pattern):
    files = sorted(glob.glob(pattern))
    return files[-1] if files else None


print("=" * 60)
print(f"  METRICS CUA SO 4.0s  (T = {T}s cho ITR)")
print("=" * 60)

# ── Single extractors ──
print("\n[Baseline extractors]")
single = {}
for ext in ['temporal_cnn', 'freq_domain_cnn', 'spatial_temporal_cnn']:
    p = f'{BASE_DIR}/{ext}_loso_results.json'
    if Path(p).exists():
        a = load_folds(p, 'best_acc')
        single[ext] = a
        print(f"  {ext:22s}: {a.mean()*100:6.2f}% +/- {a.std()*100:5.2f}%")

if single:
    best_ext = max(single, key=lambda k: single[k].mean())
    sb = single[best_ext]
    print(f"  -> Single best: {best_ext} ({sb.mean()*100:.2f}%)")

# ── Naive ──
print("\n[Naive ensemble]")
nf = find_latest(f'{ENS_DIR}/naive_ensemble_loso_*.json')
naive = load_folds(nf, 'ensemble_acc') if nf else None
if naive is not None:
    print(f"  File: {Path(nf).name}")
    print(f"  Acc : {naive.mean()*100:.2f}% +/- {naive.std()*100:.2f}% | ITR={itr(N_CLASSES,naive.mean(),T):.2f}")

# ── Collaborative ──
print("\n[Collaborative]")
cf = find_latest(f'{ENS_DIR}/collaborative_loso_*.json')
collab = load_folds(cf, 'best_acc') if cf else None
if collab is not None:
    print(f"  File: {Path(cf).name}")
    print(f"  Acc : {collab.mean()*100:.2f}% +/- {collab.std()*100:.2f}% | ITR={itr(N_CLASSES,collab.mean(),T):.2f}")

# ── Bang tong hop + kiem dinh ──
if single and naive is not None and collab is not None:
    print("\n" + "=" * 60)
    print("  BANG TONG HOP (cua so 4.0s)")
    print("=" * 60)
    print(f"  {'Phuong phap':22s} {'Acc (%)':>15s} {'ITR':>8s}")
    print(f"  {'Single best':22s} {sb.mean()*100:>8.2f}+/-{sb.std()*100:<5.2f} {itr(N_CLASSES,sb.mean(),T):>8.2f}")
    print(f"  {'Naive ensemble':22s} {naive.mean()*100:>8.2f}+/-{naive.std()*100:<5.2f} {itr(N_CLASSES,naive.mean(),T):>8.2f}")
    print(f"  {'Collaborative':22s} {collab.mean()*100:>8.2f}+/-{collab.std()*100:<5.2f} {itr(N_CLASSES,collab.mean(),T):>8.2f}")

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

    print("\n" + "=" * 60)
    print("  SO SANH 4.0s vs 5.44s")
    print("=" * 60)
    print(f"    Collab 4.0s : {collab.mean()*100:.2f}%")
    print(f"    Collab 5.44s: 93.27% (da co)")
    print(f"    Chenh lech  : {collab.mean()*100 - 93.27:+.2f}%")
    print(f"\n    So SSVEPformer @1s (cross-subject): 80.40%")
    print(f"    Collab 4.0s ({collab.mean()*100:.2f}%) vuot SSVEPformer: {collab.mean()*100 - 80.40:+.2f}%")

print("\n" + "=" * 60)

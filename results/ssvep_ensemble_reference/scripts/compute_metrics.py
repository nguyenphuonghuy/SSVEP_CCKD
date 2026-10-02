"""
evaluation/compute_metrics.py
==============================
Tính std, ITR, và kiểm định thống kê từ các file kết quả per-subject ĐÃ XÁC ĐỊNH.
Đã map đúng đường dẫn và key cho từng file (mỗi loại file có format khác nhau).

Cách dùng:
  python evaluation/compute_metrics.py
"""

import json
import math
import numpy as np
from pathlib import Path
from scipy import stats

N_CLASSES = 40
GAZE_SHIFT = 0.5

# Mỗi entry: (đường dẫn, key_trong_folds). key=None → folds là list số trực tiếp
FILES = {
    'benchmark': {
        'window': 4.0,
        'single_best': ('results/baselines_4s/temporal_cnn_loso_results.json', 'best_acc'),
        'naive':       ('results/ensemble_4s/naive_ensemble_loso_20260608_0645.json', 'ensemble_acc'),
        'collab':      ('results/ensemble_4s_3ext/collaborative_loso_20260608_1303.json', 'best_acc'),
    },
    'beta': {
        'window': 2.0,
        'single_best': ('results/baselines_beta/spatial_temporal_cnn_beta_loso_results.json', 'best_acc'),
        'naive':       ('results/ensemble_beta/naive_ensemble_beta.json', None),
        'collab':      ('results/ensemble_beta/collaborative_loso_20260609_2227.json', 'best_acc'),
    },
}


def itr(N, P, T):
    if P >= 1.0:
        bits = math.log2(N)
    elif P <= 1.0 / N:
        bits = 0.0
    else:
        bits = (math.log2(N) + P * math.log2(P)
                + (1 - P) * math.log2((1 - P) / (N - 1)))
    return (60.0 / T) * bits


def load_acc(path, key_in_folds):
    with open(path) as f:
        data = json.load(f)
    folds = data['folds']
    if key_in_folds is None:
        return np.array([float(x) for x in folds])
    return np.array([f[key_in_folds] for f in folds])


def analyze(ds_name):
    cfg = FILES[ds_name]
    window = cfg['window']
    T = window + GAZE_SHIFT
    print("\n" + "=" * 64)
    print(f"  {ds_name.upper()}  (cua so {window}s, T={T}s)")
    print("=" * 64)

    accs = {}
    for method in ['single_best', 'naive', 'collab']:
        path, key = cfg[method]
        if not Path(path).exists():
            print(f"  [!] KHONG TIM THAY: {path}")
            continue
        a = load_acc(path, key)
        accs[method] = a
        mean, std = a.mean(), a.std()
        print(f"\n  {method:12s}: {mean*100:6.2f}% +/- {std*100:5.2f}%  "
              f"| ITR={itr(N_CLASSES, mean, T):6.2f} bits/min  | n={len(a)}")

    if 'collab' in accs and 'naive' in accs:
        c, n = accs['collab'], accs['naive']
        if len(c) == len(n):
            print("\n  --- Kiem dinh thong ke: Collaborative vs Naive ---")
            print(f"    Chenh lech trung binh : {(c.mean()-n.mean())*100:+.2f}%")
            t_stat, t_p = stats.ttest_rel(c, n)
            print(f"    Paired t-test         : t={t_stat:.3f}, p={t_p:.4e}")
            try:
                w_stat, w_p = stats.wilcoxon(c, n)
                print(f"    Wilcoxon signed-rank  : W={w_stat:.1f}, p={w_p:.4e}")
            except ValueError as e:
                print(f"    Wilcoxon: {e}")
            print(f"    So subject cai thien  : {(c>n).sum()}/{len(c)}")
            sig = "CO Y NGHIA (p<0.05)" if t_p < 0.05 else "khong co y nghia"
            print(f"    -> Ket luan           : {sig}")
        else:
            print(f"\n  [!] So subject lech: collab={len(c)}, naive={len(n)}")


if __name__ == "__main__":
    for ds in ['benchmark', 'beta']:
        analyze(ds)
    print("\n" + "=" * 64 + "\n")

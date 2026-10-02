"""
run_beta_full.py — Chay TOAN BO quy trinh BETA ablation trong 1 lan (cho PBS).

Gop 2 buoc:
  Buoc 1: Train shallow_conv_net tren BETA (neu chua co)
  Buoc 2: Ablation A tren BETA (bo lan luot tung nhanh)

An toan cho batch/PBS: khong co input(), khong cho tuong tac.

Chay:  python run_beta_full.py
"""
import sys
import json
from pathlib import Path
from datetime import datetime
import numpy as np
import torch

sys.path.insert(0, str(Path.cwd()))

from training.train_extractors_beta import train_loso_beta
from evaluation.ablation_study import ablation_extractor_subset, ALL_EXTRACTORS
from data.beta_loader import BETADataset

# ── CAU HINH ──
DATA_PATH = "data/raw/beta"
N_SUBJECTS = 70
T_START = 0.14
T_END = 2.14                       # 610 mau = 2.44s (khop baseline BETA goc)
BASE_BETA = "results/baselines_beta"
ABLATION_OUT = "results/ablation_beta"
EPOCHS = 100
BATCH_SIZE = 64

CONFIG = {'epochs': EPOCHS, 'batch_size': 16, 'lr': 0.001, 'weight_decay': 1e-4}


def step1_train_shallow(device, X, y, subject_ids):
    """Train shallow_conv_net neu chua co checkpoint."""
    print("\n" + "=" * 64)
    print("  BUOC 1: TRAIN shallow_conv_net TREN BETA")
    print("=" * 64)

    shallow_ckpt = Path(BASE_BETA) / "shallow_conv_net_fold1.pt"
    if shallow_ckpt.exists():
        print(f"  [Bo qua] Da co {shallow_ckpt}")
        return

    save_dir = Path(BASE_BETA)
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"  Train shallow_conv_net (LOSO {N_SUBJECTS} folds, {EPOCHS} epochs)...")
    result = train_loso_beta(
        'shallow_conv_net', X, y, subject_ids,
        CONFIG, device, save_dir, n_folds=None
    )
    print(f"  [OK] Shallow BETA: {result['mean_acc']*100:.2f}% +/- {result['std_acc']*100:.2f}%")


def step2_ablation(device, X, y, subject_ids, n_ch, n_samp, n_cls):
    """Chay Ablation A tren BETA."""
    print("\n" + "=" * 64)
    print("  BUOC 2: ABLATION A TREN BETA")
    print("=" * 64)

    # Kiem tra du 4 nhanh
    missing = [e for e in ALL_EXTRACTORS
               if not (Path(BASE_BETA) / f"{e}_fold1.pt").exists()]
    if missing:
        print(f"  [!] THIEU checkpoint: {missing}. Dung lai.")
        sys.exit(1)
    print(f"  Du 4 nhanh checkpoint. Bat dau ablation...")

    save_dir = Path(ABLATION_OUT)
    save_dir.mkdir(parents=True, exist_ok=True)

    res_a = ablation_extractor_subset(
        X, y, subject_ids,
        ckpt_dir=BASE_BETA,
        device=device, n_ch=n_ch, n_samp=n_samp, n_cls=n_cls,
        batch_size=BATCH_SIZE
    )

    ts = datetime.now().strftime('%Y%m%d_%H%M')
    out = save_dir / f"ablation_beta_A_{ts}.json"
    with open(out, 'w') as f:
        json.dump({'ablation_A_extractor_subset': res_a}, f, indent=2)

    # In ket qua + ket luan M2
    print("\n" + "=" * 64)
    print("  KET QUA ABLATION A TREN BETA")
    print("=" * 64)
    labels = {
        'all_4': 'Giu du 4 nhanh', 'drop_ext1': 'Bo nhanh thoi gian',
        'drop_ext2': 'Bo nhanh tan so', 'drop_ext3': 'BO SHALLOW (Ext3)',
        'drop_ext4': 'Bo nhanh khong gian',
        'best_single_ext1': 'Chi temporal', 'best_single_ext2': 'Chi freq',
    }
    all_4_acc, drop3_acc = None, None
    for cfg_name, val in res_a.items():
        acc = val.get('mean_acc', 0) if isinstance(val, dict) else val
        mark = ""
        if cfg_name == 'all_4':
            all_4_acc = acc
        if cfg_name == 'drop_ext3':
            drop3_acc = acc
            mark = "  <== BO SHALLOW"
        print(f"    {labels.get(cfg_name, cfg_name):24s}: {acc*100:6.2f}%{mark}")

    print("\n" + "-" * 64)
    if all_4_acc is not None and drop3_acc is not None:
        diff = (drop3_acc - all_4_acc) * 100
        print(f"  Bo shallow ({drop3_acc*100:.2f}%) vs giu 4 nhanh ({all_4_acc*100:.2f}%): {diff:+.2f}%")
        if diff > 0:
            print(f"  => BO SHALLOW CAI THIEN tren BETA (giong Benchmark) => xac nhan M2!")
        else:
            print(f"  => Bo shallow KHONG cai thien tren BETA => can dien giai than trong.")
    print(f"  Ket qua: {out}")


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 64)
    print("  BETA ABLATION FULL (train shallow + ablation A)")
    print("=" * 64)
    print(f"  Device: {device}")
    if device.type == 'cuda':
        print(f"  GPU   : {torch.cuda.get_device_name(0)}")
    print(f"  t_end : {T_END} (610 mau = 2.44s)")

    # Load BETA 1 lan, dung cho ca 2 buoc
    print("\n  Loading BETA dataset ...")
    loader = BETADataset(data_path=DATA_PATH, t_start=T_START, t_end=T_END, verbose=False)
    X, y, subject_ids = loader.load_all_subjects(n_subjects=N_SUBJECTS)
    n_ch, n_samp, n_cls = X.shape[1], X.shape[2], len(np.unique(y))
    print(f"  BETA: X={X.shape}, classes={n_cls}, n_samples={n_samp}")
    if n_samp == 610:
        print(f"  [OK] n_samples=610 (2.44s), khop baseline BETA goc.")
    else:
        print(f"  [!] CANH BAO: n_samples={n_samp} != 610. Van tiep tuc, kiem tra ket qua sau.")

    step1_train_shallow(device, X, y, subject_ids)
    step2_ablation(device, X, y, subject_ids, n_ch, n_samp, n_cls)

    print("\n" + "=" * 64)
    print("  HOAN TAT TOAN BO BETA ABLATION")
    print("=" * 64)


if __name__ == "__main__":
    main()

"""
run_1s.py — Chay pipeline o CUA SO 1.0 GIAY VAT LY that (Benchmark).

Muc dich: co con so so sanh CONG BANG voi SSVEPformer (cross-subject @1s = 80.4%).

ĐA XAC NHAN sfreq = 250 Hz:
    - 1.0s vat ly = 250 mau  =>  t_start=0.14, t_end=0.39
    - (2.0s da co san = ket qua dang goi nham "0.5s": t_end=0.64)
    - (5.44s da co san = ket qua dang goi nham "4s": t_end=4.14)

Script chay 3 buoc tuan tu: baseline -> naive ensemble -> collaborative.

Chay:  python run_1s.py
       python run_1s.py --skip-baseline     (neu da train baseline roi)
"""
import subprocess
import sys
import argparse
from pathlib import Path

# ── THAM SO CUA SO 1.0s VAT LY ──
T_START = "0.14"
T_END = "0.39"          # 250 mau @ 250Hz = 1.0s vat ly

# ── THU MUC KET QUA ──
BASE_DIR = "results/baselines_1s"
ENS_DIR = "results/ensemble_1s"

# ── Python interpreter dang chay (dam bao dung venv) ──
PY = sys.executable


def run_step(name, cmd):
    """Chay 1 buoc, in ro rang, dung neu loi."""
    print("\n" + "=" * 64)
    print(f"  {name}")
    print("=" * 64)
    print("  Lenh:", " ".join(cmd))
    print("-" * 64)
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"\n[!] LOI o buoc: {name} (return code {result.returncode})")
        print("    Dung lai. Kiem tra loi o tren.")
        sys.exit(result.returncode)
    print(f"\n  [OK] Xong: {name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-baseline", action="store_true",
                    help="Bo qua buoc train baseline (neu da co results/baselines_1s)")
    ap.add_argument("--skip-naive", action="store_true",
                    help="Bo qua buoc naive ensemble")
    args = ap.parse_args()

    print("=" * 64)
    print("  CHAY CUA SO 1.0s VAT LY")
    print(f"  t_start={T_START}, t_end={T_END} (250 mau @ 250Hz)")
    print(f"  Python: {PY}")
    print(f"  Baseline dir: {BASE_DIR}")
    print(f"  Ensemble dir: {ENS_DIR}")
    print("=" * 64)

    # ── BUOC 1: Train 3 baseline extractors ──
    if not args.skip_baseline:
        run_step(
            "[1/3] Train 3 baseline extractors (1s)",
            [PY, "training/train_extractors.py",
             "--extractor", "all",
             "--data_path", "data/raw/benchmark",
             "--n_subjects", "35",
             "--epochs", "100",
             "--t_start", T_START,
             "--t_end", T_END,
             "--save_dir", BASE_DIR]
        )
    else:
        print("\n[Bo qua] Buoc 1 (baseline) theo yeu cau --skip-baseline")

    # ── BUOC 2: Naive ensemble ──
    if not args.skip_naive:
        run_step(
            "[2/3] Naive ensemble (1s)",
            [PY, "evaluation/naive_ensemble.py",
             "--data_path", "data/raw/benchmark",
             "--checkpoint_dir", BASE_DIR,
             "--n_subjects", "35",
             "--t_start", T_START,
             "--t_end", T_END,
             "--save_dir", ENS_DIR]
        )
    else:
        print("\n[Bo qua] Buoc 2 (naive) theo yeu cau --skip-naive")

    # ── BUOC 3: Collaborative distillation ──
    run_step(
        "[3/3] Collaborative distillation (1s)",
        [PY, "training/collaborative_ensemble.py",
         "--data_path", "data/raw/benchmark",
         "--init_checkpoint_dir", BASE_DIR,
         "--n_subjects", "35",
         "--epochs", "50",
         "--t_start", T_START,
         "--t_end", T_END,
         "--save_dir", ENS_DIR]
    )

    # ── Tong ket ──
    print("\n" + "=" * 64)
    print("  HOAN TAT CUA SO 1.0s")
    print("=" * 64)
    print(f"  Ket qua trong:")
    print(f"    - {BASE_DIR}/  (3 baseline)")
    print(f"    - {ENS_DIR}/   (naive + collaborative)")
    print()
    print("  BUOC TIEP: python compute_metrics_1s.py")
    print("  (ITR @1s dung T = 1.0 + 0.5 gaze = 1.5s)")
    print("=" * 64)


if __name__ == "__main__":
    main()

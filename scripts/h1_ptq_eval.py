"""H1 (mo-hinh-hoa/embedded_h1h4_20261101.md Muc 4-5, G1): static INT8 PTQ on the
90 FP32 ONNX models exported by export_onnx.py, re-evaluated on the SAME held-out
test set/subjects used for the original FP32 result (test set evaluated once per
the paper's protocol; PTQ is a fixed post-hoc transform of an already-selected
model, not a new tuning loop, so re-scoring it on the existing test labels does
not reopen model selection).

Calibration data: validation-split trials (NOT test, NOT train) for the matching
dataset/backbone/montage/split_seed -- the same 8 BETA subjects held out as
validation during training, re-subset/re-prepped identically to train_cckd.py.
Reusing validation data for calibration (not for choosing a model) is standard
PTQ practice and keeps the test set untouched until final scoring.

Quantization: onnxruntime.quantization.quantize_static, QDQ format, per-tensor,
weight=QInt8, activation=QUInt8, MinMax calibration (default) -- static because
mo-hinh-hoa explicitly specifies "PTQ tinh" (static), not QAT.

Output: results/module3/h1_ptq/analysis_vs_prediction.csv, one row per
(cell, split_seed): delta_fp32_pp, delta_int8_pp, diff_pp, sign_match, plus a
per-cell summary (mean across the 5 splits) and the G1 GO/NO-GO verdict.
Must run under PBS (no torch/dataset import on the login node).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import onnxruntime as ort
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType, QuantFormat

from datasets.public_ssvep import PublicSSVEPDataset
from utils.config import load_yaml, deep_get
import training.cckd_pipeline as P

DATASET = "beta"
SPLIT = (42, 8, 20)
N_SUBJECTS = 70
SPLITS = [2026, 3026, 4026, 5026, 6026]
SEEDS_8 = [2026, 2027, 2028, 2029, 2030, 2031, 2032, 2033]
SEEDS_3 = [2026, 2027, 2028]
WINDOW_S = 1.0
T_START = 0.14
DATA_ROOT = "/gpfs0/users/gv_nphuy/projects/ssvep_ensemble/data/raw/beta"

# (backbone, student, cell_dir, is_significant_fp32_cell_for_G1, seeds)
# G1 cells need the full original 8-seed protocol (the 3-seed subset flipped the
# cell-level Delta_FP32 sign for eegnet/3C vs the original RQ1 finding). deepconvnet/3C
# is H2-prep only (bit-width sweep backbone selection), not scored for G1, stays at 3 seeds.
CELLS = [
    ("eegnet", "3C", "beta_eegnet_3C", True, SEEDS_8),
    ("eegnet", "1C", "beta_eegnet_1C", True, SEEDS_8),
    ("deepconvnet", "3C", "beta_deepconvnet_3C", False, SEEDS_3),  # H2 prep only, not a G1 cell
]

RETRAIN_ROOT = ROOT / "results" / "module3" / "h1_retrain"
OUT_DIR = ROOT / "results" / "module3" / "h1_ptq"


class NumpyCalibReader(CalibrationDataReader):
    def __init__(self, X: np.ndarray, input_name: str, batch_size: int = 32):
        self.X = X
        self.input_name = input_name
        self.batch_size = batch_size
        self.idx = 0

    def get_next(self):
        if self.idx >= len(self.X):
            return None
        batch = self.X[self.idx:self.idx + self.batch_size]
        self.idx += self.batch_size
        return {self.input_name: batch}


def eval_onnx_per_subject(onnx_path: Path, X: np.ndarray, y: np.ndarray, s: np.ndarray, batch: int = 512):
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    preds = []
    for i in range(0, len(X), batch):
        out = sess.run(None, {input_name: X[i:i + batch]})[0]
        preds.append(out.argmax(1))
    pred = np.concatenate(preds)
    return {int(sub): float((pred[s == sub] == y[s == sub]).mean()) for sub in np.unique(s)}


def quantize_one(fp32_path: Path, int8_path: Path, calib_X: np.ndarray, input_name: str):
    int8_path.parent.mkdir(parents=True, exist_ok=True)
    reader = NumpyCalibReader(calib_X, input_name)
    quantize_static(
        str(fp32_path), str(int8_path), reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        per_channel=False,
    )


def main():
    cfg = load_yaml(str(ROOT / "configs" / f"{DATASET}.yaml"))
    ch_indices = deep_get(cfg, "dataset.teacher_channel_indices")
    mat_key = deep_get(cfg, "dataset.mat_key", "data")

    print("Loading BETA dataset once (shared across the 3 cells)...", flush=True)
    loader = PublicSSVEPDataset(DATA_ROOT, DATASET, sfreq=250.0, t_start=T_START,
                                 window_seconds=WINDOW_S, mat_key=mat_key, verbose=False)
    X64, y, s = loader.load_all_subjects(N_SUBJECTS)
    X9 = X64[:, ch_indices, :]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []

    for backbone, student, cell_dir, is_g1, seeds in CELLS:
        for split_seed in SPLITS:
            split_root = RETRAIN_ROOT / f"{cell_dir}_s{split_seed}"
            onnx_dir = split_root / "onnx"
            int8_dir = split_root / "onnx_int8"
            if not onnx_dir.exists():
                print(f"  SKIP {cell_dir} s{split_seed}: no onnx/ dir")
                continue

            tr_s, va_s, te_s = P.fixed_split(s, *SPLIT, split_seed=split_seed)
            m_va, m_te = P.masks_for(s, va_s), P.masks_for(s, te_s)
            Xc = P.subset(X9, student)
            Xva_c = P.prep_input(Xc[m_va], backbone, n_bands=3)
            Xte_c, yte_c, ste_c = P.prep_input(Xc[m_te], backbone, n_bands=3), y[m_te], s[m_te]
            print(f"  {cell_dir} s{split_seed}: calib n={len(Xva_c)} test n={len(Xte_c)}", flush=True)

            fp32_acc = {"baseline": [], "cckd": []}
            int8_acc = {"baseline": [], "cckd": []}
            for seed in seeds:
                for tag in ["baseline", "cckd"]:
                    stem_glob = f"{DATASET}_{backbone}_{student}_{tag}*split{split_seed}_seed{seed}"
                    matches = list(onnx_dir.glob(stem_glob + ".onnx"))
                    if not matches:
                        raise SystemExit(f"no onnx match for {stem_glob} in {onnx_dir}")
                    fp32_path = matches[0]
                    int8_path = int8_dir / (fp32_path.stem + "_int8.onnx")

                    sess0 = ort.InferenceSession(str(fp32_path), providers=["CPUExecutionProvider"])
                    input_name = sess0.get_inputs()[0].name
                    if not int8_path.exists():
                        quantize_one(fp32_path, int8_path, Xva_c, input_name)

                    fp32_acc[tag].append(eval_onnx_per_subject(fp32_path, Xte_c, yte_c, ste_c))
                    int8_acc[tag].append(eval_onnx_per_subject(int8_path, Xte_c, yte_c, ste_c))

            st_fp32 = P.paired_stats(fp32_acc["baseline"], fp32_acc["cckd"])
            st_int8 = P.paired_stats(int8_acc["baseline"], int8_acc["cckd"])
            d_fp32, d_int8 = st_fp32["mean_delta_pp"], st_int8["mean_delta_pp"]
            rows.append({
                "cell": cell_dir, "backbone": backbone, "student": student,
                "split_seed": split_seed, "is_g1_cell": is_g1,
                "delta_fp32_pp": d_fp32, "delta_int8_pp": d_int8,
                "diff_pp": d_int8 - d_fp32,
                "sign_match": (d_fp32 >= 0) == (d_int8 >= 0),
                "p_fp32": st_fp32["wilcoxon_p"], "p_int8": st_int8["wilcoxon_p"],
            })
            print(f"    Delta_FP32={d_fp32:+.2f}pp Delta_INT8={d_int8:+.2f}pp "
                  f"diff={d_int8-d_fp32:+.2f}pp sign_match={rows[-1]['sign_match']}", flush=True)

    import csv
    out_csv = OUT_DIR / "analysis_vs_prediction.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {out_csv}")

    print("\n=== Per-cell summary (mean +/- sd across 5 splits) ===")
    print("=== G1 verdict uses only is_g1_cell=True cells (significant after Holm in the original FP32 RQ1) ===")
    g1_cells = sorted({r["cell"] for r in rows if r["is_g1_cell"]})
    n_sign_match_g1 = 0
    for cell in sorted({r["cell"] for r in rows}):
        sub = [r for r in rows if r["cell"] == cell]
        d_fp32 = np.array([r["delta_fp32_pp"] for r in sub])
        d_int8 = np.array([r["delta_int8_pp"] for r in sub])
        diff = d_int8 - d_fp32
        cell_sign_match = (d_fp32.mean() >= 0) == (d_int8.mean() >= 0)
        print(f"{cell:20s} Delta_FP32={d_fp32.mean():+.2f}+/-{d_fp32.std():.2f}pp "
              f"Delta_INT8={d_int8.mean():+.2f}+/-{d_int8.std():.2f}pp "
              f"|diff|={np.abs(diff).mean():.2f}pp cell_sign_match={cell_sign_match} "
              f"{'[G1 cell]' if cell in g1_cells else '[H2 prep only, not scored for G1]'}")
        if cell in g1_cells:
            n_sign_match_g1 += int(cell_sign_match)

    frac = n_sign_match_g1 / len(g1_cells) if g1_cells else float("nan")
    verdict = "GO" if frac >= 0.8 else "NO-GO"
    print(f"\nG1: {n_sign_match_g1}/{len(g1_cells)} significant-FP32 cells keep sign under INT8 "
          f"({frac:.0%}) -> {verdict} (threshold: >=80% per mo-hinh-hoa Muc 4)")


if __name__ == "__main__":
    main()

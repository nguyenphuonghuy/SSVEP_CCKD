"""H2 (mo-hinh-hoa Muc 5): precompute the test-set + calibration-set numpy arrays
needed for the hls4ml bit-width sweep, once per (backbone, student, split_seed)
-- these only depend on the data split, not on the training seed, so 2 backbones
x 2 students x 5 splits = 20 small .npz bundles cover every (seed, type) config.
Written here (HPC, where the BETA .mat files live) then scp'd to ubuntu-huy,
which has no access to /gpfs0/users/gv_nphuy/. Must run under PBS (no
torch/dataset import on the login node).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets.public_ssvep import PublicSSVEPDataset
from utils.config import load_yaml, deep_get
import training.cckd_pipeline as P

DATASET = "beta"
SPLIT = (42, 8, 20)
N_SUBJECTS = 70
SPLITS = [2026, 3026, 4026, 5026, 6026]
WINDOW_S = 1.0
T_START = 0.14
DATA_ROOT = "/gpfs0/users/gv_nphuy/projects/ssvep_ensemble/data/raw/beta"

CELLS = [("eegnet", "3C"), ("eegnet", "1C"), ("deepconvnet", "3C")]

OUT_DIR = ROOT / "results" / "module3" / "h2_bitwidth" / "data_bundles"


def main():
    cfg = load_yaml(str(ROOT / "configs" / f"{DATASET}.yaml"))
    ch_indices = deep_get(cfg, "dataset.teacher_channel_indices")
    mat_key = deep_get(cfg, "dataset.mat_key", "data")

    print("Loading BETA dataset once...", flush=True)
    loader = PublicSSVEPDataset(DATA_ROOT, DATASET, sfreq=250.0, t_start=T_START,
                                 window_seconds=WINDOW_S, mat_key=mat_key, verbose=False)
    X64, y, s = loader.load_all_subjects(N_SUBJECTS)
    X9 = X64[:, ch_indices, :]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for backbone, student in CELLS:
        for split_seed in SPLITS:
            tr_s, va_s, te_s = P.fixed_split(s, *SPLIT, split_seed=split_seed)
            m_va, m_te = P.masks_for(s, va_s), P.masks_for(s, te_s)
            Xc = P.subset(X9, student)
            Xva = P.prep_input(Xc[m_va], backbone, n_bands=3).astype(np.float32)
            Xte = P.prep_input(Xc[m_te], backbone, n_bands=3).astype(np.float32)
            yte, ste = y[m_te].astype(np.int64), s[m_te].astype(np.int64)
            out = OUT_DIR / f"beta_{backbone}_{student}_s{split_seed}.npz"
            np.savez_compressed(out, Xva=Xva, Xte=Xte, yte=yte, ste=ste)
            print(f"  wrote {out} Xva={Xva.shape} Xte={Xte.shape}", flush=True)


if __name__ == "__main__":
    main()

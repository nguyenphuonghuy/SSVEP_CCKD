"""Module 3 / H1-H2 prep: real parameter counts for the 5 backbones x 2 montages.

Replaces the "max temporal kernel" proxy used so far with actual trainable
parameter counts, needed to pick the smallest/largest backbone for H2 and to
model inference latency for H3. Must run under PBS (no torch import on the
login node, per may/hpc-adminsm-bvnam.md).
"""
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.cckd_backbones import build_backbone, count_params

N_SAMPLES = 250  # window 1.0 s @ 250 Hz
N_CLASSES = 40
N_BANDS = 3
MONTAGES = {"3C": 3, "1C": 1}
# the 5 backbones evaluated in the paper (BACKBONE_REGISTRY also has "tcnn",
# a 1-band FBTCNN variant not used here)
BACKBONES = ["fbtcnn", "deepconvnet", "eegnet", "compactcnn", "shallowconvnet"]

out_path = Path(__file__).resolve().parents[1] / "results" / "module3" / "h1_prep" / "param_counts.csv"
out_path.parent.mkdir(parents=True, exist_ok=True)

rows = []
for backbone in BACKBONES:
    for montage, n_ch in MONTAGES.items():
        model = build_backbone(backbone, n_ch, N_SAMPLES, N_CLASSES, N_BANDS)
        n_params = count_params(model)
        rows.append({"backbone": backbone, "montage": montage, "n_channels": n_ch, "n_params": n_params})
        print(f"{backbone:15s} {montage:4s} n_params={n_params}")

with open(out_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["backbone", "montage", "n_channels", "n_params"])
    w.writeheader()
    w.writerows(rows)
print(f"\nwrote {out_path}")

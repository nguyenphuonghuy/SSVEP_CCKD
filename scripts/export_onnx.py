"""Module 3 / H1-H2: export the retrained-with-checkpoint student and baseline
models (results/module3/h1_retrain/<cell>/ckpt/*.pt) to ONNX.

Each .pt was written by training/train_cckd.py's --save_ckpt_dir (see
mo-hinh-hoa/embedded_h1h4_20261101.md Muc 5). This script rebuilds the exact
backbone from the checkpoint's own metadata, loads the weights, exports to
ONNX (opset 17, dynamic batch axis), then verifies the export by comparing
ONNX Runtime output against the original PyTorch forward pass on random
input -- an export that silently changes outputs would invalidate H1's PTQ
comparison and H2's hls4ml conversion. Must run under PBS (no torch import
on the login node, per may/hpc-adminsm-bvnam.md).

Usage:
    python scripts/export_onnx.py --root results/module3/h1_retrain
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.cckd_backbones import build_backbone

OPSET = 17


def export_one(ckpt_path: Path, onnx_dir: Path, atol: float = 1e-4) -> dict:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model = build_backbone(ckpt["backbone"], ckpt["n_channels"], ckpt["n_samples"],
                            ckpt["n_classes"], n_bands=ckpt["n_bands"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    dummy = torch.randn(1, ckpt["n_channels"], ckpt["n_samples"], dtype=torch.float32)
    onnx_path = onnx_dir / (ckpt_path.stem + ".onnx")
    onnx_dir.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model, dummy, str(onnx_path),
        input_names=["eeg"], output_names=["logits"],
        dynamic_axes={"eeg": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=OPSET,
    )

    with torch.no_grad():
        ref = model(dummy).numpy()
    import onnxruntime as ort
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    got = sess.run(None, {"eeg": dummy.numpy()})[0]
    max_abs_diff = float(np.max(np.abs(ref - got)))
    ok = max_abs_diff < atol
    return {"ckpt": str(ckpt_path), "onnx": str(onnx_path), "max_abs_diff": max_abs_diff, "ok": ok}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/module3/h1_retrain",
                     help="directory holding <cell>/ckpt/*.pt subfolders")
    ap.add_argument("--atol", type=float, default=1e-4)
    args = ap.parse_args()

    root = ROOT / args.root
    ckpts = sorted(root.glob("*/ckpt/*.pt"))
    if not ckpts:
        raise SystemExit(f"no checkpoints found under {root}/*/ckpt/*.pt")

    n_ok, n_fail = 0, 0
    for ckpt_path in ckpts:
        onnx_dir = ckpt_path.parent.parent / "onnx"
        res = export_one(ckpt_path, onnx_dir, atol=args.atol)
        status = "OK" if res["ok"] else "MISMATCH"
        print(f"[{status}] {ckpt_path.name} -> {Path(res['onnx']).name} "
              f"max_abs_diff={res['max_abs_diff']:.2e}")
        n_ok += res["ok"]
        n_fail += not res["ok"]

    print(f"\n{n_ok}/{len(ckpts)} exported and verified (tol={args.atol:g}); {n_fail} mismatch(es)")
    if n_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

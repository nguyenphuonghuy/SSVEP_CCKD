"""Regression tests for the Module 3 feature API (run on a compute node, needs torch).

    python tests/test_module3_features.py --old_backbones /path/to/old_cckd_backbones.py

1. Logits and the legacy return_feat=True features are bit-identical to the old
   implementation (same weights, same RNG seed), in eval AND train mode (dropout).
2. return_feat="all" yields the same logits; "last" equals the legacy penultimate
   feature; FB-tCNN "mid736" equals the legacy feat_mid; "mid" has the same width
   for 9/3/1-channel inputs (electrode dimension already collapsed).
3. Gram-form linear CKA equals the feature-form formula.
4. New student montages index the intended electrodes.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import models.cckd_backbones as NEW
import training.cckd_pipeline as P

BACKBONES = ["fbtcnn", "eegnet", "deepconvnet", "compactcnn", "shallowconvnet"]


def load_old(path):
    spec = importlib.util.spec_from_file_location("old_cckd_backbones", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def make_input(name, C, T=250, n=6, seed=0):
    g = np.random.default_rng(seed)
    x = g.normal(size=(n, C, T)).astype(np.float32)
    return torch.from_numpy(P.prep_input(x, name, 3))


def run(model, x, mode, seed, train):
    model.train(train)
    torch.manual_seed(seed)
    with torch.no_grad():
        return model(x, return_feat=mode)


def check_backbone(name, OLD):
    for C in (9, 3, 1):
        x = make_input(name, C)
        new = NEW.build_backbone(name, C, 250, 40, n_bands=3)
        old = OLD.build_backbone(name, C, 250, 40, n_bands=3)
        old.load_state_dict(new.state_dict())  # also proves the parameter layout is unchanged
        for train in (False, True):
            zo, fo = run(old, x, True, 123, train)
            zn, fn = run(new, x, True, 123, train)
            assert torch.equal(zo, zn), f"{name} C={C} train={train}: logits differ from old"
            assert len(fo) == len(fn) and all(torch.equal(a, b) for a, b in zip(fo, fn)), \
                f"{name} C={C} train={train}: legacy features differ from old"
            za, d = run(new, x, "all", 123, train)
            assert torch.equal(za, zn), f"{name} C={C} train={train}: 'all' changes logits"
            assert torch.equal(d["last"], fn[-1]), f"{name}: 'last' != legacy penultimate"
            if name == "fbtcnn":
                assert torch.equal(d["mid736"], fn[0]), "fbtcnn: 'mid736' != legacy feat_mid"
        _, d = run(new, x, "all", 0, False)
        widths = {k: v.flatten(1).shape[1] for k, v in d.items()}
        print(f"  {name:15s} C={C}: legacy={[f.shape[1] for f in fn]} named={widths}")
        if C == 9:
            mid9 = widths["mid"]
        else:
            assert widths["mid"] == mid9, f"{name}: mid width depends on C"


def check_cka():
    g = torch.Generator().manual_seed(0)
    X = torch.randn(300, 50, generator=g, dtype=torch.float64)
    Y = X @ torch.randn(50, 20, generator=g, dtype=torch.float64) + 0.5 * torch.randn(300, 20, generator=g, dtype=torch.float64)
    Xc, Yc = X - X.mean(0), Y - Y.mean(0)
    ref = (torch.linalg.norm(Xc.T @ Yc) ** 2 / (torch.linalg.norm(Xc.T @ Xc) * torch.linalg.norm(Yc.T @ Yc))).item()
    got = P.linear_cka(X, Y)
    assert abs(ref - got) < 1e-10, (ref, got)
    assert abs(P.linear_cka(X, X) - 1.0) < 1e-12
    print(f"  CKA gram={got:.6f} feature-form={ref:.6f}")


def check_montages():
    want = {"3C": ["O1", "Oz", "O2"], "1C": ["Oz"], "3C_PPO": ["PO3", "POz", "PO4"],
            "3C_POO": ["POz", "O1", "O2"], "1C_POz": ["POz"], "1C_O1": ["O1"]}
    for k, names in want.items():
        assert [P.CH_NAMES[i] for i in P.CH_POS[k]] == names, k
    print("  montages OK:", {k: v for k, v in want.items()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old_backbones", required=True)
    a = ap.parse_args()
    OLD = load_old(a.old_backbones)
    torch.use_deterministic_algorithms(False)
    for name in BACKBONES:
        check_backbone(name, OLD)
    check_cka()
    check_montages()
    print("MODULE3 FEATURE TESTS PASSED")


if __name__ == "__main__":
    main()

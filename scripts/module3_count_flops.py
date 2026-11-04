"""Module 3 / H3 G3 fallback (mo-hinh-hoa/embedded_h1h4_20261101.md Muc 5): real
FLOPs cho 5 backbone x 2 montage, thay cho proxy "max temporal kernel" da bi
NO-GO (xep hang latency do duoc tren Pi4/ZCU104 chi khop 1/5 voi xep hang theo
kernel -- xem results/module3/h3_latency/analysis_vs_prediction.csv).

Dem FLOPs bang forward hook tren Conv2d/Linear (nhan-cong thuc chuan: mul+add
= 2 FLOPs/MAC), batch=1, cung cau hinh voi H3 latency (N_SAMPLES=250, 1.0s @
250Hz, N_CLASSES=40, N_BANDS=3, trong so ngau nhien vi FLOPs khong phu thuoc
gia tri trong so). Phai chay qua PBS (khong import torch tren login node,
theo may/hpc-adminsm-bvnam.md).
"""
import csv
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.cckd_backbones import apply_filter_bank, build_backbone

N_SAMPLES = 250  # window 1.0 s @ 250 Hz, giong H3 latency
N_CLASSES = 40
N_BANDS = 3
MONTAGES = {"3C": 3, "1C": 1}
BACKBONES = ["fbtcnn", "deepconvnet", "eegnet", "compactcnn", "shallowconvnet"]

out_path = Path(__file__).resolve().parents[1] / "results" / "module3" / "h3_latency" / "flops_counts.csv"
out_path.parent.mkdir(parents=True, exist_ok=True)


def make_hook(record, name):
    def hook(module, inputs, output):
        x = inputs[0]
        if isinstance(module, nn.Conv2d):
            out_h, out_w = output.shape[-2], output.shape[-1]
            kh, kw = module.kernel_size
            in_ch_per_group = module.in_channels // module.groups
            macs = kh * kw * in_ch_per_group * module.out_channels * out_h * out_w
            bias_adds = module.out_channels * out_h * out_w if module.bias is not None else 0
            flops = 2 * macs + bias_adds
        elif isinstance(module, nn.Linear):
            macs = module.in_features * module.out_features
            bias_adds = module.out_features if module.bias is not None else 0
            flops = 2 * macs + bias_adds
        else:
            return
        record.append({"layer": name, "type": type(module).__name__, "flops": flops})
    return hook


def count_flops(model, x):
    record = []
    handles = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            handles.append(module.register_forward_hook(make_hook(record, name)))
    model.eval()
    with torch.no_grad():
        model(x)
    for h in handles:
        h.remove()
    return sum(r["flops"] for r in record), record


rows = []
torch.manual_seed(0)
for backbone in BACKBONES:
    for montage, n_ch in MONTAGES.items():
        model = build_backbone(backbone, n_ch, N_SAMPLES, N_CLASSES, N_BANDS)
        x_raw = torch.randn(1, n_ch, N_SAMPLES)
        if backbone == "fbtcnn":
            x = torch.from_numpy(apply_filter_bank(x_raw.numpy(), bands=((6, 50), (14, 50), (22, 50))[:N_BANDS]).astype("float32"))
        else:
            x = x_raw
        total_flops, _ = count_flops(model, x)
        rows.append({"backbone": backbone, "montage": montage, "n_channels": n_ch,
                     "flops": total_flops, "mflops": round(total_flops / 1e6, 3)})
        print(f"{backbone:15s} {montage:4s} flops={total_flops:,} ({total_flops/1e6:.3f} MFLOPs)")

with open(out_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["backbone", "montage", "n_channels", "flops", "mflops"])
    w.writeheader()
    w.writerows(rows)
print(f"\nwrote {out_path}")

"""H3 (mo-hinh-hoa/embedded_h1h4_20261101.md Muc 5): latency CPU cho 5 backbone
(FB-tCNN, DeepConvNet, EEGNet, Compact-CNN, ShallowConvNet) x 2 do rong (FP32,
INT8) tren Pi 4 / ZCU104 PS. Montage co dinh 3C, BETA n_classes=40, cua so 1.0s
(250 mau @ 250Hz), batch=1, giong quy uoc cua H2 (mo-hinh-hoa Muc 3).

Trong so ngau nhien (seed=0): latency chi phu thuoc kien truc/kich thuoc dau
vao, khong phu thuoc gia tri trong so da hoc (giong deploy/pi4_latency/
cckd_latency_pi.py cua module2).

INT8 = torch.quantization.quantize_dynamic({Conv2d, Linear}, qint8), KHONG
dung static quantization: da kiem bang smoke test tren ca pi4 va zcu104,
static quantization (QuantStub/DeQuantStub, backend qnnpack) bao loi
NotImplementedError cho FB-tCNN/DeepConvNet (aten::elu.out) va ShallowConvNet
(aten::pow.Tensor_Scalar_out) -- cac op nay chua co kernel QuantizedCPU.
Dynamic quantization la duy nhat chay duoc dong nhat tren ca 5 backbone.
Gioi han: dynamic quant chi luong tu hoa trong so Conv/Linear, kich hoat van
o fp32 giua cac lop (khong giong INT8 "full" cua H1/export ONNX) -- ghi ro
trong bao cao, khong goi la cung muc INT8 voi H1.

Chay doc lap tren moi thiet bi (khong import tu project, file nay tu chua du
cckd_backbones.py phai duoc chep cung thu muc).

Usage:
    python h3_latency_bench.py --device "Raspberry Pi 4" --out run_1.json
    python h3_latency_bench.py --device "ZCU104 PS" --out run_1.json
"""
from __future__ import annotations

import argparse
import json
import platform
import resource
import statistics as stats
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cckd_backbones import apply_filter_bank, build_backbone  # copied alongside this script

N_CLASSES = 40
N_SAMPLES = 250          # 1.0s @ 250Hz
N_CHANNELS = 3           # montage co dinh (mo-hinh-hoa Muc 3)
N_BANDS = 3
BACKBONES = ["fbtcnn", "deepconvnet", "eegnet", "compactcnn", "shallowconvnet"]
WIDTHS = ["fp32", "int8"]

N_WARMUP = 20
N_TRIALS = 100


def percentile(values, p):
    values = sorted(values)
    k = (len(values) - 1) * (p / 100)
    f, c = int(k), min(int(k) + 1, len(values) - 1)
    if f == c:
        return values[f]
    return values[f] + (values[c] - values[f]) * (k - f)


def summarize(times_ms):
    return {
        "mean_ms": stats.mean(times_ms),
        "std_ms": stats.stdev(times_ms) if len(times_ms) > 1 else 0.0,
        "median_ms": stats.median(times_ms),
        "p95_ms": percentile(times_ms, 95),
        "p99_ms": percentile(times_ms, 99),
        "min_ms": min(times_ms),
        "max_ms": max(times_ms),
        "n_trials": len(times_ms),
    }


def bench_filterbank(x_raw, n_trials=N_TRIALS, n_warmup=N_WARMUP):
    for _ in range(n_warmup):
        apply_filter_bank(x_raw)
    times = []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        apply_filter_bank(x_raw)
        times.append((time.perf_counter() - t0) * 1000.0)
    return times


def bench_model(model, x, n_trials=N_TRIALS, n_warmup=N_WARMUP):
    model.eval()
    with torch.no_grad():
        for _ in range(n_warmup):
            model(x)
        times = []
        for _ in range(n_trials):
            t0 = time.perf_counter()
            model(x)
            times.append((time.perf_counter() - t0) * 1000.0)
    return times


def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def run_config(backbone: str, width: str, n_threads: int):
    torch.backends.quantized.engine = "qnnpack"
    torch.set_num_threads(n_threads)
    torch.manual_seed(0)
    np.random.seed(0)

    model = build_backbone(backbone, N_CHANNELS, N_SAMPLES, N_CLASSES, n_bands=N_BANDS)
    model.eval()
    n_params = count_params(model)

    x_raw = np.random.default_rng(0).normal(size=(1, N_CHANNELS, N_SAMPLES)).astype(np.float32)
    if backbone == "fbtcnn":
        fb_times = bench_filterbank(x_raw)
        x_np = apply_filter_bank(x_raw, bands=((6, 50), (14, 50), (22, 50))[:N_BANDS])
    else:
        fb_times = None
        x_np = x_raw
    x_t = torch.from_numpy(x_np.astype(np.float32))

    if width == "int8":
        model = torch.quantization.quantize_dynamic(
            model, {nn.Conv2d, nn.Linear}, dtype=torch.qint8
        )

    model_times = bench_model(model, x_t)
    fb_s = summarize(fb_times) if fb_times is not None else None
    model_s = summarize(model_times)
    total_mean = (fb_s["mean_ms"] if fb_s else 0.0) + model_s["mean_ms"]
    return {
        "n_channels": N_CHANNELS, "n_params": n_params,
        "filterbank": fb_s, "model_forward": model_s,
        "total_pipeline_mean_ms": total_mean,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", required=True, choices=["Raspberry Pi 4", "ZCU104 PS"])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    result = {
        "device": args.device, "platform": platform.platform(),
        "python": platform.python_version(), "torch": torch.__version__,
        "window_s": N_SAMPLES / 250.0, "n_samples": N_SAMPLES, "n_channels": N_CHANNELS,
        "n_classes": N_CLASSES, "n_bands": N_BANDS,
        "note": "H3 (mo-hinh-hoa/embedded_h1h4_20261101.md). Trong so ngau nhien (seed=0), "
                "latency khong phu thuoc gia tri trong so. INT8 = torch.quantization."
                "quantize_dynamic({Conv2d,Linear}, qint8) -- static quantization (qnnpack) "
                "bi NotImplementedError tren FB-tCNN/DeepConvNet/ShallowConvNet (elu/pow "
                "chua co kernel QuantizedCPU), nen dung dynamic cho ca 5 backbone de dong "
                "nhat; dynamic quant chi luong tu hoa trong so Conv/Linear, kich hoat van fp32.",
        "configs": {},
    }
    for n_threads in [1, 4]:
        tk = f"{n_threads}_threads"
        result["configs"][tk] = {}
        for backbone in BACKBONES:
            for width in WIDTHS:
                key = f"{backbone}_{width}"
                r = run_config(backbone, width, n_threads)
                result["configs"][tk][key] = r
                print(f"[{tk}] {key}: model_forward={r['model_forward']['mean_ms']:.4f}ms "
                      f"total={r['total_pipeline_mean_ms']:.4f}ms params={r['n_params']}",
                      flush=True)

    peak_rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result["peak_rss_mb"] = round(peak_rss_kb / 1024, 2)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()

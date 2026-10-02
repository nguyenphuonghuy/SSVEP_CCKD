"""
cckd_latency_pi.py
===================================================================
Real-time feasibility latency benchmark for the CCKD student/teacher
FB-tCNN models on Raspberry Pi 4 (CPU, aarch64).

Measures the FULL per-trial inference pipeline: filter-bank preprocessing
(scipy Butterworth, matches training/inference exactly) + model forward
pass, batch=1, for teacher(9C)/student(3C)/student(1C), at 1.0s window
(250 samples @ 250Hz).

Latency depends only on architecture and input size, NOT on learned
weight values, so randomly-initialized weights (seed=0) are valid for
this purpose -- no checkpoint transfer needed.

Claim under test: does the low-electrode CCKD student pipeline meet
real-time requirements (window length + processing time) on cheap
embedded hardware -- NOT "is student faster than teacher" (parameter
counts are nearly identical across channel configs for this
architecture; see paper Discussion).

Usage:
    python cckd_latency_pi.py --out cckd_latency_summary.json
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import resource
import statistics as stats
import time
from typing import Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

N_CLASSES = 40
N_SAMPLES = 250          # 1.0s @ 250Hz -- matches paper's window
N_BANDS = 3
BANDS = ((6, 50), (14, 50), (22, 50))
CONFIGS = {"teacher_9C": 9, "student_3C": 3, "student_1C": 1}

N_WARMUP = 20
N_TRIALS = 200


# ===================================================================
# Filter bank (identical to models/cckd_backbones.apply_filter_bank)
# ===================================================================
def apply_filter_bank(X: np.ndarray, fs: float = 250,
                       bands: Sequence[Tuple[float, float]] = BANDS) -> np.ndarray:
    from scipy.signal import butter, filtfilt
    N, C, T = X.shape
    out = np.empty((N, len(bands), C, T), dtype=np.float32)
    nyq = fs / 2.0
    for bi, (lo, hi) in enumerate(bands):
        b, a = butter(4, [lo / nyq, min(hi, nyq - 1) / nyq], btype="band")
        out[:, bi] = filtfilt(b, a, X, axis=-1).astype(np.float32)
    return out


# ===================================================================
# FBTCNN (identical architecture to models/cckd_backbones.FBTCNN)
# ===================================================================
class _SameConvStride(nn.Module):
    def __init__(self, cin, cout, kw, stride):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, (1, kw), stride=(1, stride))
        self.kw, self.stride = kw, stride

    def forward(self, x):
        W = x.shape[-1]
        out_w = math.ceil(W / self.stride)
        pad = max((out_w - 1) * self.stride + self.kw - W, 0)
        return self.conv(F.pad(x, (pad // 2, pad - pad // 2)))


class FBTCNN(nn.Module):
    def __init__(self, n_channels, n_samples, n_classes, n_bands=3,
                 out_channel=16, lenth=5, drop=0.4):
        super().__init__()
        self.n_bands = n_bands
        self.conv1 = nn.Conv2d(1, out_channel, (n_channels, 1))
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.conv2 = _SameConvStride(out_channel, out_channel, kw=n_samples, stride=5)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.conv3 = nn.Conv2d(out_channel, out_channel, (1, lenth))
        self.bn3 = nn.BatchNorm2d(out_channel)
        w2 = math.ceil(n_samples / 5)
        w3 = w2 - lenth + 1
        self.conv4 = nn.Conv2d(out_channel, 32, (1, w3))
        self.bn4 = nn.BatchNorm2d(32)
        self.drop = nn.Dropout(drop)
        self.classifier = nn.Linear(32, n_classes)

    def _branch(self, xb):
        x = F.elu(self.bn1(self.conv1(xb)))
        x = self.drop(x)
        x = F.elu(self.bn2(self.conv2(x)))
        x = self.drop(x)
        x = F.elu(self.bn3(self.conv3(x)))
        return x

    def forward(self, x):
        feats = [self._branch(x[:, b:b + 1]) for b in range(x.shape[1])]
        x = torch.stack(feats, 0).sum(0)
        x = self.drop(x)
        x = F.elu(self.bn4(self.conv4(x)))
        feat = x.flatten(1)
        return self.classifier(self.drop(feat))


def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


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


def run_config(n_threads: int):
    torch.set_num_threads(n_threads)
    torch.manual_seed(0)
    np.random.seed(0)

    out = {}
    x_raw_9c = np.random.default_rng(0).normal(size=(1, 9, N_SAMPLES)).astype(np.float32)

    for name, n_ch in CONFIGS.items():
        x_raw = x_raw_9c[:, :n_ch, :]  # channel subset, matches CH_POS ordering irrelevant for latency
        fb_times = bench_filterbank(x_raw)
        x_fb = apply_filter_bank(x_raw)  # [1, n_bands, n_ch, T]
        x_t = torch.from_numpy(x_fb)

        model = FBTCNN(n_ch, N_SAMPLES, N_CLASSES, n_bands=N_BANDS)
        n_params = count_params(model)
        model_times = bench_model(model, x_t)

        fb_s = summarize(fb_times)
        model_s = summarize(model_times)
        total_mean = fb_s["mean_ms"] + model_s["mean_ms"]

        out[name] = {
            "n_channels": n_ch, "n_params": n_params,
            "filterbank": fb_s, "model_forward": model_s,
            "total_pipeline_mean_ms": total_mean,
        }
        print(f"  [{n_threads}thr] {name} (C={n_ch}, params={n_params}): "
              f"filterbank={fb_s['mean_ms']:.3f}ms model={model_s['mean_ms']:.3f}ms "
              f"TOTAL={total_mean:.3f}ms")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="cckd_latency_summary.json")
    args = ap.parse_args()

    result = {
        "device": "Raspberry Pi 4", "platform": platform.platform(),
        "python": platform.python_version(), "torch": torch.__version__,
        "window_s": N_SAMPLES / 250.0, "n_samples": N_SAMPLES, "n_bands": N_BANDS,
        "note": "Randomly-initialized weights (seed=0) -- latency depends only on "
                "architecture/input size, not learned weight values. "
                "total_pipeline_mean_ms = filter-bank (scipy) + model forward pass, "
                "matching the real end-to-end per-trial inference cost.",
        "configs": {},
    }
    for n_threads in [1, 4]:
        print(f"=== num_threads={n_threads} ===")
        result["configs"][f"{n_threads}_threads"] = run_config(n_threads)

    peak_rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result["peak_rss_mb"] = round(peak_rss_kb / 1024, 2)
    print(f"Peak RSS memory: {result['peak_rss_mb']:.2f} MB")

    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()

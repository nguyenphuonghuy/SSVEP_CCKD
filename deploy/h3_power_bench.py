"""H3/RQ4 bo sung: do cong suat board (INA226, hwmon2, power1_input, uW) khi
chay suy luan lien tuc tren PS cua ZCU104, cho ca 5 backbone x {fp32,int8}.
Dong thoi do cong suat idle (khong chay gi) truoc va sau moi cau hinh de tru
nen. KHONG can quyen root (chi doc hwmon qua sysfs).

Phuong phap giong deploy/h3_latency_bench.py (trong so ngau nhien seed=0,
montage 3C, BETA n_classes=40, cua so 1.0s/250 mau, batch=1, INT8 = dynamic
quantization giong H3).
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cckd_backbones import apply_filter_bank, build_backbone

N_CLASSES = 40
N_SAMPLES = 250
N_CHANNELS = 3
N_BANDS = 3
BACKBONES = ["fbtcnn", "deepconvnet", "eegnet", "compactcnn", "shallowconvnet"]
WIDTHS = ["fp32", "int8"]
POWER_PATH = "/sys/class/hwmon/hwmon2/power1_input"
IDLE_S = 3.0
ACTIVE_S = 5.0

t0 = time.time()


def log(msg):
    print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)


def read_power_uw():
    with open(POWER_PATH) as f:
        return int(f.read().strip())


def sample_power(duration_s, work_fn=None):
    samples = []
    t_end = time.time() + duration_s
    n_inf = 0
    while time.time() < t_end:
        samples.append(read_power_uw())
        if work_fn is not None:
            work_fn()
            n_inf += 1
    return samples, n_inf


def build_model(backbone, width):
    torch.manual_seed(0)
    model = build_backbone(backbone, N_CHANNELS, N_SAMPLES, N_CLASSES, n_bands=N_BANDS)
    model.eval()
    if width == "int8":
        model = torch.quantization.quantize_dynamic(model, {torch.nn.Conv2d, torch.nn.Linear}, dtype=torch.qint8)
    return model


def make_input(backbone):
    x_raw = torch.randn(1, N_CHANNELS, N_SAMPLES)
    if backbone == "fbtcnn":
        x = torch.from_numpy(apply_filter_bank(x_raw.numpy(), bands=((6, 50), (14, 50), (22, 50))).astype("float32"))
    else:
        x = x_raw
    return x


results = []

log("do cong suat idle nen (3s, truoc tat ca)")
idle0, _ = sample_power(IDLE_S)
log(f"idle nen: mean={np.mean(idle0)/1e6:.3f} W, n_samples={len(idle0)}")

for backbone in BACKBONES:
    for width in WIDTHS:
        log(f"=== {backbone} {width} ===")
        model = build_model(backbone, width)
        x = make_input(backbone)
        with torch.no_grad():
            model(x)  # warmup 1 lan

        def work():
            with torch.no_grad():
                model(x)

        active_samples, n_inf = sample_power(ACTIVE_S, work_fn=work)
        idle_samples, _ = sample_power(1.0)
        mean_active = float(np.mean(active_samples))
        mean_idle_after = float(np.mean(idle_samples))
        delta = mean_active - mean_idle_after
        log(f"{backbone} {width}: active={mean_active/1e6:.3f}W idle_after={mean_idle_after/1e6:.3f}W "
            f"delta={delta/1e6:.3f}W n_inf~{n_inf} trong {ACTIVE_S}s")
        results.append({
            "device": "ZCU104_PS", "backbone": backbone, "width": width,
            "power_active_mean_uW": mean_active, "power_idle_after_uW": mean_idle_after,
            "power_delta_uW": delta, "n_active_samples": len(active_samples),
            "n_inferences_in_window": n_inf, "active_window_s": ACTIVE_S,
        })

out = Path(__file__).resolve().parent / "zcu104_power_results.json"
with open(out, "w") as f:
    json.dump({"idle_baseline_uW": float(np.mean(idle0)), "results": results}, f, indent=2)
log(f"da ghi {out}")
log("HET")

"""H4 (mo-hinh-hoa Muc 5, G5): chay suy luan THAT tren PL cua ZCU104 (bitstream
eegnet_1c_w10_accel_prj) cho EEGNet 1C, BETA split 2026, so voi nhan thuc te
(yte) va so sanh voi do chinh xac mo phong cua H2 (Keras FP32 + hls4ml W10).

Dong thoi doc cong suat qua ina226 (hwmon2, power1_input, don vi uW) truoc/
trong luc suy luan de uoc luong cong suat tang them do PL dang hoat dong.
"""
import json
import time

import numpy as np
from axi_stream_driver import NeuralNetworkOverlay

DATA = "/home/xilinx/cckd_h4_eegnet_1c/beta_eegnet_1C_s2026.npz"
BIT = "/home/xilinx/cckd_h4_eegnet_1c/design_1_wrapper.bit"
POWER_PATH = "/sys/class/hwmon/hwmon2/power1_input"

IN_SCALE = 128.0     # ap_fixed<16,9,...>: frac = 16-9 = 7 -> 2^7
OUT_SCALE = 4096.0   # ap_fixed<16,4,...>: frac = 16-4 = 12 -> 2^12


def encode(x):
    return np.clip(np.round(x * IN_SCALE), -32768, 32767).astype(np.int16)


def decode(y):
    return y.astype(np.float32) / OUT_SCALE


def read_power_uw():
    try:
        with open(POWER_PATH) as f:
            return int(f.read().strip())
    except Exception:
        return None


t0 = time.time()


def log(msg):
    print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)


log("nap du lieu test")
d = np.load(DATA)
Xte, yte, ste = d["Xte"], d["yte"], d["ste"]
n = Xte.shape[0]
log(f"n_test={n}")

log("do cong suat idle (chua nap overlay)")
p_idle = [read_power_uw() for _ in range(5)]
log(f"power idle (uW): {p_idle}")

log("nap overlay (load bitstream)")
overlay = NeuralNetworkOverlay(BIT, x_shape=(250,), y_shape=(40,), dtype=np.int16)
log("overlay da nap")

p_loaded = [read_power_uw() for _ in range(5)]
log(f"power sau khi nap bitstream, truoc suy luan (uW): {p_loaded}")

preds = np.zeros(n, dtype=np.int64)
power_samples = []
log("bat dau suy luan tung cua so")
t_start = time.time()
for i in range(n):
    x = encode(Xte[i, 0, :])
    y_raw = overlay.predict(x)
    y = decode(np.array(y_raw, dtype=np.int16))
    preds[i] = int(np.argmax(y))
    if i % 400 == 0:
        pw = read_power_uw()
        if pw is not None:
            power_samples.append(pw)
        log(f"  {i}/{n} xong, power={pw}")
t_end = time.time()
log(f"suy luan xong, {n} cua so trong {t_end - t_start:.2f}s")

acc = float((preds == yte).mean())
log(f"DO CHINH XAC TREN BOARD THAT = {acc:.4f} (n={n})")

per_subject = {}
for sub in np.unique(ste):
    mask = ste == sub
    per_subject[int(sub)] = float((preds[mask] == yte[mask]).mean())
log(f"do chinh xac theo subject: {per_subject}")

result = {
    "device": "zcu104_PL",
    "backbone": "eegnet",
    "montage": "1C",
    "dataset": "beta",
    "split": 2026,
    "seed": 2026,
    "student": "cckd_relational_last_lsp100",
    "bitwidth": 10,
    "n_test": n,
    "accuracy_board": acc,
    "accuracy_per_subject": per_subject,
    "power_idle_uW": p_idle,
    "power_loaded_uW": p_loaded,
    "power_during_inference_uW": power_samples,
    "inference_wall_time_s": t_end - t_start,
}
with open("/home/xilinx/cckd_h4_eegnet_1c/board_result.json", "w") as f:
    json.dump(result, f, indent=2)
log("da ghi board_result.json")
log("HET")

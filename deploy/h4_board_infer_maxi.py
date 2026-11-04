"""H4 (mo-hinh-hoa Muc 5, G5): chay suy luan THAT tren PL cua ZCU104 qua IP
m_axi tu viet tay (eegnet_maxi.bit/.hwh, thay cho AXI-Stream+DMA cua hls4ml
VivadoAccelerator -- mac TLAST tren cong ra sau 6 lan sua loi khac nhau).

IP dung ap_ctrl_hs qua AXI-Lite (register_map.CTRL) + hai con tro m_axi
(in_mem, out_mem) tro vao buffer DDR da allocate boi PYNQ. Day la mo hinh
chuan cua PYNQ cho IP HLS tu viet (khong can driver rieng nhu axi_stream_driver.py).

Phai chay bang "sudo python3 h4_board_infer_maxi.py" (PYNQ can quyen root de
mo /dev/fpga0, /dev/uio* -- xem ghi chu trong mo-hinh-hoa/embedded_h1h4_20261101.md).
"""
import json
import os
import time

os.environ.setdefault("XILINX_XRT", "/usr")  # thieu bien nay thi PYNQ bao "No Devices Found"

import numpy as np
from pynq import Overlay, allocate

DATA = "/home/xilinx/cckd_h4_eegnet_1c/beta_eegnet_1C_s2026.npz"
BIT = "/home/xilinx/cckd_h4_eegnet_1c/eegnet_maxi.bit"
POWER_PATH = "/sys/class/hwmon/hwmon2/power1_input"

IN_SCALE = 128.0    # ap_fixed<16,9,...>: frac = 16-9 = 7 -> 2^7
# out_mem duoc ep kieu tu result_t (ap_fixed<10,6,...>, frac=4) sang
# mem_out_t (ap_fixed<16,6,...>, frac=16-6=10) -- ep kieu ap_fixed la GIU
# GIA TRI THUC (khong phai ghi de bit), nen so nguyen 16-bit doc tu DDR phai
# chia theo scale cua kieu CHUA no trong bo nho (16,6), tuc 2^10, khong phai
# scale cua result_t goc (10,6, 2^4).
OUT_SCALE = 1024.0  # ap_fixed<16,6,...>: frac = 16-6 = 10 -> 2^10


def encode(x):
    return np.clip(np.round(x * IN_SCALE), -32768, 32767).astype(np.int16)


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

p_idle = [read_power_uw() for _ in range(5)]
log(f"power idle (uW): {p_idle}")

log("nap overlay (load bitstream) -- can quyen root")
overlay = Overlay(BIT)
ip = overlay.eegnet_0
log(f"overlay da nap. cac IP: {list(overlay.ip_dict.keys())}")
log(f"register_map: {ip.register_map}")

p_loaded = [read_power_uw() for _ in range(5)]
log(f"power sau khi nap bitstream (uW): {p_loaded}")

in_buf = allocate(shape=(250,), dtype=np.int16)
out_buf = allocate(shape=(40,), dtype=np.int16)
log(f"da allocate buffer: in={in_buf.physical_address:#x} out={out_buf.physical_address:#x}")


def run_once(x_i16):
    in_buf[:] = x_i16
    in_buf.flush()
    in_addr = in_buf.physical_address
    out_addr = out_buf.physical_address
    ip.register_map.in_mem_1 = in_addr & 0xFFFFFFFF
    ip.register_map.in_mem_2 = (in_addr >> 32) & 0xFFFFFFFF
    ip.register_map.out_mem_1 = out_addr & 0xFFFFFFFF
    ip.register_map.out_mem_2 = (out_addr >> 32) & 0xFFFFFFFF
    ip.register_map.CTRL.AP_START = 1
    n_poll = 0
    while not ip.register_map.CTRL.AP_DONE:
        n_poll += 1
        if n_poll > 2_000_000:
            raise RuntimeError("timeout cho AP_DONE")
    out_buf.invalidate()
    return out_buf.copy()


log("smoke test: 1 cua so, in ra logit + so sanh nhan")
x0 = encode(Xte[0, 0, :])
y0 = run_once(x0)
y0f = y0.astype(np.float32) / OUT_SCALE
log(f"smoke OK: out_raw[:8]={y0[:8]}, logits[:8]={y0f[:8]}")
log(f"argmax={int(np.argmax(y0f))}, nhan thuc={int(yte[0])}")

preds = np.zeros(n, dtype=np.int64)
power_samples = []
log("bat dau suy luan toan bo test set")
t_start = time.time()
for i in range(n):
    x = encode(Xte[i, 0, :])
    y_raw = run_once(x)
    y = y_raw.astype(np.float32) / OUT_SCALE
    preds[i] = int(np.argmax(y))
    if i % 400 == 0:
        pw = read_power_uw()
        if pw is not None:
            power_samples.append(pw)
        log(f"  {i}/{n} xong, power={pw}")
t_end = time.time()
log(f"suy luan xong, {n} cua so trong {t_end - t_start:.2f}s")

acc = float((preds == yte).mean())
log(f"DO CHINH XAC TREN BOARD THAT (PL, m_axi) = {acc:.4f} (n={n})")

per_subject = {}
for sub in np.unique(ste):
    mask = ste == sub
    per_subject[int(sub)] = float((preds[mask] == yte[mask]).mean())
log(f"do chinh xac theo subject: {per_subject}")

result = {
    "device": "zcu104_PL_m_axi",
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
with open("/home/xilinx/cckd_h4_eegnet_1c/board_result_maxi.json", "w") as f:
    json.dump(result, f, indent=2)
log("da ghi board_result_maxi.json")
log("HET")

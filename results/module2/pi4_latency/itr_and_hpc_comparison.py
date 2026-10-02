"""
itr_and_hpc_comparison.py
===================================================================
Two lightweight analyses to substantiate the Raspberry Pi 4 deployment
contribution (Section 5.4 of the paper), on top of the already-measured
7-run latency benchmark:

  1. Effective/practical Wolpaw ITR: combine the measured Pi 4 pipeline
     latency (filter-bank + forward pass) with the paper's actual
     reported accuracy (Table 1, FB-tCNN/CCKD) to quantify how much the
     real compute overhead actually costs in bits/min, versus the
     idealized ITR (window + gaze-shift only, zero compute time) used
     implicitly elsewhere.
  2. HPC server-CPU baseline: the identical benchmark script re-run on
     an AMD EPYC 9124 (HPC login node, CPU-only, idle load) at the same
     thread counts (1, 4), giving a same-methodology reference point for
     "how much slower is a $35 SBC than a server-grade CPU, and does it
     matter".

Both are purely post-hoc / one extra (already collected) benchmark run
-- no new Pi 4 experiments were needed.
"""
import json
import math

import numpy as np


def itr(acc, window_s, n_classes=40, gaze=0.5, extra_latency_s=0.0):
    """Wolpaw ITR (bits/min), matching training/cckd_pipeline.py::itr()."""
    p = min(max(acc, 1e-6), 1 - 1e-6)
    N = n_classes
    bits = np.log2(N) + p * np.log2(p) + (1 - p) * np.log2((1 - p) / (N - 1))
    T = window_s + gaze + extra_latency_s
    return float(bits * 60.0 / T)


# Accuracy from Table 1 (main.tex), FB-tCNN backbone, Proposed CCKD column.
ACC = {
    "Benchmark": {"teacher_9C": 0.6560, "student_3C": 0.5122, "student_1C": 0.4529},
    "BETA": {"teacher_9C": 0.5359, "student_3C": 0.3950, "student_1C": 0.3543},
}
CONFIGS = ["teacher_9C", "student_3C", "student_1C"]

pi = json.load(open("cckd_latency_aggregated_1s.json"))
hpc = json.load(open("hpc_cpu_baseline/hpc_cpu_aggregated.json"))

out = {"speedup_pi4_vs_hpc_cpu": {}, "itr_ideal_vs_practical": {}}

for tk in ["1_threads", "4_threads"]:
    out["speedup_pi4_vs_hpc_cpu"][tk] = {}
    for cfg in CONFIGS:
        pi_ms = pi["results"][tk][cfg]["total_pipeline"]["mean_ms"]
        hpc_ms = hpc[tk][cfg]["mean_ms"]
        out["speedup_pi4_vs_hpc_cpu"][tk][cfg] = {
            "pi4_ms": pi_ms, "hpc_epyc9124_ms": hpc_ms, "pi4_slower_factor": pi_ms / hpc_ms,
        }

for ds in ACC:
    out["itr_ideal_vs_practical"][ds] = {}
    for cfg in CONFIGS:
        a = ACC[ds][cfg]
        lat_s = pi["results"]["1_threads"][cfg]["total_pipeline"]["mean_ms"] / 1000.0
        i_ideal = itr(a, 1.0, extra_latency_s=0.0)
        i_practical = itr(a, 1.0, extra_latency_s=lat_s)
        out["itr_ideal_vs_practical"][ds][cfg] = {
            "acc": a, "pi4_1thread_latency_ms": lat_s * 1000.0,
            "itr_ideal_bits_per_min": i_ideal, "itr_practical_bits_per_min": i_practical,
            "itr_drop_pct": 100.0 * (i_ideal - i_practical) / i_ideal,
        }

with open("itr_and_hpc_comparison.json", "w") as f:
    json.dump(out, f, indent=2)

print("=== Pi4 slower factor vs HPC EPYC 9124 CPU (same script, same thread counts) ===")
for tk in out["speedup_pi4_vs_hpc_cpu"]:
    for cfg, v in out["speedup_pi4_vs_hpc_cpu"][tk].items():
        print(f"  {tk} {cfg}: {v['pi4_slower_factor']:.2f}x slower "
              f"(Pi4={v['pi4_ms']:.3f}ms, HPC={v['hpc_epyc9124_ms']:.3f}ms)")

print("\n=== ITR: ideal vs practical (+ Pi4 1-thread compute latency) ===")
for ds in out["itr_ideal_vs_practical"]:
    for cfg, v in out["itr_ideal_vs_practical"][ds].items():
        print(f"  {ds} {cfg}: ITR_ideal={v['itr_ideal_bits_per_min']:.2f} "
              f"ITR_practical={v['itr_practical_bits_per_min']:.2f} bits/min "
              f"(-{v['itr_drop_pct']:.3f}%)")

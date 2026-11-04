#!/bin/bash
# H3: 7 lan chay doc lap (7 tien trinh python rieng), cooldown 15s giua cac lan,
# cung quy uoc voi deploy/pi4_latency/run_cckd_repeated.sh cua module2.
set -e
DEVICE="$1"   # "Raspberry Pi 4" hoac "ZCU104 PS"
OUTDIR="$2"   # vd: h3_runs
N_RUNS=7
COOLDOWN=15

mkdir -p "$OUTDIR"
for i in $(seq 1 $N_RUNS); do
  echo "=== run $i/$N_RUNS ==="
  python3 h3_latency_bench.py --device "$DEVICE" --out "$OUTDIR/run_${i}.json"
  if [ "$i" -lt "$N_RUNS" ]; then sleep $COOLDOWN; fi
done
echo "H3_SWEEP_DONE"

#!/usr/bin/env bash
set -euo pipefail

PROJ=/gpfs0/users/gv_nphuy/projects/cckd_ssvep
PY=/gpfs0/users/gv_nphuy/envs/cckd_ssvep/bin/python

DATA_ROOT=${1:?Usage: bash scripts/run_benchmark_smoke.sh /absolute/path/to/Benchmark}

cd "$PROJ"

"$PY" scripts/inspect_dataset.py \
    --config configs/benchmark.yaml \
    --data_root "$DATA_ROOT" \
    --subject 1

"$PY" training/train_teacher.py \
    --config configs/benchmark.yaml \
    --data_root "$DATA_ROOT" \
    --n_folds 1 \
    --epochs 5 \
    --workers 0 \
    --device cpu \
    --output_dir results/module1/benchmark_smoke

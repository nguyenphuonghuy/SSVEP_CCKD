# Resuming this project on a different HPC account

**Update (2026-10-02): the CUDA-init hang described below is fixed.** Adding
`export CUDA_MPS_PIPE_DIRECTORY=/dev/null` to the job script resolves it —
confirmed end-to-end on `gv_bvnam`/`ai01` (`scripts_smoke_gpu_test.py`
completed in 8.2s instead of hanging indefinitely; see §6). The 9 pending
jobs below already have this fix baked in and can simply be run on
`gv_bvnam` directly — **moving to a different account is no longer
necessary** for this specific issue. This guide is kept for reference (e.g.
if the hang recurs, or if running on a second account is wanted for other
reasons such as parallel GPU-slot throughput).

This guide is for picking up `cckd_ssvep` from a fresh clone on an account
other than `gv_bvnam` — the original motivation was that `gv_bvnam`'s PBS
jobs were stuck in a CUDA-init hang on node `ai01` (see "Known issue" below)
and the fastest way forward seemed to be submitting the remaining jobs from
an unaffected account.

## 1. What's already done — do NOT rerun these

Every experiment tier already has its final result JSON committed under
`results/module2/`. These are the source of truth for the tables in
`paper/latex/main.tex`; re-running them is unnecessary and would only burn
GPU time. Completed tiers:

- `tier0_multisplit` — FB-tCNN main result, 5 subject splits × 8 seeds, both datasets.
- `tier1_multisplit`, `tier1b_eegnet_multisplit` — DeepConvNet/ShallowConvNet/Compact-CNN/EEGNet, 5-split confirmation.
- `*_confirm_*_8seed` dirs — 8-seed confirmatory single-split replications per backbone.
- `cka` — linear-CKA representation analysis.
- `sota_calib`, `sota_full` — FitNets / Attention Transfer, λ calibration + single-split comparison.
- `tier4_fitnets_ms`, `tier4_attention_ms` — FitNets / Attention Transfer, 5-split confirmation.
- `tier5_lambda_sweep` — λ_sp sweep (0.1–1000) for FitNets/Attention across all 4 configs.
- `tier5_logit_ms` — Logit-KD, 5-split confirmation for τ ∈ {2, 4, 8}.
- `tier2_window` — window-length sweep (0.5/1.0/1.5/2.0 s), single (canonical) split.
- `tier2_window_ms` — window-length sweep, multi-split confirmation: **15 of 24 configs done** (see next section for the 9 missing ones).
- `eegnet_lambda_sweep_*`, `eegnet_strategy_sweep_*` — EEGNet negative-result robustness checks.

## 2. What's still pending — must be run from scratch

**9 configs under `tier2_window_ms` have no result JSON yet.** Their
`checkpoint.json` (per-seed resume state) is intentionally excluded from Git
(`.gitignore`), so there is no partial progress to resume from — these start
at seed 1 regardless of what `gv_bvnam` attempted.

| window | dataset | split_seed | job script |
|---|---|---|---|
| 2.0 s | benchmark | 3026, 4026, 5026, 6026 | `jobs/tier2_window_ms/w20_bm_s{split}.pbs` |
| 2.0 s | beta | 3026, 4026, 5026, 6026 | `jobs/tier2_window_ms/w20_bt_s{split}.pbs` |
| 1.5 s | beta | 6026 | `jobs/tier2_window_ms/w15_bt_s6026.pbs` |

All 9 scripts already include a self-healing watchdog (see §5) — just submit
them as-is once the paths below are fixed.

## 3. One-time setup on the new account

```bash
git clone git@github.com:nguyenphuonghuy/SSVEP_CCKD.git
cd SSVEP_CCKD
```

Create a Python environment matching `requirements.txt` (numpy, scipy,
PyYAML, torch>=2.1, h5py):

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

(or point `PY=` below at an existing conda env that already has these.)

Verify the raw-data path is readable from the new account — the datasets are
**not** copied into this repo, they stay at:

```
/gpfs0/users/gv_nphuy/projects/ssvep_ensemble/data/raw/benchmark
/gpfs0/users/gv_nphuy/projects/ssvep_ensemble/data/raw/beta
```

```bash
ls /gpfs0/users/gv_nphuy/projects/ssvep_ensemble/data/raw/benchmark | head
```

If this is not readable, the data owner needs to grant read access to the
new account's group/UID before anything else will work.

## 4. Fix the hardcoded paths

All 160 `.pbs` scripts hardcode `gv_bvnam`'s clone path and Python
interpreter:

```
PROJ=/gpfs0/users/gv_bvnam/projects/cckd_ssvep
PY=/gpfs0/users/gv_bvnam/envs/erd_tta_bci/bin/python
```

Replace both in one pass (adjust the right-hand side to the new clone path
and interpreter):

```bash
NEW_PROJ=/gpfs0/users/<new_account>/projects/cckd_ssvep   # wherever you cloned to
NEW_PY=/gpfs0/users/<new_account>/envs/<env_name>/bin/python

grep -rl "PROJ=/gpfs0/users/gv_bvnam/projects/cckd_ssvep" jobs/ | \
  xargs sed -i "s#PROJ=/gpfs0/users/gv_bvnam/projects/cckd_ssvep#PROJ=${NEW_PROJ}#g"

grep -rl "PY=/gpfs0/users/gv_bvnam/envs/erd_tta_bci/bin/python" jobs/ | \
  xargs sed -i "s#PY=/gpfs0/users/gv_bvnam/envs/erd_tta_bci/bin/python#PY=${NEW_PY}#g"
```

Double-check with:

```bash
grep -rn "gv_bvnam" jobs/ | wc -l   # should be 0 after the sed above
```

## 5. Submit the 9 pending jobs

```bash
for p in jobs/tier2_window_ms/w20_bm_s3026.pbs \
         jobs/tier2_window_ms/w20_bm_s4026.pbs \
         jobs/tier2_window_ms/w20_bm_s5026.pbs \
         jobs/tier2_window_ms/w20_bm_s6026.pbs \
         jobs/tier2_window_ms/w20_bt_s3026.pbs \
         jobs/tier2_window_ms/w20_bt_s4026.pbs \
         jobs/tier2_window_ms/w20_bt_s5026.pbs \
         jobs/tier2_window_ms/w20_bt_s6026.pbs \
         jobs/tier2_window_ms/w15_bt_s6026.pbs; do
  qsub "$p"
done
```

Each script trains for up to 8 seeds, self-checkpoints after every seed
(`<output_dir>/checkpoint.json`), and includes a watchdog: if 20 minutes pass
with no CPU progress in the training process, it kills and retries — up to
the job's walltime budget (10 h for benchmark, 12 h for beta), leaving a
~20 min safety margin to write a final status line. You do not need to
babysit it minute-by-minute; just check back periodically:

```bash
qstat -u <your_account>
tail -20 logs/w20_benchmark_s3026_*.log   # or whichever config you're checking
```

A finished run writes `cckd_fbtcnn_<dataset>_<window>s_<timestamp>.json`
into its `output_dir` and deletes `checkpoint.json`. If a job's log ends with
`STATUS: FAILED after N attempt(s)`, something is wrong at the
infrastructure level (see §6) rather than in the code — it is not expected
behavior on a healthy account/node.

## 6. Known issue: do not assume `gv_bvnam`'s hang is universal

While debugging these 9 configs on `gv_bvnam`, every job — down to a
script that does nothing but `import torch; torch.cuda.is_available()` —
hung indefinitely on node `ai01`, with near-zero CPU usage (process alive
but stuck, not slow). This reproduced across dozens of retries over
24+ hours, including idle periods with the cluster otherwise at ~15% GPU
load, so it is not generic queue congestion. Jobs from other accounts on the
same node ran normally throughout. The leading hypothesis is a stale
GPU/MPS client left behind by `gv_bvnam`'s own repeatedly-killed processes
during debugging, specific to that account.

**Confirmed fix.** If the cluster's `gpu_slots` resource is backed by NVIDIA
MPS (likely, given 10-14 slots per node sharing what is almost certainly 1-2
physical GPUs), a stuck/stale MPS client for an account can make every
subsequent `cudaSetDevice`/`cuda.is_available()` call from that account hang
while it waits on the MPS control daemon. Setting
`CUDA_MPS_PIPE_DIRECTORY=/dev/null` makes the CUDA runtime skip the MPS
client path entirely and talk to the GPU directly, sidestepping a hung MPS
client. **Verified end-to-end on `gv_bvnam`/`ai01` on 2026-10-02**:
`scripts_smoke_gpu_test.py`, which had hung indefinitely in every prior
attempt (including a 24h/66-attempt run with the watchdog, and a prior,
disproven hypothesis — `FI_PROVIDER=tcp` — which was also tested and did
*not* help), completed cleanly in 8.2s with this variable set:

```
[0.00s] starting, importing torch...
[4.24s] torch imported, checking cuda.is_available()...
[4.27s] cuda.is_available()=True
[4.27s] device name: NVIDIA H100 NVL
[5.21s] allocated tensor on cuda
[8.20s] matmul done, sum=164198.91
[8.20s] DONE OK
```

All 9 pending job scripts already set this variable (right after
`PYTHONUNBUFFERED`) and were resubmitted successfully with it. If a hang
ever recurs despite this being set, that would mean a different cause than
the one diagnosed here, and is worth re-reporting to the HPC admin with a
fresh diagnostic run.

There is also a minimal diagnostic script in the repo to confirm GPU health
in ~10 seconds before committing to a full 10h job:

```bash
cat > /tmp/gpu_check.pbs <<EOF
#!/bin/bash
#PBS -N gpu_check
#PBS -q sharedgpu
#PBS -l select=1:ncpus=2:mem=8gb:gpu_slots=1
#PBS -l walltime=00:10:00
#PBS -j oe
export CUDA_MPS_PIPE_DIRECTORY=/dev/null
${NEW_PY} ${NEW_PROJ}/scripts_smoke_gpu_test.py
EOF
qsub /tmp/gpu_check.pbs
```

(Uses `NEW_PY` / `NEW_PROJ` from §4 — run this in the same shell session
after exporting them there, or replace with the literal paths.)

(`scripts_smoke_gpu_test.py` at the repo root just imports torch, checks
`cuda.is_available()`, and does one matmul on the GPU — if it doesn't print
`DONE OK` within ~30 seconds, something is wrong at the node/account level
and is worth reporting to the HPC admin before submitting the real jobs.)

## 7. After the 9 jobs finish

No code changes are needed in `paper/latex/main.tex` to consume the new
results — `tab:window` and its surrounding discussion already reference
`tab:multisplit` conventions used elsewhere in the paper; extend the window
multi-split analysis the same way the `0.5s`/`1.5s` results were integrated
(see the git history / `CHANGELOG.md` for that pattern), i.e. aggregate
`results/module2/tier2_window_ms/*_w20_*` and the new `beta_w15_s6026`
result, run the same one-sample t-test + Holm correction used for the other
window-length rows, and update the reliability caveat paragraph if the
conclusions change.

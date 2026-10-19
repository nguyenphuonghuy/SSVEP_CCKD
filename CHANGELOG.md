# Changelog
## v0.3.0-dev: Module 3 (TMLR multi-backbone study, plan: `paper/TMLR_pivot_plan.md`)
- Found: Module 2 distilled two levels for FB-tCNN (`[feat_mid, feat]`) but only the penultimate level for the other four backbones, so backbone and number of distilled levels were confounded in the cross-backbone table.
- `models/cckd_backbones.py`: new `return_feat="all"` returns named levels `mid` (spatial-conv block output after BN + nonlinearity, defined the same way for all backbones) and `last` (penultimate); FB-tCNN also `mid736`. `return_feat=True` is unchanged (legacy).
- `training/cckd_pipeline.py`: `--kd_layers` support (`legacy|last|mid|both`), alternative montages `3C_PPO`, `3C_POO`, `1C_POz`, `1C_O1`, Gram-form linear CKA.
- `training/train_cckd.py`: `--kd_layers` (several values share one teacher/baseline per seed), per-run teacher-student CKA on the test set (`--no_cka` to skip), Python `random` seeded, per-subject CSV `{dataset}_{model}_{setting}_seed{N}_{YYYYMMDD}.csv` + `_config.json` (git commit, dirty flag).
- `tests/test_module3_features.py`: regression test against the HEAD backbones (bit-identical logits and legacy features in eval and train mode), CKA formula check, montage check. Smoke jobs: `jobs/module3/n0_smoke/`.

## v0.2.0 — Module 2
- CCKD pipeline: `models/cckd_backbones.py` (FB-tCNN multi-level, EEGNet), `training/kd_losses.py` (logit/relational/hybrid), `training/cckd_pipeline.py`, `training/train_cckd.py`.
- Real results (FB-tCNN, fixed split, 3 seeds): CCKD improves 3C+1C on BETA (p<0.05 both) and 1C on Benchmark (p=0.084); ablation shows relational KD is the only strategy with consistently positive gains, more stable than response-based/hybrid.
- EEGNet generalization check: relational-only CCKD fails to generalize (neutral on Benchmark, significant degradation on BETA), confirmed robust via a $\lambda_{sp}$ sensitivity sweep (1–2000, monotonic degradation, no positive result anywhere). Follow-up strategy sweep + 8-seed confirmatory replication found response-based/hybrid KD gives a significant, dataset-dependent benefit (Benchmark-3C: p=0.0020, 10/10 subjects; BETA-3C: not significant) — isolates the failure to the Gram-matrix relational mechanism specifically, not cross-channel KD in general.
- Raspberry Pi 4 real-hardware deployment (`deploy/pi4_latency/`): full pipeline (filter-bank + forward) latency and memory benchmark, 7 independent runs with cooldown; HPC server-CPU baseline comparison; effective-ITR impact analysis — confirms real-time feasibility with negligible (~1.1%) ITR cost.
- `paper/latex/main.tex` rewritten end-to-end with verified real data (replacing earlier fabricated/mismatched numbers); every table cross-checked against its source JSON in `results/module2/`.

## v0.1.0 — Module 1
- Standalone FB-tCNN teacher project.
- Built-in Benchmark/BETA public MAT loaders.
- Dataset location controlled by `dataset.data_root` or `--data_root`.
- Leakage-free LOSO: validation subjects select the checkpoint; test subject evaluated once.
- Filter-bank cache, logs, checkpoints, JSON results, dataset inspection, and smoke test.

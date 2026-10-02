# Module 1 acceptance criteria
1. `python tests/smoke_test.py` passes.
2. Dataset inspection reports 40 classes and the expected trials per subject.
3. One-fold training creates one `.pt`, `training.log`, and `teacher_results.json`.
4. No test accuracy is used to select an epoch.
5. Module 2 starts only after these outputs are reviewed.

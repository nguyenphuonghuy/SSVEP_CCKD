"""Select lambda_sp for CCKD by validation accuracy, then report its test result.

Rule (fixed before the sweep was run): for each dataset and electrode
configuration, choose the lambda_sp whose mean per-seed validation accuracy
(val_best_per_seed, best epoch per run) is highest over the 8 seeds. The test
accuracy of the chosen lambda is reported; test accuracy is never used to pick.
"""
import glob
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1] / "results" / "module2" / "tier6_lambda_val"
LAMBDAS = [1, 10, 100, 1000]
DATASETS = ["benchmark", "beta"]
CONFIGS = ["3C", "1C"]


def load(dataset, lam):
    files = glob.glob(str(ROOT / f"{dataset}_lam{lam}_s2026_8seed" / "*.json"))
    assert len(files) == 1, (dataset, lam, files)
    with open(files[0]) as f:
        return json.load(f)


def main():
    for dataset in DATASETS:
        print(f"== {dataset} (window 1.0 s, split 2026, FB-tCNN, relational, 8 seeds)")
        runs = {lam: load(dataset, lam) for lam in LAMBDAS}
        for cfg in CONFIGS:
            key = f"{cfg}_relational"
            val_mean = {lam: float(np.mean(d["val_best_per_seed"][key])) for lam, d in runs.items()}
            best = max(val_mean, key=val_mean.get)
            print(f"  {cfg}: mean val per lambda " +
                  ", ".join(f"{lam}={val_mean[lam]*100:.2f}" for lam in LAMBDAS) +
                  f"  -> selected lambda_sp={best}")
            for lam in LAMBDAS:
                d = runs[lam]
                st = d["runs"][key]["paired_vs_baseline"]
                test_mean = d["runs"][key]["mean_acc"] * 100
                mark = " <-- selected" if lam == best else ""
                print(f"     lambda={lam:>4}: test {test_mean:.2f}%  delta {st['mean_delta_pp']:+.2f}pp  "
                      f"Wilcoxon p={st['wilcoxon_p']:.4f}{mark}")


if __name__ == "__main__":
    main()

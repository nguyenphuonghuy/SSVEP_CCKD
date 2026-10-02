"""
aggregate_runs.py
===================================================================
Aggregate the 7 independent repeated Raspberry Pi 4 latency benchmark
runs (cckd_latency_pi.py, run_{1..7}.json) into mean/std/95% CI across
RUNS (not just across trials within one process) for each
config x thread-count, on total_pipeline_mean_ms.

Rationale: within-run statistics (N_TRIALS=200) capture measurement
noise but not run-to-run variability (OS jitter, thermal state,
background load). Aggregating across 7 independent process runs with
15s cooldown between them (matching the group's established
run_all_windows.sh N_RUNS/COOLDOWN methodology) gives a defensible
mean +/- 95% CI for the paper.
"""
import glob
import json
import math
import statistics as stats

RUNS_DIR = "runs_1s"
OUT_JSON = "cckd_latency_aggregated_1s.json"

files = sorted(glob.glob(f"{RUNS_DIR}/run_*.json"))
assert len(files) == 7, f"expected 7 run files, found {len(files)}: {files}"

runs = [json.load(open(f)) for f in files]

thread_keys = ["1_threads", "4_threads"]
configs = ["teacher_9C", "student_3C", "student_1C"]

agg = {"n_runs": len(runs), "source_files": files, "results": {}}

for tk in thread_keys:
    agg["results"][tk] = {}
    for cfg in configs:
        totals = [r["configs"][tk][cfg]["total_pipeline_mean_ms"] for r in runs]
        fb = [r["configs"][tk][cfg]["filterbank"]["mean_ms"] for r in runs]
        model = [r["configs"][tk][cfg]["model_forward"]["mean_ms"] for r in runs]
        n_params = runs[0]["configs"][tk][cfg]["n_params"]

        def summarize(vals):
            m = stats.mean(vals)
            sd = stats.stdev(vals) if len(vals) > 1 else 0.0
            # 95% CI via t-distribution, df=n-1=6 -> t=2.447
            t_crit = 2.447
            ci95 = t_crit * sd / math.sqrt(len(vals))
            return {"mean_ms": m, "std_ms": sd, "ci95_ms": ci95,
                    "min_ms": min(vals), "max_ms": max(vals), "values_ms": vals}

        agg["results"][tk][cfg] = {
            "n_params": n_params,
            "total_pipeline": summarize(totals),
            "filterbank": summarize(fb),
            "model_forward": summarize(model),
        }

peak_rss = [r["peak_rss_mb"] for r in runs]
agg["peak_rss_mb"] = {"mean": stats.mean(peak_rss), "min": min(peak_rss), "max": max(peak_rss)}

with open(OUT_JSON, "w") as f:
    json.dump(agg, f, indent=2)

print(f"Aggregated {len(runs)} runs -> {OUT_JSON}\n")
for tk in thread_keys:
    print(f"=== {tk} ===")
    for cfg in configs:
        r = agg["results"][tk][cfg]["total_pipeline"]
        print(f"  {cfg:14s} (params={agg['results'][tk][cfg]['n_params']}): "
              f"{r['mean_ms']:.3f} +/- {r['ci95_ms']:.3f} ms (95% CI, n=7)  "
              f"[std={r['std_ms']:.3f}, range {r['min_ms']:.3f}-{r['max_ms']:.3f}]")
print(f"\nPeak RSS: {agg['peak_rss_mb']['mean']:.2f} MB "
      f"(range {agg['peak_rss_mb']['min']:.2f}-{agg['peak_rss_mb']['max']:.2f})")

"""H3: gop N lan chay doc lap (moi lan 1 tien trinh, tu h3_latency_bench.py)
thanh thong ke lien-tien-trinh (mean/std/CI95 cua model_forward va
total_pipeline_mean_ms), cho moi (backbone, do rong, so thread).

Usage:
    python h3_aggregate.py --pattern "h3_runs/run_*.json" --out h3_summary.json
"""
import argparse
import glob
import json
import statistics as stats
import sys

T_CRIT = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571,
          7: 2.447, 8: 2.365, 9: 2.306, 10: 2.262}


def agg(values):
    n = len(values)
    mean = stats.mean(values)
    sd = stats.stdev(values) if n > 1 else 0.0
    se = sd / (n ** 0.5) if n > 1 else 0.0
    t = T_CRIT.get(n, 1.96)
    return {"mean_ms": mean, "std_ms": sd, "n_runs": n,
            "ci95_ms": [mean - t * se, mean + t * se], "per_run_mean_ms": values}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    files = sorted(glob.glob(args.pattern))
    if not files:
        print(f"Khong tim thay file khop '{args.pattern}'")
        sys.exit(1)
    runs = [json.load(open(f)) for f in files]

    thread_keys = list(runs[0]["configs"].keys())
    cell_keys = list(runs[0]["configs"][thread_keys[0]].keys())

    summary = {
        "n_runs": len(files), "source_files": files,
        "device": runs[0]["device"], "torch": runs[0]["torch"],
        "window_s": runs[0]["window_s"], "n_channels": runs[0]["n_channels"],
        "configs": {},
    }
    for tk in thread_keys:
        summary["configs"][tk] = {}
        for ck in cell_keys:
            mf_vals = [r["configs"][tk][ck]["model_forward"]["mean_ms"] for r in runs]
            tot_vals = [r["configs"][tk][ck]["total_pipeline_mean_ms"] for r in runs]
            summary["configs"][tk][ck] = {
                "n_params": runs[0]["configs"][tk][ck]["n_params"],
                "model_forward": agg(mf_vals),
                "total_pipeline": agg(tot_vals),
            }
        print(f"\n=== {tk} ===")
        for ck in cell_keys:
            a = summary["configs"][tk][ck]["model_forward"]
            print(f"  {ck:24s} model_forward={a['mean_ms']:.4f}±{a['std_ms']:.4f}ms "
                  f"CI95=[{a['ci95_ms'][0]:.4f},{a['ci95_ms'][1]:.4f}]")

    rss_vals = [r.get("peak_rss_mb") for r in runs if r.get("peak_rss_mb") is not None]
    if rss_vals:
        summary["peak_rss_mb"] = agg(rss_vals)

    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()

"""Generate RQ1 forest plot, RQ2a scatter, RQ3 slope chart for the IJRCS paper.
Journal figure style (locked in bo-nao-claude/memory.md, 2026-10-31):
serif font matching body text, dark journal colors used with point/line shape
coding (not color alone), black labels, light square-cornered background,
built at print size, no rescaling on insert.
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Nimbus Roman", "Times New Roman", "DejaVu Serif"],
    "font.size": 10,
    "axes.edgecolor": "black",
    "axes.labelcolor": "black",
    "text.color": "black",
    "xtick.color": "black",
    "ytick.color": "black",
    "axes.facecolor": "#FAFAFA",
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
})

BLUE = "#2166AC"
RED = "#B2182B"
GREEN = "#1B7837"

ROOT = Path(__file__).resolve().parents[3] / "results" / "module3" / "analysis"
OUT = Path(__file__).resolve().parent

BACKBONE_NAME = {"fbtcnn": "FB-tCNN", "deepconvnet": "DeepConvNet", "eegnet": "EEGNet",
                 "shallowconvnet": "ShallowConvNet", "compactcnn": "Compact-CNN"}
DATASET_NAME = {"benchmark": "Benchmark", "beta": "BETA"}


def read_csv(name):
    with open(ROOT / name) as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------------------
# Figure 2: RQ1 forest plot
# ---------------------------------------------------------------------------
def fig_rq1():
    rows = read_csv("rq1_backbone_effect_lsp100.csv")
    backbones = ["fbtcnn", "deepconvnet", "eegnet", "shallowconvnet", "compactcnn"]
    settings = [("benchmark", "3C"), ("benchmark", "1C"), ("beta", "3C"), ("beta", "1C")]

    t_crit = stats.t.ppf(0.975, df=4)  # n=5 splits
    labels, means, los, his, colors, markers = [], [], [], [], [], []
    for bb in backbones:
        for ds, st in settings:
            r = next(x for x in rows if x["backbone"] == bb and x["dataset"] == ds and x["student"] == st)
            mean = float(r["mean_delta_pp"]); sd = float(r["sd_pp"])
            se = sd / np.sqrt(5)
            ci = t_crit * se
            p_t = float(r["p_t"]); p_holm = float(r["p_holm"])
            labels.append(f"{BACKBONE_NAME[bb]} – {DATASET_NAME[ds]} {st}")
            means.append(mean); los.append(mean - ci); his.append(mean + ci)
            if p_holm < 0.05:
                colors.append(RED); markers.append("D")
            elif p_t < 0.05:
                colors.append(BLUE if mean > 0 else RED); markers.append("s")
            else:
                colors.append("0.35"); markers.append("o")

    y = np.arange(len(labels))[::-1]
    fig, ax = plt.subplots(figsize=(6.3, 6.6))
    ax.axvline(0, color="black", linewidth=0.8, linestyle="-")
    for yi, m, lo, hi, c, mk in zip(y, means, los, his, colors, markers):
        ax.plot([lo, hi], [yi, yi], color=c, linewidth=1.3, zorder=2)
        ax.scatter([m], [yi], color=c, marker=mk, s=38, zorder=3, edgecolor="black", linewidth=0.4)
    for i in (4, 8, 12, 16):
        ax.axhline(y[i] + 0.5, color="0.75", linewidth=0.6, linestyle=":")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.set_xlabel(r"Mean $\Delta$ accuracy over 5 splits (pp)")
    ax.set_xlim(-2.2, 2.2)
    ax.tick_params(axis="both", which="both", direction="out")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    from matplotlib.lines import Line2D
    legend_elems = [
        Line2D([0], [0], marker="D", color=RED, label=r"survives Holm ($p_{\mathrm{Holm}}<0.05$)",
               markerfacecolor=RED, markeredgecolor="black", markeredgewidth=0.4, linestyle="None", markersize=6),
        Line2D([0], [0], marker="s", color=BLUE, label=r"raw $p<0.05$, improvement",
               markerfacecolor=BLUE, markeredgecolor="black", markeredgewidth=0.4, linestyle="None", markersize=6),
        Line2D([0], [0], marker="s", color=RED, label=r"raw $p<0.05$, degradation",
               markerfacecolor=RED, markeredgecolor="black", markeredgewidth=0.4, linestyle="None", markersize=6),
        Line2D([0], [0], marker="o", color="0.35", label="not significant",
               markerfacecolor="0.35", markeredgecolor="black", markeredgewidth=0.4, linestyle="None", markersize=6),
    ]
    ax.legend(handles=legend_elems, loc="lower right", fontsize=7.5, frameon=True,
              edgecolor="black", facecolor="white")
    fig.tight_layout()
    fig.savefig(OUT / "rq1_forest.pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3: RQ2a scatter, delta CKA vs delta accuracy
# ---------------------------------------------------------------------------
def fig_rq2a():
    rows = read_csv("rq2a_cka_correlation.csv")
    dcka = np.array([float(r["mean_delta_cka_last"]) for r in rows])
    dacc = np.array([float(r["mean_delta_acc_pp"]) for r in rows])
    datasets = [r["dataset"] for r in rows]

    r_val, p_val = stats.pearsonr(dcka, dacc)
    slope, intercept, *_ = stats.linregress(dcka, dacc)
    xs = np.linspace(dcka.min() - 0.01, dcka.max() + 0.01, 50)

    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    ax.plot(xs, slope * xs + intercept, color="0.3", linewidth=1.1, linestyle="--", zorder=1)
    for ds, mk, col in [("benchmark", "o", BLUE), ("beta", "^", GREEN)]:
        mask = [d == ds for d in datasets]
        ax.scatter(dcka[mask], dacc[mask], marker=mk, s=42, facecolor=col, edgecolor="black",
                   linewidth=0.5, label=DATASET_NAME[ds], zorder=2)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.axvline(0, color="black", linewidth=0.6)
    ax.set_xlabel(r"$\Delta$CKA (last layer, CCKD $-$ baseline)")
    ax.set_ylabel(r"Mean $\Delta$ accuracy (pp)")
    ax.text(0.03, 0.96, rf"Pearson $r={r_val:.2f}$, $p={p_val:.3f}$", transform=ax.transAxes,
            fontsize=9, va="top", ha="left", color="black")
    ax.legend(loc="lower right", fontsize=8.5, frameon=True, edgecolor="black", facecolor="white")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "rq2a_scatter.pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 4: RQ3 slope chart, single-split vs 5-split
# ---------------------------------------------------------------------------
def fig_rq3():
    rows = read_csv("rq3_single_vs_multisplit.csv")
    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    labels = []
    for i, r in enumerate(rows):
        single = float(r["single_split_delta_pp"])
        multi = float(r["multi_split_mean_delta_pp"])
        mismatch = r["any_mismatch"] == "True"
        col = RED if mismatch else "0.4"
        lw = 1.4 if mismatch else 0.8
        ax.plot([0, 1], [single, multi], color=col, linewidth=lw, zorder=2,
                marker="o", markersize=4, markerfacecolor=col, markeredgecolor="black", markeredgewidth=0.3)
        labels.append(f"{BACKBONE_NAME[r['backbone']]} {DATASET_NAME[r['dataset']]} {r['student']}")
    ax.axhline(0, color="black", linewidth=0.7, zorder=1)
    ax.set_xlim(-0.15, 1.15)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Split 2026 only\n(Wilcoxon)", "5-split mean\n(one-sample $t$)"])
    ax.set_ylabel(r"$\Delta$ accuracy (pp)")
    from matplotlib.lines import Line2D
    legend_elems = [
        Line2D([0], [0], color=RED, marker="o", markerfacecolor=RED, markeredgecolor="black",
               label="sign or significance differs", linewidth=1.4, markersize=5),
        Line2D([0], [0], color="0.4", marker="o", markerfacecolor="0.4", markeredgecolor="black",
               label="same verdict", linewidth=0.8, markersize=5),
    ]
    ax.legend(handles=legend_elems, loc="upper left", fontsize=8.5, frameon=True,
              edgecolor="black", facecolor="white")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "rq3_slope.pdf")
    plt.close(fig)


if __name__ == "__main__":
    fig_rq1()
    fig_rq2a()
    fig_rq3()
    print("Wrote:", list(OUT.glob("*.pdf")))

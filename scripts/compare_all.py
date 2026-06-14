"""
scripts/compare_all.py
──────────────────────
Generates all 5 paper figures from algorithm metrics CSVs.

Expected CSV locations:
    results/boids/            metrics_easy.csv  metrics_medium.csv  metrics_hard.csv
    results/mappo/            metrics_easy.csv  metrics_medium.csv  metrics_hard.csv
    results/qmix/             metrics_easy.csv  metrics_medium.csv  metrics_hard.csv
    results/potential_fields/ metrics_easy.csv  metrics_medium.csv  metrics_hard.csv
    results/random_walk/      metrics_easy.csv  metrics_medium.csv  metrics_hard.csv

Usage:
    python scripts/compare_all.py
    python scripts/compare_all.py --skip-missing   # generate partial figures while waiting for teammates
    python scripts/compare_all.py --out figures/
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt

matplotlib.rcParams.update({
    "figure.dpi":         150,
    "savefig.dpi":        600,
    "font.family":        "serif",
    "font.serif":         ["Times New Roman", "DejaVu Serif", "serif"],
    "axes.titlesize":     14,
    "axes.titleweight":   "bold",
    "axes.labelsize":     16,
    "axes.labelweight":   "bold",
    "xtick.labelsize":    14,
    "ytick.labelsize":    14,
    "legend.fontsize":    14,
    "figure.titlesize":   14,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.alpha":         0.25,
    "grid.linestyle":     "--",
    "grid.linewidth":     0.8,
    "axes.linewidth":     1.2,
    "xtick.major.width":  1.2,
    "ytick.major.width":  1.2,
})
ALGORITHMS = ["boids", "mappo", "qmix", "potential_fields", "random_walk"]
SCENARIOS  = ["easy", "medium", "hard"]

PALETTE = {
    "boids":            "#2196F3",
    "mappo":            "#4CAF50",
    "qmix":             "#FF9800",
    "potential_fields": "#9C27B0",
    "random_walk":      "#F44336",
}

LABELS = {
    "boids":            "Boids",
    "mappo":            "MAPPO",
    "qmix":             "QMIX",
    "potential_fields": "Potential Fields",
    "random_walk":      "Random Walk",
}

def _bold_ax(ax):
    for l in ax.get_xticklabels() + ax.get_yticklabels():
        l.set_fontweight('bold')
    leg = ax.get_legend()
    if leg:
        for t in leg.get_texts():
            t.set_fontweight('bold')


def load_csv(algo, scenario, base="results"):
    path = os.path.join(base, algo, f"metrics_{scenario}.csv")
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
        expected = {"episode", "coverage_pct", "detection_rate",
                    "total_collisions", "steps_to_first_target",
                    "steps_to_90pct_coverage"}
        missing = expected - set(df.columns)
        if missing:
            warnings.warn(f"[{algo}/{scenario}] missing columns: {missing}")
        return df
    except Exception as exc:
        warnings.warn(f"Could not load {path}: {exc}")
        return None


def load_all(base="results", skip_missing=False):
    data = {}
    for algo in ALGORITHMS:
        data[algo] = {}
        for scen in SCENARIOS:
            df = load_csv(algo, scen, base)
            if df is None and not skip_missing:
                print(f"  [MISSING] results/{algo}/metrics_{scen}.csv")
            data[algo][scen] = df
    return data


def _grouped_bar(ax, matrix, algo_labels, group_labels, colors,
                 err_matrix=None, ylabel="", title=""):
    n_algos, n_groups = matrix.shape
    x = np.arange(n_groups)
    width = 0.8 / n_algos
    offsets = np.linspace(-(0.8 - width) / 2, (0.8 - width) / 2, n_algos)
    for i, (algo, offset) in enumerate(zip(algo_labels, offsets)):
        err = err_matrix[i] if err_matrix is not None else None
        ax.bar(x + offset, matrix[i], width,
               color=colors[i], alpha=0.85, label=LABELS[algo],
               yerr=err, capsize=3, error_kw={"elinewidth": 1.2})
    ax.set_xticks(x)
    ax.set_xticklabels([g.capitalize() for g in group_labels])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="upper right", framealpha=0.7)


def fig1_coverage(data, out_dir):
    available = [a for a in ALGORITHMS if any(data[a][s] is not None for s in SCENARIOS)]
    if not available:
        print("  [SKIP] fig1 -- no data"); return
    means = np.zeros((len(available), len(SCENARIOS)))
    stds  = np.zeros_like(means)
    for i, algo in enumerate(available):
        for j, scen in enumerate(SCENARIOS):
            df = data[algo][scen]
            if df is not None:
                means[i, j] = df["coverage_pct"].mean()
                stds[i, j]  = df["coverage_pct"].std()
    fig, ax = plt.subplots(figsize=(8, 5))
    _grouped_bar(ax, means, available, SCENARIOS, [PALETTE[a] for a in available],
                 stds, ylabel="Mean coverage (%)",
                 title="Figure 1 -- Area Coverage by Scenario")
    ax.set_ylim(0, 105)
    _bold_ax(ax)
    fig.tight_layout()
    fig.tight_layout()
    path = os.path.join(out_dir, "fig1_coverage_by_scenario.png")
    fig.savefig(path); plt.close(fig)
    print(f"  Saved {path}")


def fig2_detection(data, out_dir):
    available = [a for a in ALGORITHMS if any(data[a][s] is not None for s in SCENARIOS)]
    if not available:
        print("  [SKIP] fig2 -- no data"); return
    means = np.zeros((len(available), len(SCENARIOS)))
    stds  = np.zeros_like(means)
    for i, algo in enumerate(available):
        for j, scen in enumerate(SCENARIOS):
            df = data[algo][scen]
            if df is not None:
                means[i, j] = df["detection_rate"].mean()
                stds[i, j]  = df["detection_rate"].std()
    fig, ax = plt.subplots(figsize=(8, 5))
    _grouped_bar(ax, means, available, SCENARIOS, [PALETTE[a] for a in available],
                 stds, ylabel="Mean detection rate (0-1)",
                 title="Figure 2 -- Target Detection Rate by Scenario")
    ax.set_ylim(0, 1.1)
    _bold_ax(ax)
    fig.tight_layout()
    path = os.path.join(out_dir, "fig2_detection_rate.png")
    fig.savefig(path); plt.close(fig)
    print(f"  Saved {path}")


def fig3_collisions(data, out_dir, scenario="medium"):
    available = [a for a in ALGORITHMS if data[a][scenario] is not None]
    if not available:
        print(f"  [SKIP] fig3 -- no {scenario} data"); return
    means  = [data[a][scenario]["total_collisions"].mean() for a in available]
    stds   = [data[a][scenario]["total_collisions"].std()  for a in available]
    colors = [PALETTE[a] for a in available]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(available))
    ax.bar(x, means, color=colors, alpha=0.85, yerr=stds, capsize=4,
           error_kw={"elinewidth": 1.2})
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[a] for a in available], rotation=15, ha="right")
    ax.set_ylabel("Mean total collisions per episode")
    ax.set_title(f"Figure 3 -- Collisions ({scenario.capitalize()} scenario)")
    _bold_ax(ax)
    fig.tight_layout()
    path = os.path.join(out_dir, "fig3_collisions.png")
    fig.savefig(path); plt.close(fig)
    print(f"  Saved {path}")


def fig4_steps_first_target(data, out_dir, scenario="medium"):
    available = [a for a in ALGORITHMS if data[a][scenario] is not None]
    if not available:
        print(f"  [SKIP] fig4 -- no {scenario} data"); return
    plot_data, plot_labels = [], []
    for algo in available:
        df = data[algo][scenario]
        vals = df["steps_to_first_target"]
        vals = vals[vals > 0].values
        plot_data.append(vals if len(vals) > 0 else np.array([np.nan]))
        plot_labels.append(LABELS[algo])
    fig, ax = plt.subplots(figsize=(7, 5))
    bp = ax.boxplot(plot_data, patch_artist=True, notch=False,
                    medianprops={"color": "black", "linewidth": 2})
    for patch, algo in zip(bp["boxes"], available):
        patch.set_facecolor(PALETTE[algo]); patch.set_alpha(0.75)
    ax.set_xticklabels(plot_labels, rotation=15, ha="right")
    ax.set_ylabel("Steps to first target detection")
    ax.set_title(f"Figure 4 -- Steps to First Target ({scenario.capitalize()} scenario)")
    ax.text(0.98, 0.02, "Lower = better", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=8, color="grey")
    _bold_ax(ax)
    fig.tight_layout()
    path = os.path.join(out_dir, "fig4_steps_to_first_target.png")
    fig.savefig(path); plt.close(fig)
    print(f"  Saved {path}")


def fig5_steps_90pct(data, out_dir, scenario="medium"):
    available = [a for a in ALGORITHMS if data[a][scenario] is not None]
    if not available:
        print(f"  [SKIP] fig5 -- no {scenario} data"); return
    plot_data, plot_labels, dnf_counts = [], [], {}
    for algo in available:
        df = data[algo][scenario]
        all_vals = df["steps_to_90pct_coverage"]
        vals = all_vals[all_vals > 0].values
        dnf_counts[algo] = len(all_vals) - len(vals)
        plot_data.append(vals if len(vals) > 0 else np.array([np.nan]))
        plot_labels.append(LABELS[algo])
    fig, ax = plt.subplots(figsize=(7, 5))
    bp = ax.boxplot(plot_data, patch_artist=True, notch=False,
                    medianprops={"color": "black", "linewidth": 2})
    for patch, algo in zip(bp["boxes"], available):
        patch.set_facecolor(PALETTE[algo]); patch.set_alpha(0.75)
    for idx, algo in enumerate(available, start=1):
        if dnf_counts[algo] > 0:
            ax.text(idx, ax.get_ylim()[1] * 0.97, f"DNF={dnf_counts[algo]}",
                    ha="center", va="top", fontsize=7.5, color="grey")
    ax.set_xticklabels(plot_labels, rotation=15, ha="right")
    ax.set_ylabel("Steps to reach 90% coverage")
    ax.set_title(f"Figure 5 -- Steps to 90% Coverage ({scenario.capitalize()} scenario)")
    ax.text(0.98, 0.02, "Lower = better  |  DNF = never reached 90%",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color="grey")
    _bold_ax(ax)
    fig.tight_layout()
    path = os.path.join(out_dir, "fig5_steps_to_90pct_coverage.png")
    fig.savefig(path); plt.close(fig)
    print(f"  Saved {path}")


def print_summary(data):
    SEP = "-" * 80
    print(f"\n{SEP}")
    print(f"{'ALGORITHM':<20} {'SCENARIO':<8} {'DET':>6} {'COV%':>7} {'COL':>7} {'S1T':>7} {'S90':>7}")
    print(SEP)
    for algo in ALGORITHMS:
        for scen in SCENARIOS:
            df = data[algo][scen]
            if df is None:
                print(f"  {LABELS[algo]:<18} {scen:<8}  {'--':>6} {'--':>7} {'--':>7} {'--':>7} {'--':>7}")
                continue
            det = df["detection_rate"].mean()
            cov = df["coverage_pct"].mean()
            col = df["total_collisions"].mean()
            s1t_v = df["steps_to_first_target"];   s1t = s1t_v[s1t_v>0].mean() if (s1t_v>0).any() else -1
            s90_v = df["steps_to_90pct_coverage"]; s90 = s90_v[s90_v>0].mean() if (s90_v>0).any() else -1
            print(f"  {LABELS[algo]:<18} {scen:<8} {det:>6.2f} {cov:>7.1f} {col:>7.1f} "
                  f"  {str(round(s1t)) if s1t>0 else 'DNF':>7} {str(round(s90)) if s90>0 else 'DNF':>7}")
        print()
    print(SEP)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="results")
    p.add_argument("--out", default="figures")
    p.add_argument("--skip-missing", action="store_true")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    print(f"\nLoading CSVs from '{args.results}/' ...")
    data = load_all(base=args.results, skip_missing=args.skip_missing)
    print_summary(data)
    print(f"\nGenerating figures -> '{args.out}/' ...")
    fig1_coverage(data, args.out)
    fig2_detection(data, args.out)
    fig3_collisions(data, args.out)
    fig4_steps_first_target(data, args.out)
    fig5_steps_90pct(data, args.out)
    print("\nDone. All figures saved at 300 DPI.")


if __name__ == "__main__":
    main()
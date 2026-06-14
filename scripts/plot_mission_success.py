# scripts/plot_mission_success.py
import sys, os
sys.path.insert(0, '.')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams.update({
    'font.family':      'serif',
    'font.serif':       ['Times New Roman', 'DejaVu Serif', 'serif'],
    'axes.labelsize':   16, 'axes.labelweight': 'bold',
    'xtick.labelsize':  14, 'ytick.labelsize':  14,
    'legend.fontsize':  14, 'axes.titlesize':   14,
    'axes.spines.top':  False, 'axes.spines.right': False,
    'axes.grid':        True, 'grid.alpha': 0.25,
    'grid.linestyle':   '--',
})

def bold_ticks(ax):
    for l in ax.get_xticklabels() + ax.get_yticklabels():
        l.set_fontweight('bold')

PALETTE = {
    'mappo':            '#2ECC71',
    'qmix':             '#E67E22',
    'random_walk':      '#E74C3C',
    'boids':            '#3498DB',
    'potential_fields': '#9B59B6',
}
LABELS = {
    'mappo':            'MAPPO',
    'qmix':             'QMIX',
    'random_walk':      'Random Walk',
    'boids':            'Boids',
    'potential_fields': 'Potential Fields',
}

algos     = ['mappo', 'qmix', 'random_walk', 'boids', 'potential_fields']
scenarios = ['easy', 'medium', 'hard']
thresholds = [0.5, 0.75, 1.0]
thresh_labels = ['S50 (≥50%)', 'S75 (≥75%)', 'S100 (100%)']

# Load data
data = {}
for algo in algos:
    data[algo] = {}
    for scen in scenarios:
        path = f'results/{algo}/metrics_{scen}.csv'
        if os.path.exists(path):
            df = pd.read_csv(path)
            data[algo][scen] = [
                (df['detection_rate'] >= t).mean() * 100
                for t in thresholds
            ]
        else:
            data[algo][scen] = [0, 0, 0]

# ── Figure: 3 subplots (one per scenario) ────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 6), sharey=False)
fig.suptitle('Mission Success Rate Across Algorithms and Scenarios\n'
             '(S50: ≥50% targets detected, S75: ≥75%, S100: all targets)',
             fontsize=14, fontweight='bold', y=1.03)

x      = np.arange(len(thresh_labels))
width  = 0.15
offset = np.linspace(-(len(algos)-1)/2 * width,
                      (len(algos)-1)/2 * width, len(algos))

for ax, scen in zip(axes, scenarios):
    for i, algo in enumerate(algos):
        vals = data[algo][scen]
        bars = ax.bar(x + offset[i], vals, width,
                      color=PALETTE[algo], alpha=0.85,
                      label=LABELS[algo], edgecolor='white',
                      linewidth=0.8)
        # Annotate non-zero bars
        for bar, v in zip(bars, vals):
            if v >= 5:
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + 1.5,
                        f'{v:.0f}%', ha='center', va='bottom',
                        fontsize=9, fontweight='bold',
                        color=PALETTE[algo])

    ax.set_xticks(x)
    ax.set_xticklabels(thresh_labels, fontsize=13, fontweight='bold')
    ax.set_xlabel('Success Threshold', fontweight='bold')
    ax.set_ylabel('Mission Success Rate (%)', fontweight='bold')
    ax.set_title(f'{scen.capitalize()} Scenario',
                 fontsize=14, fontweight='bold', pad=8)
    ax.set_ylim(0, 105)
    ax.yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda v, _: f'{v:.0f}%'))
    bold_ticks(ax)

# Single legend
handles = [plt.Rectangle((0,0),1,1, color=PALETTE[a], alpha=0.85)
           for a in algos]
lbls    = [LABELS[a] for a in algos]
fig.legend(handles, lbls, loc='lower center', ncol=5,
           fontsize=14, framealpha=0.9,
           bbox_to_anchor=(0.5, -0.08),
           prop={'weight': 'bold', 'size': 14})

fig.tight_layout()
os.makedirs('figures', exist_ok=True)
path = 'figures/fig10_mission_success.png'
fig.savefig(path, dpi=600, bbox_inches='tight')
plt.close(fig)
print(f"Saved {path}")
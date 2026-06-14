# scripts/plot_performance_metrics.py
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

def load(algo, scen, col, positive_only=False):
    path = f'results/{algo}/metrics_{scen}.csv'
    if not os.path.exists(path):
        return None, None
    df = pd.read_csv(path)
    if col not in df.columns:
        return None, None
    vals = df[col]
    if positive_only:
        vals = vals[vals > 0]
    if len(vals) == 0:
        return None, None
    return float(vals.mean()), float(vals.std())

# ── 4-panel figure ────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(15, 11))
fig.suptitle('Extended Performance Metrics — All Algorithms Across Scenarios',
             fontsize=15, fontweight='bold', y=1.01)

metrics = [
    ('steps_to_first_target',  'Steps to First Detection',     True,  axes[0,0]),
    ('time_to_detect_all',     'Steps to Detect All Targets',  True,  axes[0,1]),
    ('episode_length',         'Mean Episode Length (steps)',  False, axes[1,0]),
    ('total_path_length',      'Total Path Length (units)',    False, axes[1,1]),
]

x        = np.arange(len(scenarios))
width    = 0.15
offsets  = np.linspace(-(len(algos)-1)/2*width,
                        (len(algos)-1)/2*width, len(algos))

for col, ylabel, pos_only, ax in metrics:
    for i, algo in enumerate(algos):
        means, stds = [], []
        for scen in scenarios:
            m, s = load(algo, scen, col, pos_only)
            means.append(m if m is not None else 0)
            stds.append(s if s is not None else 0)

        bars = ax.bar(x + offsets[i], means, width,
                      color=PALETTE[algo], alpha=0.85,
                      label=LABELS[algo], edgecolor='white',
                      linewidth=0.8,
                      yerr=stds, capsize=3,
                      error_kw={'elinewidth': 1.2,
                                'ecolor': PALETTE[algo],
                                'alpha': 0.7})

    ax.set_xticks(x)
    ax.set_xticklabels([s.capitalize() for s in scenarios],
                       fontweight='bold')
    ax.set_xlabel('Scenario', fontweight='bold')
    ax.set_ylabel(ylabel, fontweight='bold')
    ax.set_title(ylabel, fontsize=14, fontweight='bold', pad=8)
    bold_ticks(ax)

    if pos_only:
        ax.text(0.98, 0.97, 'Lower = Better',
                transform=ax.transAxes, ha='right', va='top',
                fontsize=11, color='grey', fontstyle='italic',
                fontweight='bold')

# Shared legend
handles = [plt.Rectangle((0,0),1,1, color=PALETTE[a], alpha=0.85)
           for a in algos]
lbls    = [LABELS[a] for a in algos]
fig.legend(handles, lbls, loc='lower center', ncol=5,
           fontsize=14, framealpha=0.9,
           bbox_to_anchor=(0.5, -0.04),
           prop={'weight': 'bold', 'size': 14})

fig.tight_layout()
os.makedirs('figures', exist_ok=True)
path = 'figures/fig13_performance_metrics.png'
fig.savefig(path, dpi=600, bbox_inches='tight')
plt.close(fig)
print(f"Saved {path}")
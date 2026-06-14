# scripts/plot_radar.py
import sys, os
sys.path.insert(0, '.')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams.update({
    'font.family':    'serif',
    'font.serif':     ['Times New Roman', 'DejaVu Serif', 'serif'],
    'legend.fontsize': 14,
})

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

# Compute per-algo averages across scenarios
def load_avg(algo, metric, invert=False):
    vals = []
    for scen in scenarios:
        path = f'results/{algo}/metrics_{scen}.csv'
        if os.path.exists(path):
            df = pd.read_csv(path)
            vals.append(df[metric].mean())
    if not vals:
        return 0.0
    v = np.mean(vals)
    return v

# 5 metrics — all normalised 0-1 (higher = better)
raw = {}
for algo in algos:
    cov  = load_avg(algo, 'coverage_pct')   / 100.0
    det  = load_avg(algo, 'detection_rate')
    col  = load_avg(algo, 'total_collisions')
    s1t  = load_avg(algo, 'steps_to_first_target')

    raw[algo] = {
        'coverage':    cov,
        'detection':   det,
        'collision_avoid': 1.0,   # placeholder, normalise below
        'speed':           1.0,   # placeholder
        'mission_success': 0.0,   # placeholder
    }

# Collision avoidance: lower collisions = better; normalise inverted
max_col = max(load_avg(a, 'total_collisions') for a in algos)
for algo in algos:
    col = load_avg(algo, 'total_collisions')
    raw[algo]['collision_avoid'] = 1.0 - (col / (max_col + 1e-6))

# Speed (steps to first target): lower = better
s1t_vals = {}
for algo in algos:
    vals = []
    for scen in scenarios:
        path = f'results/{algo}/metrics_{scen}.csv'
        if os.path.exists(path):
            df = pd.read_csv(path)
            v = df[df['steps_to_first_target'] > 0]['steps_to_first_target']
            if len(v):
                vals.append(v.mean())
    s1t_vals[algo] = np.mean(vals) if vals else 500

max_s1t = max(s1t_vals.values())
for algo in algos:
    raw[algo]['speed'] = 1.0 - (s1t_vals[algo] / (max_s1t + 1e-6))

# Mission success (S50 avg across scenarios)
for algo in algos:
    vals = []
    for scen in scenarios:
        path = f'results/{algo}/metrics_{scen}.csv'
        if os.path.exists(path):
            df = pd.read_csv(path)
            vals.append((df['detection_rate'] >= 0.5).mean())
    raw[algo]['mission_success'] = np.mean(vals) if vals else 0.0

# Radar setup
categories  = ['Coverage', 'Detection\nRate', 'Collision\nAvoidance',
               'Detection\nSpeed', 'Mission\nSuccess']
N = len(categories)
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]

fig, ax = plt.subplots(figsize=(9, 9),
                        subplot_kw=dict(polar=True))

for algo in algos:
    vals = [
        raw[algo]['coverage'],
        raw[algo]['detection'],
        raw[algo]['collision_avoid'],
        raw[algo]['speed'],
        raw[algo]['mission_success'],
    ]
    vals += vals[:1]
    ax.plot(angles, vals, 'o-', linewidth=2.5,
            color=PALETTE[algo], label=LABELS[algo], markersize=7)
    ax.fill(angles, vals, alpha=0.08, color=PALETTE[algo])

# Style
ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=14, fontweight='bold')
ax.set_ylim(0, 1)
ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'],
                    fontsize=11, fontweight='bold')
ax.grid(color='grey', linestyle='--', linewidth=0.6, alpha=0.5)
ax.spines['polar'].set_linewidth(1.5)

ax.set_title('Multi-Metric Algorithm Comparison\n'
             'Normalised Performance Across 5 Key Metrics',
             fontsize=14, fontweight='bold', pad=20)

leg = ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.15),
                fontsize=14, framealpha=0.9)
for t in leg.get_texts():
    t.set_fontweight('bold')

plt.tight_layout()
os.makedirs('figures', exist_ok=True)
path = 'figures/fig11_radar_comparison.png'
fig.savefig(path, dpi=600, bbox_inches='tight')
plt.close(fig)
print(f"Saved {path}")
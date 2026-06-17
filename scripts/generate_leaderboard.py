# scripts/generate_leaderboard.py
# M11 — Full benchmark leaderboard across all 12 configs x 5 algorithms
import sys, os
sys.path.insert(0, '.')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from env.benchmark_configs import SWARN_CONFIGS

ALGOS = ['mappo', 'random_walk', 'boids', 'potential_fields']
# qmix added when Yeshita pushes results

METRICS = {
    'coverage_pct':          'Coverage (%)',
    'detection_rate':        'Detection Rate',
    'total_collisions':      'Collisions',
    'steps_to_first_target': 'Steps to 1st Target',
    'episode_length':        'Episode Length',
    'total_path_length':     'Path Length',
}

def load(algo, config_code, metric, positive_only=False):
    path = f'results/{algo}/{config_code}/metrics.csv'
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    if metric not in df.columns:
        return None
    vals = df[metric]
    if positive_only:
        vals = vals[vals > 0]
    return float(vals.mean()) if len(vals) > 0 else None

# ── Build leaderboard matrix ─────────────────────────────────────────────────
rows = []
for config_code in SWARN_CONFIGS:
    for algo in ALGOS:
        row = {'Config': config_code, 'Algorithm': algo.upper().replace('_', ' ')}
        for metric, label in METRICS.items():
            pos_only = metric in ['steps_to_first_target']
            v = load(algo, config_code, metric, pos_only)
            row[label] = round(v, 2) if v is not None else '-'
        rows.append(row)

df_board = pd.DataFrame(rows)
os.makedirs('results', exist_ok=True)
df_board.to_csv('results/leaderboard.csv', index=False)
print("Saved results/leaderboard.csv")
print(df_board.to_string(index=False))

# ── Coverage heatmap ─────────────────────────────────────────────────────────
pivot = pd.pivot_table(
    df_board[df_board['Coverage (%)'] != '-'],
    values='Coverage (%)',
    index='Algorithm',
    columns='Config',
    aggfunc='mean'
)

fig, ax = plt.subplots(figsize=(16, 5))
sns.heatmap(pivot, annot=True, fmt='.1f', cmap='YlGn',
            linewidths=0.5, linecolor='white',
            annot_kws={'size': 11, 'weight': 'bold'},
            ax=ax, cbar_kws={'label': 'Coverage (%)'})
ax.set_title('SWARN Benchmark Leaderboard — Coverage (%) Across All Configs',
             fontsize=14, fontweight='bold', pad=12)
ax.set_xlabel('Benchmark Configuration', fontsize=14, fontweight='bold')
ax.set_ylabel('Algorithm', fontsize=14, fontweight='bold')
for l in ax.get_xticklabels() + ax.get_yticklabels():
    l.set_fontsize(12)
    l.set_fontweight('bold')
plt.tight_layout()
os.makedirs('figures', exist_ok=True)
fig.savefig('figures/fig14_leaderboard_coverage.png', dpi=600, bbox_inches='tight')
plt.close(fig)
print("Saved figures/fig14_leaderboard_coverage.png")

# ── Detection heatmap ─────────────────────────────────────────────────────────
pivot2 = pd.pivot_table(
    df_board[df_board['Detection Rate'] != '-'],
    values='Detection Rate',
    index='Algorithm',
    columns='Config',
    aggfunc='mean'
)

fig, ax = plt.subplots(figsize=(16, 5))
sns.heatmap(pivot2, annot=True, fmt='.2f', cmap='YlOrRd',
            linewidths=0.5, linecolor='white',
            annot_kws={'size': 11, 'weight': 'bold'},
            ax=ax, cbar_kws={'label': 'Detection Rate'})
ax.set_title('SWARN Benchmark Leaderboard — Detection Rate Across All Configs',
             fontsize=14, fontweight='bold', pad=12)
ax.set_xlabel('Benchmark Configuration', fontsize=14, fontweight='bold')
ax.set_ylabel('Algorithm', fontsize=14, fontweight='bold')
for l in ax.get_xticklabels() + ax.get_yticklabels():
    l.set_fontsize(12)
    l.set_fontweight('bold')
plt.tight_layout()
fig.savefig('figures/fig15_leaderboard_detection.png', dpi=600, bbox_inches='tight')
plt.close(fig)
print("Saved figures/fig15_leaderboard_detection.png")
# scripts/plot_scaling_comparison.py
import sys, os
sys.path.insert(0, '.')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

agents = [5, 10, 25, 50]

mappo = {
    'coverage':  [9.3,  18.0, 36.8, 54.8],
    'detection': [0.24, 0.37, 0.64, 0.87],
}
qmix = {
    'coverage':  [3.4,  7.6,  14.9, 27.0],
    'detection': [0.14, 0.25, 0.41, 0.59],
}
rw = {
    'coverage':  [7.0,  11.8, None, None],
    'detection': [0.24, 0.23, None, None],
}

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, metric, ylabel in zip(
    axes,
    ['coverage', 'detection'],
    ['Mean Coverage (%)', 'Mean Detection Rate']
):
    ax.plot(agents, mappo[metric], 'o-', color='#4CAF50',
            linewidth=2.5, markersize=9, label='MAPPO')
    ax.plot(agents, qmix[metric],  's-', color='#FF9800',
            linewidth=2.5, markersize=9, label='QMIX')
    ax.set_xlabel('Number of Agents', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(f'{ylabel} vs Swarm Size', fontsize=12)
    ax.set_xticks(agents)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

fig.suptitle('MAPPO vs QMIX — Scaling Analysis\n(Both trained on 10 agents, zero-shot generalisation)',
             fontsize=12)
fig.tight_layout()
os.makedirs('figures', exist_ok=True)
fig.savefig('figures/fig9_scaling_comparison.png', dpi=300)
plt.close(fig)
print("Saved figures/fig9_scaling_comparison.png")
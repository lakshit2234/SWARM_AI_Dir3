# scripts/dropout_analysis.py
import sys, os
sys.path.insert(0, '.')
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
from algos.mappo.env_wrapper import SwarmParallelEnv
from algos.mappo.networks import Actor

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

DEVICE     = torch.device('cuda')
OBS_DIM    = 29
ACTION_DIM = 2
HIDDEN     = 128
N_EPISODES = 20

actor = Actor(OBS_DIM, ACTION_DIM, HIDDEN).to(DEVICE)
ckpt  = torch.load('results/mappo/ckpt_stage3_sensed_1000000.pt', map_location=DEVICE)
actor.load_state_dict(ckpt['actor'])
actor.eval()
print("Checkpoint loaded!")

dropouts = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7]

results = {'coverage': [], 'detection': [], 'collisions': []}

for dropout in dropouts:
    cfg = {
        'n_agents': 10, 'map_size': 100, 'n_targets': 5,
        'n_obstacles': 15, 'comm_dropout': dropout,
        'max_steps': 500, 'obs_radius': 10, 'coverage_mode': 'sensed',
    }
    env = SwarmParallelEnv(cfg)
    cov_list, det_list, col_list = [], [], []

    print(f"\nDropout={dropout:.1f}")
    for ep in range(1, N_EPISODES + 1):
        obs, _ = env.reset()
        while True:
            obs_tensor = torch.tensor(
                np.stack([obs[a] for a in env.agents]),
                dtype=torch.float32
            ).to(DEVICE)
            with torch.no_grad():
                actions, _ = actor(obs_tensor)
            actions_dict = {a: actions.cpu().numpy()[i]
                           for i, a in enumerate(env.agents)}
            obs, _, dones, _, info = env.step(actions_dict)
            if dones[env.agents[0]]:
                break
        cov_list.append(info['coverage_pct'])
        det_list.append(info['detection_rate'])
        col_list.append(info.get('collisions', 0))
        print(f"  Ep {ep:>2} | Cov: {info['coverage_pct']:.1f}% | Det: {info['detection_rate']:.2f}")

    results['coverage'].append(np.mean(cov_list))
    results['detection'].append(np.mean(det_list))
    results['collisions'].append(np.mean(col_list))

# ── Plot ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle('MAPPO Communication Robustness — Performance vs Dropout Rate',
             fontsize=14, fontweight='bold', y=1.02)

for ax, metric, ylabel, color in [
    (axes[0], 'coverage',  'Mean Coverage (%)',   '#2ECC71'),
    (axes[1], 'detection', 'Mean Detection Rate', '#27AE60'),
]:
    vals = results[metric]
    ax.plot(dropouts, vals, 'o-', color=color, linewidth=2.8,
            markersize=11, markerfacecolor='white',
            markeredgecolor=color, markeredgewidth=2.8, zorder=5)
    ax.fill_between(dropouts,
                    [v * 0.9 for v in vals],
                    [v * 1.1 for v in vals],
                    alpha=0.12, color=color)

    # Annotate each point
    for x, y in zip(dropouts, vals):
        ax.annotate(f'{y:.1f}{"%" if "Cov" in ylabel else ""}',
                    (x, y), textcoords='offset points',
                    xytext=(0, 12), ha='center',
                    fontsize=11, fontweight='bold', color=color)

    # Mark 20% dropout (training condition)
    ax.axvline(x=0.2, color='red', linestyle='--',
               linewidth=1.5, alpha=0.7, label='Training dropout (0.2)')

    ax.set_xlabel('Communication Dropout Rate', fontweight='bold')
    ax.set_ylabel(ylabel, fontweight='bold')
    ax.set_title(ylabel + ' vs Dropout', fontsize=14,
                 fontweight='bold', pad=8)
    ax.set_xticks(dropouts)
    ax.set_xticklabels([f'{d:.1f}' for d in dropouts],
                       fontweight='bold')
    for l in ax.get_yticklabels():
        l.set_fontweight('bold')
    ax.legend(fontsize=12, framealpha=0.9)
    for t in ax.get_legend().get_texts():
        t.set_fontweight('bold')

fig.tight_layout()
os.makedirs('figures', exist_ok=True)
path = 'figures/fig12_dropout_robustness.png'
fig.savefig(path, dpi=600, bbox_inches='tight')
plt.close(fig)
print(f"\nSaved {path}")

print("\nDropout Summary:")
print("Dropout   Coverage   Detection")
print("-" * 35)
for d, c, det in zip(dropouts, results['coverage'], results['detection']):
    print(f"{d:.1f}       {c:.1f}%      {det:.2f}")
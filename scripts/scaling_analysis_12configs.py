# scripts/scaling_analysis_12configs.py
# M15 — Scaling analysis: MAPPO trained on S5 (N=10), eval at N=5,10,25,50,100
import sys, os
sys.path.insert(0, '.')
import numpy as np
import torch
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from env.benchmark_configs import get_config
from algos.mappo.env_wrapper import SwarmParallelEnv
from algos.mappo.networks import Actor

DEVICE     = torch.device('cuda')
OBS_DIM    = 29
ACTION_DIM = 2
HIDDEN     = 128
N_EPISODES = 20

# Use S5 checkpoint (standard trained config N=10)
ckpt_path = 'results/mappo/SWARN-S5/checkpoint.pt'
actor = Actor(OBS_DIM, ACTION_DIM, HIDDEN).to(DEVICE)
ckpt  = torch.load(ckpt_path, map_location=DEVICE)
actor.load_state_dict(ckpt['actor'])
actor.eval()
print("S5 checkpoint loaded (trained on N=10)")

# Fixed map, vary only n_agents
scale_configs = {
    5:   {'n_agents': 5,   'map_size': 100, 'n_targets': 5,
          'n_obstacles': 15, 'comm_dropout': 0.0, 'max_steps': 500,
          'obs_radius': 10, 'coverage_mode': 'sensed', 'detect_radius': 2.0},
    10:  {'n_agents': 10,  'map_size': 100, 'n_targets': 5,
          'n_obstacles': 15, 'comm_dropout': 0.0, 'max_steps': 500,
          'obs_radius': 10, 'coverage_mode': 'sensed', 'detect_radius': 2.0},
    25:  {'n_agents': 25,  'map_size': 100, 'n_targets': 5,
          'n_obstacles': 15, 'comm_dropout': 0.0, 'max_steps': 500,
          'obs_radius': 10, 'coverage_mode': 'sensed', 'detect_radius': 2.0},
    50:  {'n_agents': 50,  'map_size': 100, 'n_targets': 5,
          'n_obstacles': 15, 'comm_dropout': 0.0, 'max_steps': 500,
          'obs_radius': 10, 'coverage_mode': 'sensed', 'detect_radius': 2.0},
    100: {'n_agents': 100, 'map_size': 100, 'n_targets': 5,
          'n_obstacles': 15, 'comm_dropout': 0.0, 'max_steps': 500,
          'obs_radius': 10, 'coverage_mode': 'sensed', 'detect_radius': 2.0},
}

results = {}

for n, cfg in scale_configs.items():
    print(f"\nScaling — {n} agents")
    env = SwarmParallelEnv(cfg)
    cov_list, det_list, col_list, s1t_list = [], [], [], []

    for ep in range(1, N_EPISODES + 1):
        obs, _ = env.reset()
        step = 0
        s1t  = -1
        while True:
            obs_tensor = torch.tensor(
                np.stack([obs[a] for a in env.agents]),
                dtype=torch.float32).to(DEVICE)
            with torch.no_grad():
                actions, _ = actor(obs_tensor)
            actions_dict = {a: actions.cpu().numpy()[i]
                           for i, a in enumerate(env.agents)}
            obs, _, dones, _, info = env.step(actions_dict)
            step += 1
            if info['detection_rate'] > 0 and s1t == -1:
                s1t = step
            if dones[env.agents[0]]:
                break
        cov_list.append(info['coverage_pct'])
        det_list.append(info['detection_rate'])
        col_list.append(info.get('collisions', 0))
        s1t_list.append(s1t)
        print(f"  Ep {ep:>2} | Cov: {info['coverage_pct']:.1f}% | "
              f"Det: {info['detection_rate']:.2f}")

    results[n] = {
        'coverage':   np.mean(cov_list),
        'detection':  np.mean(det_list),
        'collisions': np.mean(col_list),
        's1t':        np.mean([v for v in s1t_list if v > 0]),
    }

# ── Save CSV ──────────────────────────────────────────────────────────────────
df = pd.DataFrame([
    {'n_agents': n, **v} for n, v in results.items()
])
os.makedirs('results', exist_ok=True)
df.to_csv('results/scaling_analysis.csv', index=False)
print("\nSaved results/scaling_analysis.csv")

# ── Plot ──────────────────────────────────────────────────────────────────────
agents = sorted(results.keys())
fig, axes = plt.subplots(1, 4, figsize=(18, 5))
fig.suptitle('MAPPO Scaling Analysis — Trained on N=10, Zero-Shot at N=5,10,25,50,100',
             fontsize=14, fontweight='bold', y=1.02)

metrics = [
    ('coverage',   'Mean Coverage (%)',          '#2ECC71'),
    ('detection',  'Mean Detection Rate',         '#27AE60'),
    ('collisions', 'Mean Collisions / Episode',   '#E74C3C'),
    ('s1t',        'Steps to First Detection',    '#3498DB'),
]

for ax, (metric, ylabel, color) in zip(axes, metrics):
    vals = [results[n][metric] for n in agents]
    ax.plot(agents, vals, 'o-', color=color, linewidth=2.8,
            markersize=11, markerfacecolor='white',
            markeredgecolor=color, markeredgewidth=2.8, zorder=5)
    ax.fill_between(agents, [v*0.9 for v in vals],
                    [v*1.1 for v in vals], alpha=0.1, color=color)
    for x, y in zip(agents, vals):
        ax.annotate(f'{y:.1f}', (x, y),
                    textcoords='offset points', xytext=(0, 12),
                    ha='center', fontsize=10, fontweight='bold', color=color)
    ax.set_xlabel('Number of Agents', fontweight='bold', fontsize=14)
    ax.set_ylabel(ylabel, fontweight='bold', fontsize=14)
    ax.set_xticks(agents)
    for l in ax.get_xticklabels() + ax.get_yticklabels():
        l.set_fontweight('bold')
        l.set_fontsize(12)
    ax.axvline(x=10, color='grey', linestyle='--',
               linewidth=1.5, alpha=0.7, label='Training N')
    ax.legend(fontsize=11)
    for t in ax.get_legend().get_texts():
        t.set_fontweight('bold')

fig.tight_layout()
os.makedirs('figures', exist_ok=True)
fig.savefig('figures/fig17_scaling_12configs.png', dpi=600, bbox_inches='tight')
plt.close(fig)
print("Saved figures/fig17_scaling_12configs.png")

print("\nScaling Summary:")
print(f"{'Agents':<8} {'Coverage':>10} {'Detection':>10} {'Collisions':>12} {'S1T':>8}")
print("-" * 55)
for n in agents:
    r = results[n]
    print(f"{n:<8} {r['coverage']:>9.1f}% {r['detection']:>10.2f} "
          f"{r['collisions']:>12.1f} {r['s1t']:>8.1f}")
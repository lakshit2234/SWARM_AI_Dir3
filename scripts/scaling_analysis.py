# scripts/scaling_analysis.py
import sys, os
sys.path.insert(0, '.')
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from algos.mappo.env_wrapper import SwarmParallelEnv
from algos.mappo.networks import Actor
from utils.metrics import MetricsTracker

DEVICE     = torch.device('cuda')
OBS_DIM    = 29
ACTION_DIM = 2
HIDDEN     = 128
N_EPISODES = 20

# Load best MAPPO checkpoint
actor = Actor(OBS_DIM, ACTION_DIM, HIDDEN).to(DEVICE)
ckpt  = torch.load('results/mappo/ckpt_stage3_sensed_1000000.pt', map_location=DEVICE)
actor.load_state_dict(ckpt['actor'])
actor.eval()
print("Checkpoint loaded!")

# Scaling configs — vary n_agents, keep everything else fixed
scale_configs = {
    5:  {'n_agents': 5,  'map_size': 100, 'n_targets': 5, 'n_obstacles': 15,
         'comm_dropout': 0.0, 'max_steps': 500, 'obs_radius': 10, 'coverage_mode': 'sensed'},
    10: {'n_agents': 10, 'map_size': 100, 'n_targets': 5, 'n_obstacles': 15,
         'comm_dropout': 0.0, 'max_steps': 500, 'obs_radius': 10, 'coverage_mode': 'sensed'},
    25: {'n_agents': 25, 'map_size': 100, 'n_targets': 5, 'n_obstacles': 15,
         'comm_dropout': 0.0, 'max_steps': 500, 'obs_radius': 10, 'coverage_mode': 'sensed'},
    50: {'n_agents': 50, 'map_size': 100, 'n_targets': 5, 'n_obstacles': 15,
         'comm_dropout': 0.0, 'max_steps': 500, 'obs_radius': 10, 'coverage_mode': 'sensed'},
}

results = {}

for n, cfg in scale_configs.items():
    print(f"\nScaling eval — {n} agents")
    env     = SwarmParallelEnv(cfg)
    tracker = MetricsTracker()
    cov_list, det_list, col_list = [], [], []

    for ep in range(1, N_EPISODES + 1):
        obs, _ = env.reset()
        tracker.episode_start()
        step = 0
        while True:
            obs_tensor = torch.tensor(
                np.stack([obs[a] for a in env.agents]),
                dtype=torch.float32
            ).to(DEVICE)
            with torch.no_grad():
                actions, _ = actor(obs_tensor)
            actions_np   = actions.cpu().numpy()
            actions_dict = {a: actions_np[i] for i, a in enumerate(env.agents)}
            obs, rews, dones, _, info = env.step(actions_dict)
            step += 1
            tracker.update(step, info, info.get('collisions', 0))
            if dones[env.agents[0]]:
                break
        tracker.episode_end(ep)
        cov_list.append(info['coverage_pct'])
        det_list.append(info['detection_rate'])
        col_list.append(info.get('collisions', 0))
        print(f"  Ep {ep:>2} | Coverage: {info['coverage_pct']:.1f}% | "
              f"Detected: {info['detection_rate']:.2f}")

    results[n] = {
        'coverage': np.mean(cov_list),
        'detection': np.mean(det_list),
        'collisions': np.mean(col_list),
    }
    os.makedirs('results/mappo/scaling', exist_ok=True)
    tracker.save_csv(f'results/mappo/scaling/scale_{n}_agents.csv')

# Plot
os.makedirs('figures', exist_ok=True)
agents = sorted(results.keys())
covs   = [results[n]['coverage']   for n in agents]
dets   = [results[n]['detection']  for n in agents]
cols   = [results[n]['collisions'] for n in agents]

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, vals, title, ylabel, color in zip(
    axes,
    [covs, dets, cols],
    ['Coverage vs Swarm Size', 'Detection vs Swarm Size', 'Collisions vs Swarm Size'],
    ['Mean Coverage (%)', 'Mean Detection Rate', 'Mean Collisions/Episode'],
    ['#4CAF50', '#2196F3', '#F44336']
):
    ax.plot(agents, vals, 'o-', color=color, linewidth=2, markersize=8)
    ax.set_xlabel('Number of Agents')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(agents)
    ax.grid(True, alpha=0.3)

fig.suptitle('MAPPO Scaling Analysis — Trained on 10 agents, evaluated across swarm sizes',
             fontsize=11)
fig.tight_layout()
fig.savefig('figures/fig6_mappo_scaling.png', dpi=300)
plt.close(fig)
print("\nSaved figures/fig6_mappo_scaling.png")

# Print summary
print("\nScaling Summary:")
print(f"{'Agents':<8} {'Coverage':>10} {'Detection':>10} {'Collisions':>12}")
print("-" * 45)
for n in agents:
    r = results[n]
    print(f"{n:<8} {r['coverage']:>9.1f}% {r['detection']:>10.2f} {r['collisions']:>12.1f}")
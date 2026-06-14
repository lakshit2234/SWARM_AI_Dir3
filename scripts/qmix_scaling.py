# scripts/qmix_scaling.py
import sys, os
sys.path.insert(0, '.')
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from env.swarm_env import SwarmEnv
from utils.metrics import MetricsTracker
from algos.qmix.networks import AgentQNetwork

DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OBS_DIM    = 29
N_ACTIONS  = 9
HIDDEN     = 128
TRAINED_N  = 10
N_EPISODES = 20

ACTION_MAP = np.array([
    [ 0.0,  0.0], [ 0.0,  2.0], [ 0.0, -2.0],
    [-2.0,  0.0], [ 2.0,  0.0], [ 1.4,  1.4],
    [-1.4,  1.4], [ 1.4, -1.4], [-1.4, -1.4],
], dtype=np.float32)

# Load 10 trained agent networks
ckpt_path = 'results/qmix/checkpoints_10agents/qmix_step_500000.pt'
ckpt      = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
base_nets = [AgentQNetwork(OBS_DIM, N_ACTIONS, HIDDEN).to(DEVICE) for _ in range(TRAINED_N)]
for i, net in enumerate(base_nets):
    net.load_state_dict(ckpt['agent_states'][i])
    net.eval()
print(f"Loaded {TRAINED_N} agent networks!")

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
    print(f"\nQMIX scaling — {n} agents")
    env     = SwarmEnv(cfg)
    cov_list, det_list, col_list = [], [], []
    tracker = MetricsTracker()

    for ep in range(1, N_EPISODES + 1):
        obs_array, _ = env.reset()
        tracker.episode_start()
        step = 0
        while True:
            int_actions = np.zeros(n, dtype=np.int64)
            for i in range(n):
                net    = base_nets[i % TRAINED_N]
                obs_t  = torch.FloatTensor(obs_array[i]).unsqueeze(0).to(DEVICE)
                q_vals = net(obs_t)
                int_actions[i] = q_vals.argmax(dim=1).item()
            vel_actions = ACTION_MAP[int_actions]
            obs_array, _, done, _, info = env.step(vel_actions)
            step += 1
            tracker.update(step, info, info.get('collisions', 0))
            if done:
                break
        tracker.episode_end(ep)
        cov_list.append(info['coverage_pct'])
        det_list.append(info['detection_rate'])
        col_list.append(info.get('collisions', 0))
        print(f"  Ep {ep:>2} | Coverage: {info['coverage_pct']:.1f}% | Detected: {info['detection_rate']:.2f}")

    results[n] = {
        'coverage':   np.mean(cov_list),
        'detection':  np.mean(det_list),
        'collisions': np.mean(col_list),
    }
    os.makedirs('results/qmix/scaling', exist_ok=True)
    tracker.save_csv(f'results/qmix/scaling/scale_{n}_agents.csv')

# Plot
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
    ['#FF9800', '#FF9800', '#FF9800']
):
    ax.plot(agents, vals, 'o-', color=color, linewidth=2, markersize=8)
    ax.set_xlabel('Number of Agents')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(agents)
    ax.grid(True, alpha=0.3)

fig.suptitle('QMIX Scaling Analysis — Trained on 10 agents', fontsize=11)
fig.tight_layout()
os.makedirs('figures', exist_ok=True)
fig.savefig('figures/fig8_qmix_scaling.png', dpi=300)
plt.close(fig)
print("\nSaved figures/fig8_qmix_scaling.png")

print("\nQMIX Scaling Summary:")
print(f"{'Agents':<8} {'Coverage':>10} {'Detection':>10} {'Collisions':>12}")
print("-" * 45)
for n in agents:
    r = results[n]
    print(f"{n:<8} {r['coverage']:>9.1f}% {r['detection']:>10.2f} {r['collisions']:>12.1f}")
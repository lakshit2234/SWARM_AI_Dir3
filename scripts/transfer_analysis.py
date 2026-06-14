# scripts/transfer_analysis.py
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

checkpoints = {
    'stage1': 'results/mappo/ckpt_stage1_sensed_500000.pt',
    'stage3': 'results/mappo/ckpt_stage3_sensed_1000000.pt',
}

eval_scenarios = {
    'easy':   {'n_agents': 5,  'map_size': 50,  'n_targets': 3,
               'n_obstacles': 5,  'comm_dropout': 0.0, 'max_steps': 300,
               'obs_radius': 10, 'coverage_mode': 'sensed'},
    'medium': {'n_agents': 10, 'map_size': 100, 'n_targets': 5,
               'n_obstacles': 15, 'comm_dropout': 0.0, 'max_steps': 500,
               'obs_radius': 10, 'coverage_mode': 'sensed'},
    'hard':   {'n_agents': 10, 'map_size': 100, 'n_targets': 8,
               'n_obstacles': 20, 'comm_dropout': 0.2, 'max_steps': 500,
               'obs_radius': 10, 'coverage_mode': 'sensed'},
}

results = {}

for ckpt_name, ckpt_path in checkpoints.items():
    actor = Actor(OBS_DIM, ACTION_DIM, HIDDEN).to(DEVICE)
    ckpt  = torch.load(ckpt_path, map_location=DEVICE)
    actor.load_state_dict(ckpt['actor'])
    actor.eval()
    print(f"\nEvaluating checkpoint: {ckpt_name}")
    results[ckpt_name] = {}

    for scen_name, cfg in eval_scenarios.items():
        env     = SwarmParallelEnv(cfg)
        cov_list, det_list = [], []

        for ep in range(1, N_EPISODES + 1):
            obs, _ = env.reset()
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
                obs, _, dones, _, info = env.step(actions_dict)
                step += 1
                if dones[env.agents[0]]:
                    break
            cov_list.append(info['coverage_pct'])
            det_list.append(info['detection_rate'])

        results[ckpt_name][scen_name] = {
            'coverage':  np.mean(cov_list),
            'detection': np.mean(det_list),
        }
        print(f"  {scen_name:<8} | Cov: {np.mean(cov_list):.1f}% | Det: {np.mean(det_list):.2f}")

# Plot heatmap
os.makedirs('figures', exist_ok=True)
scenarios  = list(eval_scenarios.keys())
ckpt_names = list(checkpoints.keys())

for metric, title, fmt in [('coverage', 'Coverage (%)', '.1f'), ('detection', 'Detection Rate', '.2f')]:
    matrix = np.array([
        [results[c][s][metric] for s in scenarios]
        for c in ckpt_names
    ])
    fig, ax = plt.subplots(figsize=(7, 4))
    im = ax.imshow(matrix, cmap='YlGn', aspect='auto')
    ax.set_xticks(range(len(scenarios)))
    ax.set_yticks(range(len(ckpt_names)))
    ax.set_xticklabels([s.capitalize() for s in scenarios])
    ax.set_yticklabels(['Stage 1\n(Easy trained)', 'Stage 3\n(Hard trained)'])
    ax.set_xlabel('Evaluation Scenario')
    ax.set_title(f'Transfer Analysis — MAPPO {title}')
    for i in range(len(ckpt_names)):
        for j in range(len(scenarios)):
            ax.text(j, i, f'{matrix[i,j]:{fmt}}', ha='center', va='center',
                    fontsize=11, fontweight='bold',
                    color='black' if matrix[i,j] < matrix.max()*0.7 else 'white')
    plt.colorbar(im, ax=ax)
    fig.tight_layout()
    fname = f'figures/fig7_transfer_{metric}.png'
    fig.savefig(fname, dpi=300)
    plt.close(fig)
    print(f"Saved {fname}")
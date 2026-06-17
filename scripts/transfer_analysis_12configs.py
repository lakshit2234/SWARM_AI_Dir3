# scripts/transfer_analysis_12configs.py
# M14 — Transfer analysis: 4 source configs x 12 target configs
import sys, os
sys.path.insert(0, '.')
import numpy as np
import torch
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from env.benchmark_configs import SWARN_CONFIGS, get_config
from algos.mappo.env_wrapper import SwarmParallelEnv
from algos.mappo.networks import Actor

DEVICE     = torch.device('cuda')
OBS_DIM    = 29
ACTION_DIM = 2
HIDDEN     = 128
N_EPISODES = 10

SOURCE_CKPTS = {
    'SWARN-S1':  'results/mappo/SWARN-S1/checkpoint.pt',
    'SWARN-S5':  'results/mappo/SWARN-S5/checkpoint.pt',
    'SWARN-S9':  'results/mappo/SWARN-S9/checkpoint.pt',
    'SWARN-S12': 'results/mappo/SWARN-S10/checkpoint.pt',
}

results = {}

for source, ckpt_path in SOURCE_CKPTS.items():
    actor = Actor(OBS_DIM, ACTION_DIM, HIDDEN).to(DEVICE)
    ckpt  = torch.load(ckpt_path, map_location=DEVICE)
    actor.load_state_dict(ckpt['actor'])
    actor.eval()
    print(f"\nSource: {source}")
    results[source] = {}

    for target in SWARN_CONFIGS:
        cfg = get_config(target)
        env = SwarmParallelEnv(cfg)
        cov_list, det_list = [], []

        for ep in range(1, N_EPISODES + 1):
            obs, _ = env.reset()
            while True:
                obs_tensor = torch.tensor(
                    np.stack([obs[a] for a in env.agents]),
                    dtype=torch.float32).to(DEVICE)
                with torch.no_grad():
                    actions, _ = actor(obs_tensor)
                actions_dict = {a: actions.cpu().numpy()[i]
                               for i, a in enumerate(env.agents)}
                obs, _, dones, _, info = env.step(actions_dict)
                if dones[env.agents[0]]:
                    break
            cov_list.append(info['coverage_pct'])
            det_list.append(info['detection_rate'])

        results[source][target] = {
            'coverage':  np.mean(cov_list),
            'detection': np.mean(det_list),
        }
        print(f"  → {target} | Cov: {np.mean(cov_list):.1f}% | "
              f"Det: {np.mean(det_list):.2f}")

# ── Save CSV ──────────────────────────────────────────────────────────────────
rows = []
for source in SOURCE_CKPTS:
    for target in SWARN_CONFIGS:
        rows.append({
            'source':    source,
            'target':    target,
            'coverage':  results[source][target]['coverage'],
            'detection': results[source][target]['detection'],
        })
df = pd.DataFrame(rows)
os.makedirs('results', exist_ok=True)
df.to_csv('results/transfer_matrix.csv', index=False)
print("\nSaved results/transfer_matrix.csv")

# ── Heatmaps ──────────────────────────────────────────────────────────────────
os.makedirs('figures', exist_ok=True)
configs = list(SWARN_CONFIGS.keys())
sources = list(SOURCE_CKPTS.keys())

for metric, title, cmap, fmt in [
    ('coverage',  'Coverage (%)',   'YlGn',   '.1f'),
    ('detection', 'Detection Rate', 'YlOrRd', '.2f'),
]:
    matrix = np.array([
        [results[s][t][metric] for t in configs]
        for s in sources
    ])

    fig, ax = plt.subplots(figsize=(18, 5))
    sns.heatmap(matrix, annot=True, fmt=fmt, cmap=cmap,
                linewidths=0.5, linecolor='white',
                annot_kws={'size': 10, 'weight': 'bold'},
                xticklabels=configs, yticklabels=sources,
                ax=ax, cbar_kws={'label': title})

    # Highlight diagonal (in-distribution)
    for i, s in enumerate(sources):
        j = configs.index(s) if s in configs else -1
        if j >= 0:
            ax.add_patch(plt.Rectangle((j, i), 1, 1,
                fill=False, edgecolor='blue', lw=3))

    ax.set_title(f'Transfer Analysis — {title}\n'
                 f'(Blue border = in-distribution performance)',
                 fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel('Target Configuration', fontsize=14, fontweight='bold')
    ax.set_ylabel('Source (Training) Config', fontsize=14, fontweight='bold')
    for l in ax.get_xticklabels() + ax.get_yticklabels():
        l.set_fontsize(11)
        l.set_fontweight('bold')

    fname = f'figures/fig16_transfer_{metric}.png'
    fig.savefig(fname, dpi=600, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {fname}")
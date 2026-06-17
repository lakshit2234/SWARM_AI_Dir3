# scripts/classical_transfer_analysis.py
# M16 — Classical algorithm transfer analysis
# Boids + PF evaluated on all 12 configs (no training needed)
import sys, os
sys.path.insert(0, '.')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from env.benchmark_configs import SWARN_CONFIGS, get_config
from env.swarm_env import SwarmEnv
from algos.boids import boids_action
from algos.potential_fields import pf_action
from utils.metrics import MetricsTracker

N_EPISODES = 10

def run_algo(algo_name, cfg, config_code):
    env     = SwarmEnv(cfg)
    tracker = MetricsTracker()
    cov_list, det_list = [], []

    for ep in range(1, N_EPISODES + 1):
        obs, _ = env.reset()
        tracker.episode_start()
        step = 0
        while True:
            if algo_name == 'boids':
                actions = np.array([
                    boids_action(i, env.agent_pos, env.agent_vel, cfg)
                    for i in range(cfg['n_agents'])
                ])
            else:
                actions = np.array([
                    pf_action(i, env.agent_pos, cfg)
                    for i in range(cfg['n_agents'])
                ])
            obs, _, done, _, info = env.step(actions)
            step += 1
            tracker.update(step, info, info.get('collisions', 0),
                          agent_positions=env.agent_pos)
            if done:
                break
        tracker.episode_end(ep)
        cov_list.append(info['coverage_pct'])
        det_list.append(info['detection_rate'])

    return np.mean(cov_list), np.mean(det_list)

algos   = ['boids', 'potential_fields']
configs = list(SWARN_CONFIGS.keys())
results = {a: {} for a in algos}

for algo in algos:
    print(f"\nClassical Transfer — {algo.upper().replace('_',' ')}")
    for config_code in configs:
        cfg = get_config(config_code)
        cov, det = run_algo(algo, cfg, config_code)
        results[algo][config_code] = {'coverage': cov, 'detection': det}
        print(f"  {config_code} | Cov: {cov:.1f}% | Det: {det:.2f}")

# ── Save CSV ──────────────────────────────────────────────────────────────────
rows = []
for algo in algos:
    for config_code in configs:
        rows.append({
            'algorithm':   algo,
            'config':      config_code,
            'coverage':    results[algo][config_code]['coverage'],
            'detection':   results[algo][config_code]['detection'],
        })
df = pd.DataFrame(rows)
os.makedirs('results', exist_ok=True)
df.to_csv('results/classical_transfer.csv', index=False)
print("\nSaved results/classical_transfer.csv")

# ── Compute transfer penalty vs MAPPO ────────────────────────────────────────
mappo_df = pd.read_csv('results/transfer_matrix.csv')
mappo_s5 = mappo_df[mappo_df['source'] == 'SWARN-S5'].set_index('target')

print("\nTransfer Penalty (MAPPO S5 vs Classical):")
print(f"{'Config':<12} {'MAPPO Cov':>10} {'Boids Cov':>10} "
      f"{'PF Cov':>8} {'MAPPO Det':>10} {'Boids Det':>10} {'PF Det':>8}")
print("-" * 72)
for config_code in configs:
    mc = mappo_s5.loc[config_code, 'coverage'] if config_code in mappo_s5.index else 0
    md = mappo_s5.loc[config_code, 'detection'] if config_code in mappo_s5.index else 0
    bc = results['boids'][config_code]['coverage']
    bd = results['boids'][config_code]['detection']
    pc = results['potential_fields'][config_code]['coverage']
    pd_ = results['potential_fields'][config_code]['detection']
    print(f"{config_code:<12} {mc:>9.1f}% {bc:>9.1f}% {pc:>7.1f}% "
          f"{md:>9.2f}  {bd:>9.2f}  {pd_:>7.2f}")

# ── Comparison heatmap ────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Classical vs Learned Transfer Analysis — Coverage (%)\n'
             'Classical algorithms use parameters from SWARN-S5 only',
             fontsize=14, fontweight='bold', y=1.03)

algo_labels = ['MAPPO\n(S5 trained)', 'Boids', 'Potential\nFields']
all_covs = np.array([
    [mappo_s5.loc[c, 'coverage'] if c in mappo_s5.index else 0 for c in configs],
    [results['boids'][c]['coverage'] for c in configs],
    [results['potential_fields'][c]['coverage'] for c in configs],
])

im = axes[0].imshow(all_covs, cmap='YlGn', aspect='auto',
                    vmin=0, vmax=100)
axes[0].set_xticks(range(len(configs)))
axes[0].set_yticks(range(3))
axes[0].set_xticklabels(configs, rotation=45, ha='right',
                         fontsize=10, fontweight='bold')
axes[0].set_yticklabels(algo_labels, fontsize=12, fontweight='bold')
axes[0].set_title('Coverage Comparison', fontsize=13, fontweight='bold')
for i in range(3):
    for j in range(len(configs)):
        axes[0].text(j, i, f'{all_covs[i,j]:.1f}',
                    ha='center', va='center', fontsize=9,
                    fontweight='bold',
                    color='white' if all_covs[i,j] > 60 else 'black')
plt.colorbar(im, ax=axes[0])

# Transfer penalty bar chart
mappo_avg = np.mean([mappo_s5.loc[c,'coverage']
                     if c in mappo_s5.index else 0 for c in configs])
boids_avg = np.mean([results['boids'][c]['coverage'] for c in configs])
pf_avg    = np.mean([results['potential_fields'][c]['coverage'] for c in configs])

ax = axes[1]
bars = ax.bar(['MAPPO\n(Learned)', 'Boids\n(Classical)', 'PF\n(Classical)'],
              [mappo_avg, boids_avg, pf_avg],
              color=['#2ECC71', '#3498DB', '#9B59B6'],
              alpha=0.85, edgecolor='white', width=0.5)
for bar, val in zip(bars, [mappo_avg, boids_avg, pf_avg]):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.5,
            f'{val:.1f}%', ha='center', va='bottom',
            fontsize=13, fontweight='bold')
ax.set_ylabel('Mean Coverage (%) Across All 12 Configs',
              fontweight='bold', fontsize=13)
ax.set_title('Average Transfer Performance', fontsize=13, fontweight='bold')
for l in ax.get_xticklabels() + ax.get_yticklabels():
    l.set_fontweight('bold')
    l.set_fontsize(12)
ax.set_ylim(0, 100)
ax.grid(True, alpha=0.25, linestyle='--')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Per-config gap chart
mappo_vals = [mappo_s5.loc[c,'coverage']
              if c in mappo_s5.index else 0 for c in configs]
boids_vals = [results['boids'][c]['coverage'] for c in configs]
x = np.arange(len(configs))
ax = axes[2]
ax.plot(x, mappo_vals, 'o-', color='#2ECC71', linewidth=2.5,
        markersize=8, label='MAPPO', markerfacecolor='white',
        markeredgewidth=2.5)
ax.plot(x, boids_vals, 's--', color='#3498DB', linewidth=2.5,
        markersize=8, label='Boids', markerfacecolor='white',
        markeredgewidth=2.5)
ax.fill_between(x, boids_vals, mappo_vals, alpha=0.15, color='#2ECC71',
                label='MAPPO advantage')
ax.set_xticks(x)
ax.set_xticklabels(configs, rotation=45, ha='right',
                   fontsize=10, fontweight='bold')
ax.set_ylabel('Coverage (%)', fontweight='bold', fontsize=13)
ax.set_title('MAPPO vs Boids Per Config', fontsize=13, fontweight='bold')
ax.legend(fontsize=12)
for t in ax.get_legend().get_texts():
    t.set_fontweight('bold')
for l in ax.get_yticklabels():
    l.set_fontweight('bold')
ax.grid(True, alpha=0.25, linestyle='--')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

fig.tight_layout()
fig.savefig('figures/fig18_classical_transfer.png', dpi=600, bbox_inches='tight')
plt.close(fig)
print("Saved figures/fig18_classical_transfer.png")
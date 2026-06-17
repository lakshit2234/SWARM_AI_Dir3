# scripts/eval_mappo_all_configs.py
import sys, os
sys.path.insert(0, '.')
import torch
import numpy as np
from env.benchmark_configs import SWARN_CONFIGS, get_config
from algos.mappo.env_wrapper import SwarmParallelEnv
from algos.mappo.networks import Actor
from utils.metrics import MetricsTracker

DEVICE     = torch.device('cuda')
OBS_DIM    = 29
ACTION_DIM = 2
HIDDEN     = 128
N_EPISODES = 30

# S11/S12 use best available checkpoint (S10 — closest scale)
CKPT_MAP = {
    'SWARN-S1':  'results/mappo/SWARN-S1/checkpoint.pt',
    'SWARN-S2':  'results/mappo/SWARN-S2/checkpoint.pt',
    'SWARN-S3':  'results/mappo/SWARN-S3/checkpoint.pt',
    'SWARN-S4':  'results/mappo/SWARN-S4/checkpoint.pt',
    'SWARN-S5':  'results/mappo/SWARN-S5/checkpoint.pt',
    'SWARN-S6':  'results/mappo/SWARN-S6/checkpoint.pt',
    'SWARN-S7':  'results/mappo/SWARN-S7/checkpoint.pt',
    'SWARN-S8':  'results/mappo/SWARN-S8/checkpoint.pt',
    'SWARN-S9':  'results/mappo/SWARN-S9/checkpoint.pt',
    'SWARN-S10': 'results/mappo/SWARN-S10/checkpoint.pt',
    'SWARN-S11': 'results/mappo/SWARN-S10/checkpoint.pt',  # zero-shot
    'SWARN-S12': 'results/mappo/SWARN-S10/checkpoint.pt',  # zero-shot
}

for config_code in SWARN_CONFIGS:
    print(f"\n{'='*55}")
    print(f"MAPPO Eval — {config_code}")
    print(f"  {SWARN_CONFIGS[config_code]['description']}")
    print('='*55)

    cfg      = get_config(config_code)
    n_agents = cfg['n_agents']

    actor = Actor(OBS_DIM, ACTION_DIM, HIDDEN).to(DEVICE)
    ckpt  = torch.load(CKPT_MAP[config_code], map_location=DEVICE)
    actor.load_state_dict(ckpt['actor'])
    actor.eval()

    env     = SwarmParallelEnv(cfg)
    tracker = MetricsTracker()

    for ep in range(1, N_EPISODES + 1):
        obs, _ = env.reset()
        tracker.episode_start()
        step = 0
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
            tracker.update(step, info, info.get('collisions', 0),
                          agent_positions=env.env.agent_pos)
            if dones[env.agents[0]]:
                break
        tracker.episode_end(ep)
        print(f"  Ep {ep:>2} | Cov: {info['coverage_pct']:.1f}% | "
              f"Det: {info['detection_rate']:.2f}")

    save_path = f'results/mappo/{config_code}/metrics.csv'
    tracker.save_csv(save_path)
    print(f"  Saved {save_path}")
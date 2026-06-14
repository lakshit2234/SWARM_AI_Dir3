# algos/mappo/evaluate.py
import torch
import numpy as np
import os
from algos.mappo.env_wrapper import SwarmParallelEnv
from algos.mappo.networks import Actor
from utils.metrics import MetricsTracker

DEVICE     = torch.device('cuda')
OBS_DIM    = 29
ACTION_DIM = 2
HIDDEN     = 128
N_EPISODES = 30

# Load best Stage 3 actor
actor = Actor(OBS_DIM, ACTION_DIM, HIDDEN).to(DEVICE)
ckpt = torch.load('results/mappo/ckpt_stage3_sensed_1000000.pt', map_location=DEVICE)
actor.load_state_dict(ckpt['actor'])
actor.eval()
print("Stage 3 checkpoint loaded for evaluation!")

scenarios = {
    'easy': {
        'n_agents': 5, 'map_size': 50, 'n_targets': 3,
        'n_obstacles': 5, 'comm_dropout': 0.0, 'max_steps': 300,
        'obs_radius': 10, 'coverage_mode': 'sensed',
    },
    'medium': {
        'n_agents': 10, 'map_size': 100, 'n_targets': 5,
        'n_obstacles': 15, 'comm_dropout': 0.0, 'max_steps': 500,
        'obs_radius': 10, 'coverage_mode': 'sensed',
    },
    'hard': {
        'n_agents': 10, 'map_size': 100, 'n_targets': 8,
        'n_obstacles': 20, 'comm_dropout': 0.2, 'max_steps': 500,
        'obs_radius': 10, 'coverage_mode': 'sensed',
    },
}

for scenario_name, cfg in scenarios.items():
    print(f"\nEvaluating scenario: {scenario_name}")
    env     = SwarmParallelEnv(cfg)
    tracker = MetricsTracker()

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
            tracker.update(step, info, info.get('collisions', 0),
                          agent_positions=env.env.agent_pos)

            if dones[env.agents[0]]:
                break

        tracker.episode_end(ep)
        print(f"  Episode {ep:>2} | Coverage: {info['coverage_pct']:.1f}% | "
              f"Detected: {info['detection_rate']:.2f}")

    os.makedirs('results/mappo', exist_ok=True)
    tracker.save_csv(f'results/mappo/metrics_{scenario_name}.csv')
    print(f"Saved results/mappo/metrics_{scenario_name}.csv")
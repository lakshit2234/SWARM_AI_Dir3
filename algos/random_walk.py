# algos/random_walk.py
import numpy as np
from env.swarm_env import SwarmEnv
from utils.metrics import MetricsTracker
import os

def run_random_walk(cfg, n_episodes=30, save_path=None):
    env = SwarmEnv(cfg)
    tracker = MetricsTracker()

    for ep in range(1, n_episodes + 1):
        obs, _ = env.reset()
        tracker.episode_start()
        step = 0

        while True:
            # Random action: uniform in [-1, 1] for each agent
            actions = np.random.uniform(-1, 1, (cfg['n_agents'], 2))
            obs, rews, done, _, info = env.step(actions)
            step += 1
            tracker.update(step, info, info.get('collisions', 0))
            if done:
                break

        tracker.episode_end(ep)
        print(f"  Ep {ep:>2} | Coverage: {info['coverage_pct']:.1f}% | "
              f"Detected: {info['detection_rate']:.2f}")

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        tracker.save_csv(save_path)
        print(f"Saved {save_path}")

if __name__ == "__main__":
    cfg = {
        'n_agents': 10, 'map_size': 100, 'n_targets': 5,
        'n_obstacles': 15, 'comm_dropout': 0.0, 'max_steps': 500,
        'obs_radius': 10, 'coverage_mode': 'sensed',
    }
    for scenario in ['easy', 'medium', 'hard']:
        if scenario == 'easy':
            cfg.update({'n_agents': 5, 'map_size': 50, 'n_targets': 3,
                       'n_obstacles': 5, 'max_steps': 300})
        elif scenario == 'medium':
            cfg.update({'n_agents': 10, 'map_size': 100, 'n_targets': 5,
                       'n_obstacles': 15, 'max_steps': 500})
        elif scenario == 'hard':
            cfg.update({'n_agents': 10, 'map_size': 100, 'n_targets': 8,
                       'n_obstacles': 20, 'comm_dropout': 0.2, 'max_steps': 500})
        print(f"\nRunning Random Walk — {scenario}")
        run_random_walk(cfg, n_episodes=30,
                       save_path=f'results/random_walk/metrics_{scenario}.csv')
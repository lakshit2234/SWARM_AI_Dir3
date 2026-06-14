# algos/boids.py
import numpy as np
from env.swarm_env import SwarmEnv
from utils.metrics import MetricsTracker
import os

def boids_action(agent_idx, positions, velocities, cfg):
    n = cfg['n_agents']
    comm_r = cfg.get('comm_radius', 20)
    obs_r  = cfg.get('obs_radius', 10)

    pos = positions[agent_idx]
    vel = velocities[agent_idx]

    sep = np.zeros(2)
    ali = np.zeros(2)
    coh = np.zeros(2)
    neighbors = 0

    for j in range(n):
        if j == agent_idx:
            continue
        diff = pos - positions[j]
        dist = np.linalg.norm(diff)
        if dist < comm_r and dist > 0:
            # Separation
            if dist < obs_r:
                sep += diff / (dist ** 2 + 1e-6)
            # Alignment
            ali += velocities[j]
            # Cohesion
            coh += positions[j]
            neighbors += 1

    if neighbors > 0:
        ali /= neighbors
        coh  = (coh / neighbors) - pos
        coh  = coh / (np.linalg.norm(coh) + 1e-6)
        ali  = ali / (np.linalg.norm(ali) + 1e-6)

    sep_w, ali_w, coh_w = 1.5, 1.0, 1.0
    action = sep_w * sep + ali_w * ali + coh_w * coh
    norm = np.linalg.norm(action)
    if norm > 1e-6:
        action = action / norm
    return np.clip(action, -1, 1)


def run_boids(cfg, n_episodes=30, save_path=None):
    env = SwarmEnv(cfg)
    tracker = MetricsTracker()

    for ep in range(1, n_episodes + 1):
        obs, _ = env.reset()
        tracker.episode_start()
        step = 0

        while True:
            actions = np.array([
                boids_action(i, env.agent_pos, env.agent_vel, cfg)
                for i in range(cfg['n_agents'])
            ])
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
    scenarios = {
        'easy': {'n_agents': 5,  'map_size': 50,  'n_targets': 3,
                 'n_obstacles': 5,  'comm_dropout': 0.0, 'max_steps': 300,
                 'obs_radius': 10, 'coverage_mode': 'sensed'},
        'medium': {'n_agents': 10, 'map_size': 100, 'n_targets': 5,
                   'n_obstacles': 15, 'comm_dropout': 0.0, 'max_steps': 500,
                   'obs_radius': 10, 'coverage_mode': 'sensed'},
        'hard': {'n_agents': 10, 'map_size': 100, 'n_targets': 8,
                 'n_obstacles': 20, 'comm_dropout': 0.2, 'max_steps': 500,
                 'obs_radius': 10, 'coverage_mode': 'sensed'},
    }
    for scenario, cfg in scenarios.items():
        print(f"\nRunning Boids — {scenario}")
        run_boids(cfg, n_episodes=30,
                  save_path=f'results/boids/metrics_{scenario}.csv')
# algos/potential_fields.py
import numpy as np
from env.swarm_env import SwarmEnv
from utils.metrics import MetricsTracker
import os

def pf_action(agent_idx, positions, cfg):
    pos     = positions[agent_idx]
    n       = cfg['n_agents']
    map_s   = cfg['map_size']
    obs_r   = cfg.get('obs_radius', 10)

    force = np.zeros(2)

    # Attractive force toward map centre (exploration bias)
    centre     = np.array([map_s / 2, map_s / 2])
    to_centre  = centre - pos
    dist_c     = np.linalg.norm(to_centre) + 1e-6
    force     += 0.5 * (to_centre / dist_c)

    # Repulsive force from other agents
    for j in range(n):
        if j == agent_idx:
            continue
        diff = pos - positions[j]
        dist = np.linalg.norm(diff) + 1e-6
        if dist < obs_r:
            force += 2.0 * (diff / dist) / (dist + 1e-6)

    # Repulsive force from map boundaries
    margin = 5.0
    if pos[0] < margin:
        force[0] += 2.0
    if pos[0] > map_s - margin:
        force[0] -= 2.0
    if pos[1] < margin:
        force[1] += 2.0
    if pos[1] > map_s - margin:
        force[1] -= 2.0

    norm = np.linalg.norm(force)
    if norm > 1e-6:
        force = force / norm
    return np.clip(force, -1, 1)


def run_potential_fields(cfg, n_episodes=30, save_path=None):
    env     = SwarmEnv(cfg)
    tracker = MetricsTracker()

    for ep in range(1, n_episodes + 1):
        obs, _ = env.reset()
        tracker.episode_start()
        step = 0

        while True:
            actions = np.array([
                pf_action(i, env.agent_pos, cfg)
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
        print(f"\nRunning Potential Fields — {scenario}")
        run_potential_fields(cfg, n_episodes=30,
                             save_path=f'results/potential_fields/metrics_{scenario}.csv')
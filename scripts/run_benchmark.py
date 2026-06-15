# scripts/run_benchmark.py
# Master benchmark runner — runs any algorithm on any SWARN config
# Usage: python scripts/run_benchmark.py --algo mappo --config SWARN-S1
# Usage: python scripts/run_benchmark.py --algo all --config all

import sys, os, argparse
sys.path.insert(0, '.')
from env.benchmark_configs import SWARN_CONFIGS, get_config

def run_random_walk(cfg, config_code, n_episodes=30):
    import numpy as np
    from env.swarm_env import SwarmEnv
    from utils.metrics import MetricsTracker
    env     = SwarmEnv(cfg)
    tracker = MetricsTracker()
    for ep in range(1, n_episodes + 1):
        obs, _ = env.reset()
        tracker.episode_start()
        step = 0
        while True:
            actions = np.random.uniform(-1, 1, (cfg['n_agents'], 2))
            obs, _, done, _, info = env.step(actions)
            step += 1
            tracker.update(step, info, info.get('collisions', 0),
                          agent_positions=env.agent_pos)
            if done:
                break
        tracker.episode_end(ep)
        print(f"  Ep {ep:>2} | Cov: {info['coverage_pct']:.1f}% | Det: {info['detection_rate']:.2f}")
    save_results(tracker, 'random_walk', config_code)

def run_boids(cfg, config_code, n_episodes=30):
    import numpy as np
    from env.swarm_env import SwarmEnv
    from utils.metrics import MetricsTracker
    from algos.boids import boids_action
    env     = SwarmEnv(cfg)
    tracker = MetricsTracker()
    for ep in range(1, n_episodes + 1):
        obs, _ = env.reset()
        tracker.episode_start()
        step = 0
        while True:
            actions = np.array([boids_action(i, env.agent_pos, env.agent_vel, cfg)
                               for i in range(cfg['n_agents'])])
            obs, _, done, _, info = env.step(actions)
            step += 1
            tracker.update(step, info, info.get('collisions', 0),
                          agent_positions=env.agent_pos)
            if done:
                break
        tracker.episode_end(ep)
        print(f"  Ep {ep:>2} | Cov: {info['coverage_pct']:.1f}% | Det: {info['detection_rate']:.2f}")
    save_results(tracker, 'boids', config_code)

def run_potential_fields(cfg, config_code, n_episodes=30):
    import numpy as np
    from env.swarm_env import SwarmEnv
    from utils.metrics import MetricsTracker
    from algos.potential_fields import pf_action
    env     = SwarmEnv(cfg)
    tracker = MetricsTracker()
    for ep in range(1, n_episodes + 1):
        obs, _ = env.reset()
        tracker.episode_start()
        step = 0
        while True:
            actions = np.array([pf_action(i, env.agent_pos, cfg)
                               for i in range(cfg['n_agents'])])
            obs, _, done, _, info = env.step(actions)
            step += 1
            tracker.update(step, info, info.get('collisions', 0),
                          agent_positions=env.agent_pos)
            if done:
                break
        tracker.episode_end(ep)
        print(f"  Ep {ep:>2} | Cov: {info['coverage_pct']:.1f}% | Det: {info['detection_rate']:.2f}")
    save_results(tracker, 'potential_fields', config_code)

def save_results(tracker, algo, config_code):
    path = f'results/{algo}/{config_code}/metrics.csv'
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tracker.save_csv(path)
    print(f"  Saved {path}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--algo',   default='random_walk',
                   choices=['random_walk', 'boids', 'potential_fields', 'mappo', 'qmix', 'all'])
    p.add_argument('--config', default='SWARN-S1',
                   help='Config code e.g. SWARN-S1 or "all"')
    p.add_argument('--episodes', type=int, default=30)
    args = p.parse_args()

    configs = list(SWARN_CONFIGS.keys()) if args.config == 'all' else [args.config]
    algos   = ['random_walk', 'boids', 'potential_fields'] \
              if args.algo == 'all' else [args.algo]

    for config_code in configs:
        cfg = get_config(config_code)
        for algo in algos:
            print(f"\n{'='*55}")
            print(f"{algo.upper()} — {config_code}")
            print(f"  {SWARN_CONFIGS[config_code]['description']}")
            print('='*55)
            if algo == 'random_walk':
                run_random_walk(cfg, config_code, args.episodes)
            elif algo == 'boids':
                run_boids(cfg, config_code, args.episodes)
            elif algo == 'potential_fields':
                run_potential_fields(cfg, config_code, args.episodes)
            else:
                print(f"  {algo} — use dedicated training script")

if __name__ == '__main__':
    main()
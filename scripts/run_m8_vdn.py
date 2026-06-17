# scripts/run_m8_vdn.py
"""
M8 VDN — train and eval on SWARN-S1 to SWARN-S10.

Usage:
    python scripts/run_m8_vdn.py --config SWARN-S1
    python scripts/run_m8_vdn.py --config SWARN-S1 --eval_only --ckpt results/vdn/SWARN-S1/checkpoints/vdn_step_200000_final.pt
    python scripts/run_m8_vdn.py --list
"""

import argparse, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.benchmark_configs import get_config, SWARN_CONFIGS, list_configs
from algos.vdn.train_vdn   import train, save_checkpoint, ACTION_MAP, N_ACTIONS
from algos.vdn.eval_vdn    import eval_vdn


VRAM_SAFE_MAX_AGENTS = 50


def get_train_cfg(n_agents: int) -> dict:
    if n_agents <= 10:
        return {
            "total_steps": 200_000, "batch_size": 16,
            "epsilon_decay_steps": 120_000, "buffer_size": 20_000,
            "train_start": 2_000, "checkpoint_freq": 50_000,
        }
    elif n_agents <= 25:
        return {
            "total_steps": 150_000, "batch_size": 16,
            "epsilon_decay_steps": 90_000, "buffer_size": 10_000,
            "train_start": 1_000, "checkpoint_freq": 50_000,
        }
    else:
        return {
            "total_steps": 100_000, "batch_size": 16,
            "epsilon_decay_steps": 60_000, "buffer_size": 5_000,
            "train_start": 500, "checkpoint_freq": 25_000,
        }


def run(config_code: str, n_eval_episodes: int = 30):
    env_cfg   = get_config(config_code)
    n         = env_cfg["n_agents"]
    train_cfg = get_train_cfg(n)

    results_dir = os.path.join("results", "vdn", config_code)
    ckpt_dir    = os.path.join(results_dir, "checkpoints")

    device_str = "cuda" if (torch.cuda.is_available()
                            and n <= VRAM_SAFE_MAX_AGENTS) else "cpu"

    print(f"\n{'='*60}")
    print(f"  VDN | {config_code} | {SWARN_CONFIGS[config_code]['description']}")
    print(f"  n_agents={n} | map={env_cfg['map_size']} | "
          f"targets={env_cfg['n_targets']} | obstacles={env_cfg['n_obstacles']}")
    print(f"  detect_radius={env_cfg.get('detect_radius',2.0)} | "
          f"coverage_mode={env_cfg.get('coverage_mode','sensed')}")
    print(f"  total_steps={train_cfg['total_steps']:,} | batch={train_cfg['batch_size']}")
    print(f"{'='*60}\n")

    cfg = {
        **env_cfg,
        **train_cfg,
        "checkpoint_dir": ckpt_dir,
        "results_dir":    results_dir,
        "obs_dim":        29,
        "lr":             5e-5,
        "gamma":          0.99,
        "target_update_freq": 500,
    }

    agent_nets = train(cfg)

    # Eval with final checkpoint
    final_ckpt = os.path.join(ckpt_dir,
                              f"vdn_step_{train_cfg['total_steps']}_final.pt")
    eval_vdn(
        checkpoint_path = final_ckpt,
        env_cfg         = env_cfg,
        n_episodes      = n_eval_episodes,
        results_dir     = results_dir,
        config_code     = config_code,
    )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",    type=str, default=None)
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--ckpt",      type=str, default=None)
    parser.add_argument("--episodes",  type=int, default=30)
    parser.add_argument("--list",      action="store_true")
    args = parser.parse_args()

    if args.list:
        list_configs()
        return

    if not args.config:
        parser.error("--config required")

    if args.eval_only:
        if not args.ckpt:
            parser.error("--ckpt required with --eval_only")
        env_cfg     = get_config(args.config)
        results_dir = os.path.join("results", "vdn", args.config)
        eval_vdn(args.ckpt, env_cfg, args.episodes, results_dir, args.config)
    else:
        run(args.config, args.episodes)


if __name__ == "__main__":
    main()
# algos/vdn/eval_vdn.py

import os
import numpy as np
import torch

from env.swarm_env       import SwarmEnv
from utils.metrics       import MetricsTracker
from algos.vdn.networks  import AgentQNetwork
from algos.vdn.train_vdn import ACTION_MAP, N_ACTIONS


def eval_vdn(
    checkpoint_path: str,
    env_cfg:         dict,
    n_episodes:      int = 30,
    results_dir:     str = "results/vdn",
    config_code:     str = "SWARN-S1",
):
    n       = env_cfg["n_agents"]
    obs_dim = 29
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    agent_nets = [AgentQNetwork(obs_dim, N_ACTIONS, hidden=128).to(device) for _ in range(n)]
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    for i, net in enumerate(agent_nets):
        net.load_state_dict(ckpt["agent_states"][i])
        net.eval()

    print(f"  Loaded: {checkpoint_path}")
    print(f"  Eval: {config_code} | {n_episodes} episodes\n")

    env     = SwarmEnv(env_cfg)
    metrics = MetricsTracker()

    for ep in range(1, n_episodes + 1):
        obs_array, _ = env.reset()
        metrics.episode_start()
        done = False

        while not done:
            int_actions = np.zeros(n, dtype=np.int64)
            with torch.no_grad():
                for i, net in enumerate(agent_nets):
                    obs_t  = torch.FloatTensor(obs_array[i]).unsqueeze(0).to(device)
                    q_vals = net(obs_t)
                    best   = q_vals.argmax(dim=1).item()
                    if best == 0:
                        best = q_vals.squeeze(0)[1:].argmax().item() + 1
                    int_actions[i] = best

            vel_actions = ACTION_MAP[int_actions]
            obs_array, _, done, _, info = env.step(vel_actions)
            metrics.update(env.step_count, info,
                           int(info.get("collisions", 0)),
                           agent_positions=env.agent_pos)

        metrics.episode_end(ep)
        cov = metrics.episodes[-1]["coverage_pct"]
        det = metrics.episodes[-1]["detection_rate"]
        col = metrics.episodes[-1]["total_collisions"]
        print(f"    ep {ep:>3}/{n_episodes} | cov={cov:5.1f}% | det={det:.2f} | col={col}")

    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, "metrics.csv")
    metrics.save_csv(csv_path)

    covs = [e["coverage_pct"]    for e in metrics.episodes]
    dets = [e["detection_rate"]  for e in metrics.episodes]
    cols = [e["total_collisions"] for e in metrics.episodes]
    print(f"\n  ── {config_code} VDN Results ──")
    print(f"  Coverage:   {np.mean(covs):.1f}% ± {np.std(covs):.1f}")
    print(f"  Detection:  {np.mean(dets):.3f} ± {np.std(dets):.3f}")
    print(f"  Collisions: {np.mean(cols):.1f} ± {np.std(cols):.1f}")
    print(f"  CSV → {csv_path}")
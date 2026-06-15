# algos/qmix/eval_qmix.py

import os
import numpy as np
import torch
import matplotlib

# import env
# from utils import metrics
matplotlib.use("Agg")   # no display needed — saves to file
import matplotlib.pyplot as plt
import seaborn as sns

from env.swarm_env import SwarmEnv
from utils.metrics import MetricsTracker
from algos.qmix.networks import AgentQNetwork, QMIXMixer
from algos.qmix.train_qmix import ACTION_MAP, N_ACTIONS


def load_checkpoint(checkpoint_path: str, n_agents: int, obs_dim: int, state_dim: int, embed_dim: int = 64):

    """Load trained weights from a .pt checkpoint file."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    agent_nets = [AgentQNetwork(obs_dim, N_ACTIONS).to(device) for _ in range(n_agents)]
    mixer      = QMIXMixer(n_agents, state_dim, embed_dim).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    for i, net in enumerate(agent_nets):
        net.load_state_dict(ckpt["agent_states"][i])
    mixer.load_state_dict(ckpt["mixer_state"])

    # Set to eval mode — disables dropout if any
    for net in agent_nets:
        net.eval()
    mixer.eval()

    print(f"  Loaded checkpoint from {checkpoint_path}")
    return agent_nets, mixer, device

# Add this at the top of run_eval() in eval_qmix.py as a sanity check
AGREED_ENV = {
    "easy":   {"map_size": 50,  "n_targets": 3, "n_obstacles": 5,  "max_steps": 300},
    "medium": {"map_size": 100, "n_targets": 5, "n_obstacles": 15, "max_steps": 500},
    "hard":   {"map_size": 100, "n_targets": 8, "n_obstacles": 20, "max_steps": 500},
}


def run_eval(
    checkpoint_path: str,
    scenario:        str  = "medium",
    n_episodes:      int  = 30,
    n_agents:        int  = 6,
    results_dir:     str  = "results/qmix",
    record_weights:  bool = True,   # for heatmap figure
):
    """
    Run evaluation episodes with a trained QMIX model.
    Saves:
      - metrics_{scenario}.csv
      - mixer_weights_heatmap.png  (if record_weights=True)
    """
    obs_dim   = 29
    state_dim = obs_dim * n_agents

    agent_nets, mixer, device = load_checkpoint(
        checkpoint_path, n_agents, obs_dim, state_dim
    )

    # Environment configs per scenario
    env_configs = {
        "easy":   {"map_size": 50,  "n_targets": 3, "n_obstacles": 5,  "max_steps": 300},
        "medium": {"map_size": 100, "n_targets": 5, "n_obstacles": 15, "max_steps": 500},
        "hard":   {"map_size": 100, "n_targets": 8, "n_obstacles": 20, "max_steps": 500, "comm_dropout": 0.2},
    }
    env_cfg = env_configs.get(scenario, env_configs["medium"])
    env_cfg["n_agents"] = n_agents

    env     = SwarmEnv(env_cfg)
    metrics = MetricsTracker()

    all_weights = []   # collect for heatmap

    print(f"\n  Evaluating QMIX on '{scenario}' scenario ({n_episodes} episodes)...\n")

    for ep in range(1, n_episodes + 1):
        obs_array, _ = env.reset()
        metrics.episode_start()
        done        = False
        ep_weights  = []

        while not done:
            # Greedy actions only (epsilon = 0 during eval)
            int_actions = np.zeros(n_agents, dtype=np.int64)
            agent_qs    = []

            for i, net in enumerate(agent_nets):
                obs_t  = torch.FloatTensor(obs_array[i]).unsqueeze(0).to(device)
                q_vals = net(obs_t)                          # [1, n_actions]
                int_actions[i] = q_vals.argmax(dim=1).item()
                agent_qs.append(q_vals.max(dim=1)[0])       # scalar Q for chosen action

            # Record mixer weights for first episode only
            if ep == 1 and record_weights:
                state_t  = torch.FloatTensor(obs_array.flatten()).unsqueeze(0).to(device)
                qs_t     = torch.cat(agent_qs, dim=0).unsqueeze(0)  # [1, n_agents]
                with torch.no_grad():
                    w1 = torch.abs(mixer.hyper_w1(state_t))
                    # Sum weights per agent as a proxy for attention
                    w_per_agent = w1.view(n_agents, -1).sum(dim=1).cpu().numpy()
                ep_weights.append(w_per_agent)

            vel_actions = ACTION_MAP[int_actions]
            obs_array, rewards, done, _, info = env.step(vel_actions)
            metrics.update(env.step_count, info,
               int(info.get("collisions", 0)),
               agent_positions=env.agent_pos)

        metrics.episode_end(ep)
        if ep_weights:
            all_weights = ep_weights

        cov = metrics.episodes[-1]["coverage_pct"]
        det = metrics.episodes[-1]["detection_rate"]
        print(f"  Ep {ep:>3}/{n_episodes} | cov={cov:5.1f}% | det={det:.2f}")

    # ── Save CSV ─────────────────────────────────────────────────────────────
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, f"metrics_{scenario}.csv")
    metrics.save_csv(csv_path)

    # ── Mixer weight heatmap ─────────────────────────────────────────────────
    if record_weights and all_weights:
        weights_matrix = np.array(all_weights)   # [timesteps, n_agents]

        fig, ax = plt.subplots(figsize=(14, 4))
        sns.heatmap(
            weights_matrix.T,        # [n_agents, timesteps]
            cmap="Blues",
            ax=ax,
            cbar_kws={"label": "Mixing weight (relative attention)"},
            xticklabels=False,       # let us set them manually below
        )

        # Manually place tick marks every 20 steps
        n_steps   = weights_matrix.shape[0]
        tick_pos   = list(range(0, n_steps, 20))
        tick_labels = [str(t) for t in tick_pos]
        ax.set_xticks(tick_pos)
        ax.set_xticklabels(tick_labels, rotation=0, fontsize=8)

        ax.set_yticks(range(n_agents))
        ax.set_yticklabels(
            [f"Agent {i}" for i in range(n_agents)],
            rotation=0, fontsize=9
        )
        ax.set_xlabel("Timestep", fontsize=10)
        ax.set_title(
            f"QMIX Mixer Attention Weights — {scenario} scenario (evaluation episode 1)",
            fontsize=11
        )
        plt.tight_layout()
        heatmap_path = os.path.join(results_dir, "mixer_weights_heatmap.png")
        plt.savefig(heatmap_path, dpi=300)
        plt.close()
        print(f"\n  Heatmap saved to {heatmap_path}")

    # Summary stats
    covs = [e["coverage_pct"]    for e in metrics.episodes]
    dets = [e["detection_rate"]   for e in metrics.episodes]
    cols = [e["total_collisions"] for e in metrics.episodes]
    print(f"\n  ── {scenario.upper()} Results ──────────────────")
    print(f"  Coverage:    {np.mean(covs):.1f}% ± {np.std(covs):.1f}")
    print(f"  Detection:   {np.mean(dets):.3f} ± {np.std(dets):.3f}")
    print(f"  Collisions:  {np.mean(cols):.1f} ± {np.std(cols):.1f}")
# scripts/run_m8_qmix.py
"""
M8 — QMIX on SWARN-S1 to SWARN-S10
RTX 4050 safe: VRAM guard, CPU fallback for large configs.

Usage:
    python scripts/run_m8_qmix.py --config SWARN-S1
    python scripts/run_m8_qmix.py --config SWARN-S1 --eval_only --ckpt results/qmix/SWARN-S1/checkpoints/final.pt
    python scripts/run_m8_qmix.py --list
"""

import argparse, os, sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.swarm_env          import SwarmEnv
from env.benchmark_configs  import get_config, SWARN_CONFIGS
from utils.metrics          import MetricsTracker
from algos.qmix.networks    import AgentQNetwork, QMIXMixer
from algos.qmix.replay_buffer import ReplayBuffer
from algos.qmix.train_qmix  import ACTION_MAP, N_ACTIONS, select_actions, compute_qmix_loss, save_checkpoint

# ── VRAM limits per agent count ───────────────────────────────────────────────
# If n_agents exceeds threshold, fall back to CPU to avoid OOM crash
VRAM_SAFE_MAX_AGENTS = 25   # above this → CPU

# ── Training budget per config tier ──────────────────────────────────────────
def get_train_cfg(n_agents: int, config_code: str) -> dict:
    """Returns training hyperparams scaled to agent count and GPU safety."""

    if n_agents <= 10:
        return {
            "total_steps":         200_000,
            "batch_size":          16,
            "lr":                  5e-5,
            "epsilon_decay_steps": 120_000,
            "buffer_size":         20_000,
            "train_start":         2_000,
            "target_update_freq":  500,
            "checkpoint_freq":     50_000,
            "embed_dim":           64,
        }
    elif n_agents <= 25:
        # S7, S8 — reduce batch and buffer to stay under 6GB VRAM
        return {
            "total_steps":         150_000,
            "batch_size":           16,      # halved
            "lr":                  5e-5,
            "epsilon_decay_steps": 90_000,
            "buffer_size":         10_000,  # halved
            "train_start":         1_000,
            "target_update_freq":  500,
            "checkpoint_freq":     50_000,
            "embed_dim":           32,      # smaller mixer
        }
    else:
        # S9, S10 — CPU only, minimal memory
        return {
            "total_steps":         100_000,
            "batch_size":          16,
            "lr":                  5e-5,
            "epsilon_decay_steps": 60_000,
            "buffer_size":         5_000,
            "train_start":         500,
            "target_update_freq":  300,
            "checkpoint_freq":     25_000,
            "embed_dim":           32,
        }


def train_and_eval(config_code: str, n_eval_episodes: int = 30):
    env_cfg   = get_config(config_code)
    n         = env_cfg["n_agents"]
    train_cfg = get_train_cfg(n, config_code)
    obs_dim   = 29
    state_dim = obs_dim * n

    # ── Device selection ──────────────────────────────────────────────────────
    if torch.cuda.is_available() and n <= VRAM_SAFE_MAX_AGENTS:
        device = torch.device("cuda")
        print(f"  Device: cuda ({torch.cuda.get_device_name(0)})")
        torch.cuda.set_per_process_memory_fraction(0.5)  # cap at 3GB of 6GB
    else:
        device = torch.device("cpu")
        if n > VRAM_SAFE_MAX_AGENTS:
            print(f"  Device: CPU (n_agents={n} > VRAM safe limit {VRAM_SAFE_MAX_AGENTS})")
        else:
            print(f"  Device: CPU (no CUDA)")

    # ── Dirs ──────────────────────────────────────────────────────────────────
    results_dir = os.path.join("results", "qmix", config_code)
    ckpt_dir    = os.path.join(results_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  {config_code} | {SWARN_CONFIGS[config_code]['description']}")
    print(f"  n_agents={n} | map={env_cfg['map_size']} | "
          f"targets={env_cfg['n_targets']} | obstacles={env_cfg['n_obstacles']}")
    print(f"  detect_radius={env_cfg.get('detect_radius', 5.0)} | "
          f"coverage_mode={env_cfg.get('coverage_mode','sensed')}")
    print(f"  total_steps={train_cfg['total_steps']:,} | "
          f"batch={train_cfg['batch_size']} | embed_dim={train_cfg['embed_dim']}")
    print(f"{'='*60}\n")

    # ── Networks ──────────────────────────────────────────────────────────────
    embed_dim    = train_cfg["embed_dim"]
    agent_nets   = [AgentQNetwork(obs_dim, N_ACTIONS, hidden=128).to(device) for _ in range(n)]
    target_nets  = [AgentQNetwork(obs_dim, N_ACTIONS, hidden=128).to(device) for _ in range(n)]
    mixer        = QMIXMixer(n, state_dim, embed_dim).to(device)
    target_mixer = QMIXMixer(n, state_dim, embed_dim).to(device)

    for i in range(n):
        target_nets[i].load_state_dict(agent_nets[i].state_dict())
    target_mixer.load_state_dict(mixer.state_dict())

    all_params = list(mixer.parameters())
    for net in agent_nets:
        all_params += list(net.parameters())
    optimizer = optim.Adam(all_params, lr=train_cfg["lr"])

    # ── Buffer + metrics ──────────────────────────────────────────────────────
    buffer  = ReplayBuffer(train_cfg["buffer_size"], n, obs_dim, N_ACTIONS)
    metrics = MetricsTracker()

    # ── Env ───────────────────────────────────────────────────────────────────
    env           = SwarmEnv(env_cfg)
    obs_array, _  = env.reset()
    metrics.episode_start()

    # epsilon       = 1.0
    epsilon_delta = (1.0 - 0.05) / train_cfg["epsilon_decay_steps"]
    episode_count = 0
    loss_log      = []

    resume_step = 0
    latest_ckpt = None
    if os.path.exists(ckpt_dir):
        ckpts = sorted([f for f in os.listdir(ckpt_dir) if f.endswith(".pt")])
        if ckpts:
            latest_ckpt = os.path.join(ckpt_dir, ckpts[-1])

    if latest_ckpt:
        print(f"  Resuming from: {latest_ckpt}")
        ckpt_data = torch.load(latest_ckpt, map_location=device, weights_only=True)
        for i, net in enumerate(agent_nets):
            net.load_state_dict(ckpt_data["agent_states"][i])
        mixer.load_state_dict(ckpt_data["mixer_state"])
        optimizer.load_state_dict(ckpt_data["optimizer_state"])
        for i in range(n):
            target_nets[i].load_state_dict(agent_nets[i].state_dict())
        target_mixer.load_state_dict(mixer.state_dict())
        resume_step = ckpt_data.get("step", 0)
        steps_decayed = min(resume_step, train_cfg["epsilon_decay_steps"])
        epsilon = max(0.05, 1.0 - (0.95 * steps_decayed / train_cfg["epsilon_decay_steps"]))
        print(f"  Resumed from step {resume_step:,} | epsilon={epsilon:.4f}\n")
    else:
        epsilon = 1.0

    # ── Training loop ─────────────────────────────────────────────────────────
    print(f"  Training started...\n")
    # for step in range(1, train_cfg["total_steps"] + 1):
    for step in range(resume_step + 1, train_cfg["total_steps"] + 1):

        int_actions = select_actions(obs_array, agent_nets, epsilon, device)
        vel_actions = ACTION_MAP[int_actions]

        next_obs, rewards, done, _, info = env.step(vel_actions)
        collisions = int(info.get("collisions", 0))

        # Stay penalty
        stay_mask = (int_actions == 0).astype(np.float32)
        rewards  -= 0.05 * stay_mask

        # Spread reward
        positions = next_obs[:, :2]
        for i in range(n):
            dists = [np.linalg.norm(positions[i] - positions[j])
                     for j in range(n) if j != i]
            rewards[i] += 0.02 * (min(dists) if dists else 0.0)

        rewards = rewards / 2.0

        buffer.add(obs_array, next_obs, int_actions, rewards, done)
        metrics.update(step, info,
                       collisions_this_step=collisions,
                       agent_positions=env.agent_pos)
        obs_array = next_obs

        if len(buffer) >= train_cfg["train_start"]:
            batch = buffer.sample(train_cfg["batch_size"])
            loss  = compute_qmix_loss(
                batch, agent_nets, target_nets,
                mixer, target_mixer, 0.99, device
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
            optimizer.step()
            loss_log.append(loss.item())

        if step % train_cfg["target_update_freq"] == 0:
            for i in range(n):
                target_nets[i].load_state_dict(agent_nets[i].state_dict())
            target_mixer.load_state_dict(mixer.state_dict())

        epsilon = max(0.05, epsilon - epsilon_delta)

        if done:
            episode_count += 1
            metrics.episode_end(episode_count)

            if episode_count % 20 == 0:
                recent   = metrics.episodes[-20:]
                avg_cov  = np.mean([e["coverage_pct"]     for e in recent])
                avg_det  = np.mean([e["detection_rate"]   for e in recent])
                avg_col  = np.mean([e["total_collisions"] for e in recent])
                avg_loss = np.mean(loss_log[-200:]) if loss_log else 0.0
                print(f"  Step {step:>7,} | Ep {episode_count:>4} | "
                      f"ε={epsilon:.3f} | cov={avg_cov:5.1f}% | "
                      f"det={avg_det:.2f} | col={avg_col:6.1f} | "
                      f"loss={avg_loss:.5f}")

            obs_array, _ = env.reset()
            metrics.episode_start()

        if step % train_cfg["checkpoint_freq"] == 0:
            save_checkpoint(step, agent_nets, mixer, optimizer, ckpt_dir)

    # Save final checkpoint
    save_checkpoint(train_cfg["total_steps"], agent_nets, mixer, optimizer,
                    ckpt_dir, tag="_final")
    print(f"\n  Training complete.")

    # ── Evaluation ────────────────────────────────────────────────────────────
    print(f"\n  Evaluating {n_eval_episodes} episodes...\n")
    eval_metrics = MetricsTracker()

    for ep in range(1, n_eval_episodes + 1):
        obs_array, _ = env.reset()
        eval_metrics.episode_start()
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
            eval_metrics.update(env.step_count, info,
                                int(info.get("collisions", 0)),
                                agent_positions=env.agent_pos)

        eval_metrics.episode_end(ep)
        cov = eval_metrics.episodes[-1]["coverage_pct"]
        det = eval_metrics.episodes[-1]["detection_rate"]
        col = eval_metrics.episodes[-1]["total_collisions"]
        print(f"    ep {ep:>3}/{n_eval_episodes} | "
              f"cov={cov:5.1f}% | det={det:.2f} | col={col}")

    # Save CSV to results/qmix/SWARN-SX/metrics.csv
    csv_path = os.path.join(results_dir, "metrics.csv")
    eval_metrics.save_csv(csv_path)

    covs = [e["coverage_pct"]    for e in eval_metrics.episodes]
    dets = [e["detection_rate"]  for e in eval_metrics.episodes]
    cols = [e["total_collisions"] for e in eval_metrics.episodes]
    print(f"\n  ── {config_code} Results ──")
    print(f"  Coverage:   {np.mean(covs):.1f}% ± {np.std(covs):.1f}")
    print(f"  Detection:  {np.mean(dets):.3f} ± {np.std(dets):.3f}")
    print(f"  Collisions: {np.mean(cols):.1f} ± {np.std(cols):.1f}")
    print(f"  CSV saved → {csv_path}\n")

    # Free GPU memory before next config
    del agent_nets, target_nets, mixer, target_mixer, optimizer, buffer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def eval_only(config_code: str, ckpt_path: str, n_eval_episodes: int = 30):
    """Re-run eval only on an existing checkpoint."""
    env_cfg   = get_config(config_code)
    n         = env_cfg["n_agents"]
    obs_dim   = 29
    state_dim = obs_dim * n
    train_cfg = get_train_cfg(n, config_code)
    embed_dim = train_cfg["embed_dim"]

    device = (torch.device("cuda")
              if torch.cuda.is_available() and n <= VRAM_SAFE_MAX_AGENTS
              else torch.device("cpu"))

    agent_nets = [AgentQNetwork(obs_dim, N_ACTIONS, hidden=128).to(device) for _ in range(n)]
    mixer      = QMIXMixer(n, state_dim, embed_dim).to(device)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    for i, net in enumerate(agent_nets):
        net.load_state_dict(ckpt["agent_states"][i])
    mixer.load_state_dict(ckpt["mixer_state"])
    for net in agent_nets:
        net.eval()
    mixer.eval()

    print(f"  Loaded: {ckpt_path}")
    print(f"  Eval only mode — {config_code} | {n_eval_episodes} episodes\n")

    env          = SwarmEnv(env_cfg)
    eval_metrics = MetricsTracker()

    for ep in range(1, n_eval_episodes + 1):
        obs_array, _ = env.reset()
        eval_metrics.episode_start()
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
            eval_metrics.update(env.step_count, info,
                                int(info.get("collisions", 0)),
                                agent_positions=env.agent_pos)

        eval_metrics.episode_end(ep)
        cov = eval_metrics.episodes[-1]["coverage_pct"]
        det = eval_metrics.episodes[-1]["detection_rate"]
        print(f"    ep {ep:>3}/{n_eval_episodes} | cov={cov:5.1f}% | det={det:.2f}")

    results_dir = os.path.join("results", "qmix", config_code)
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, "metrics.csv")
    eval_metrics.save_csv(csv_path)
    print(f"  CSV saved → {csv_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",    type=str, default=None,
                        help="e.g. SWARN-S1")
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--ckpt",      type=str, default=None)
    parser.add_argument("--episodes",  type=int, default=30)
    parser.add_argument("--list",      action="store_true",
                        help="List all configs and exit")
    args = parser.parse_args()

    if args.list:
        from env.benchmark_configs import list_configs
        list_configs()
        return

    if not args.config:
        parser.error("--config required (e.g. --config SWARN-S1)")

    if args.eval_only:
        if not args.ckpt:
            parser.error("--ckpt required with --eval_only")
        eval_only(args.config, args.ckpt, args.episodes)
    else:
        train_and_eval(args.config, args.episodes)


if __name__ == "__main__":
    main()
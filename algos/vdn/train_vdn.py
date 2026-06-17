# algos/vdn/train_vdn.py

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from env.swarm_env           import SwarmEnv
from utils.metrics           import MetricsTracker
from algos.vdn.networks      import AgentQNetwork
from algos.qmix.replay_buffer import ReplayBuffer  # reuse QMIX buffer

# Same action map as QMIX
ACTION_MAP = np.array([
    [ 0.0,  0.0],
    [ 0.0,  2.0],
    [ 0.0, -2.0],
    [-2.0,  0.0],
    [ 2.0,  0.0],
    [ 1.4,  1.4],
    [-1.4,  1.4],
    [ 1.4, -1.4],
    [-1.4, -1.4],
], dtype=np.float32)

N_ACTIONS = len(ACTION_MAP)


def select_actions(obs_array, agent_nets, epsilon, device):
    n       = len(agent_nets)
    actions = np.zeros(n, dtype=np.int64)
    for i, net in enumerate(agent_nets):
        if np.random.random() < epsilon:
            actions[i] = np.random.randint(1, N_ACTIONS)
        else:
            obs_t = torch.FloatTensor(obs_array[i]).unsqueeze(0).to(device)
            with torch.no_grad():
                q_vals = net(obs_t).squeeze(0)
            best = q_vals.argmax().item()
            if best == 0:
                best = q_vals[1:].argmax().item() + 1
            actions[i] = best
    return actions


def compute_vdn_loss(batch, agent_nets, target_nets, gamma, device):
    """
    VDN loss — identical to QMIX except:
    joint Q = sum of individual Q(a_i) instead of mixer network.
    This is the ONLY difference from QMIX.
    """
    obs      = torch.FloatTensor(batch["obs"]).to(device)       # [B, n, obs]
    next_obs = torch.FloatTensor(batch["next_obs"]).to(device)
    actions  = torch.LongTensor(batch["actions"]).to(device)    # [B, n]
    rewards  = torch.FloatTensor(batch["rewards"]).to(device)   # [B, n]
    dones    = torch.FloatTensor(batch["dones"]).to(device)     # [B]

    B, n_agents, _ = obs.shape

    # Current Q-values for chosen actions
    current_qs = []
    for i, net in enumerate(agent_nets):
        q_all = net(obs[:, i, :])                    # [B, n_actions]
        q_a   = q_all.gather(1, actions[:, i:i+1])  # [B, 1]
        current_qs.append(q_a)

    # VDN: joint Q = simple sum
    q_total = torch.cat(current_qs, dim=1).sum(dim=1, keepdim=True)  # [B, 1]

    # Target Q-values (Double DQN)
    with torch.no_grad():
        target_qs = []
        for i, (net, tnet) in enumerate(zip(agent_nets, target_nets)):
            best_a = net(next_obs[:, i, :]).argmax(dim=1, keepdim=True)
            q_max  = tnet(next_obs[:, i, :]).gather(1, best_a)
            target_qs.append(q_max)

        # VDN: target joint Q = sum of target Q_i
        q_total_next = torch.cat(target_qs, dim=1).sum(dim=1, keepdim=True)

        team_reward = rewards.mean(dim=1, keepdim=True)
        targets     = team_reward + gamma * q_total_next * (1.0 - dones.unsqueeze(1))
        targets     = targets.clamp(-10.0, 10.0)

    return nn.SmoothL1Loss()(q_total, targets)


def save_checkpoint(step, agent_nets, optimizer, save_dir, tag=""):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"vdn_step_{step}{tag}.pt")
    torch.save({
        "step":            step,
        "agent_states":    [n.state_dict() for n in agent_nets],
        "optimizer_state": optimizer.state_dict(),
    }, path)
    print(f"  [checkpoint] saved vdn_step_{step}{tag}.pt")


def train(config: dict = None):
    cfg = {
        "n_agents":            6,
        "map_size":            50,
        "n_targets":           3,
        "n_obstacles":         5,
        "max_steps":           300,
        "comm_dropout":        0.0,
        "detect_radius":       2.0,
        "coverage_mode":       "sensed",
        "total_steps":         200_000,
        "batch_size":          16,
        "lr":                  5e-5,
        "gamma":               0.99,
        "epsilon_decay_steps": 120_000,
        "target_update_freq":  500,
        "buffer_size":         20_000,
        "train_start":         2_000,
        "checkpoint_freq":     50_000,
        "checkpoint_dir":      "results/vdn/checkpoints",
        "results_dir":         "results/vdn",
        "obs_dim":             29,
    }
    if config:
        cfg.update(config)

    n_agents     = cfg.get("n_agents", 6)
    vram_max     = cfg.get("vram_safe_max_agents", 50)
    use_cuda     = torch.cuda.is_available() and n_agents <= vram_max
    device       = torch.device("cuda" if use_cuda else "cpu")
    print(f"  Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        torch.cuda.set_per_process_memory_fraction(0.85)

    env = SwarmEnv({
        "n_agents":      cfg["n_agents"],
        "map_size":      cfg["map_size"],
        "n_targets":     cfg["n_targets"],
        "n_obstacles":   cfg["n_obstacles"],
        "max_steps":     cfg["max_steps"],
        "comm_dropout":  cfg["comm_dropout"],
        "detect_radius": cfg["detect_radius"],
        "coverage_mode": cfg["coverage_mode"],
    })

    n       = cfg["n_agents"]
    obs_dim = cfg["obs_dim"]

    agent_nets  = [AgentQNetwork(obs_dim, N_ACTIONS, hidden=128).to(device) for _ in range(n)]
    target_nets = [AgentQNetwork(obs_dim, N_ACTIONS, hidden=128).to(device) for _ in range(n)]

    for i in range(n):
        target_nets[i].load_state_dict(agent_nets[i].state_dict())

    all_params = []
    for net in agent_nets:
        all_params += list(net.parameters())
    optimizer = optim.Adam(all_params, lr=cfg["lr"])

    buffer  = ReplayBuffer(cfg["buffer_size"], n, obs_dim, N_ACTIONS)
    metrics = MetricsTracker()

    # Auto-resume from latest checkpoint
    ckpt_dir    = cfg["checkpoint_dir"]
    resume_step = 0
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpts = sorted(
        [f for f in os.listdir(ckpt_dir) if f.endswith(".pt")],
        key=lambda x: int(''.join(filter(str.isdigit, x)) or 0)
    )
    if ckpts:
        latest = os.path.join(ckpt_dir, ckpts[-1])
        print(f"  Resuming from: {latest}")
        ckpt_data = torch.load(latest, map_location=device, weights_only=True)
        for i, net in enumerate(agent_nets):
            net.load_state_dict(ckpt_data["agent_states"][i])
        optimizer.load_state_dict(ckpt_data["optimizer_state"])
        for i in range(n):
            target_nets[i].load_state_dict(agent_nets[i].state_dict())
        resume_step   = ckpt_data.get("step", 0)
        steps_decayed = min(resume_step, cfg["epsilon_decay_steps"])
        epsilon       = max(0.05, 1.0 - 0.95 * steps_decayed / cfg["epsilon_decay_steps"])
        print(f"  Resumed step {resume_step:,} | epsilon={epsilon:.4f}")
    else:
        epsilon = 1.0

    epsilon_delta = 0.95 / cfg["epsilon_decay_steps"]
    episode_count = 0
    loss_log      = []

    obs_array, _ = env.reset()
    metrics.episode_start()

    print(f"\n  VDN training | {cfg['total_steps']:,} steps | {n} agents\n")

    for step in range(resume_step + 1, cfg["total_steps"] + 1):

        int_actions = select_actions(obs_array, agent_nets, epsilon, device)
        vel_actions = ACTION_MAP[int_actions]

        next_obs, rewards, done, _, info = env.step(vel_actions)
        collisions = int(info.get("collisions", 0))

        stay_mask = (int_actions == 0).astype(np.float32)
        rewards  -= 0.05 * stay_mask

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

        if len(buffer) >= cfg["train_start"]:
            batch = buffer.sample(cfg["batch_size"])
            loss  = compute_vdn_loss(
                batch, agent_nets, target_nets, cfg["gamma"], device
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
            optimizer.step()
            loss_log.append(loss.item())

        if step % cfg["target_update_freq"] == 0:
            for i in range(n):
                target_nets[i].load_state_dict(agent_nets[i].state_dict())

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

        if step % cfg["checkpoint_freq"] == 0:
            save_checkpoint(step, agent_nets, optimizer, ckpt_dir)

    save_checkpoint(cfg["total_steps"], agent_nets, optimizer, ckpt_dir, tag="_final")
    print(f"\n  Training complete.")
    return agent_nets
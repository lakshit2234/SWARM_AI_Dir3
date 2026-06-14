# # algos/qmix/train_qmix.py

# import os
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim

# from env.swarm_env import SwarmEnv
# from utils.metrics import MetricsTracker
# from algos.qmix.networks import AgentQNetwork, QMIXMixer
# from algos.qmix.replay_buffer import ReplayBuffer


# # ── Action mapping ────────────────────────────────────────────────────────────
# # QMIX uses discrete actions. We map each integer to a 2D velocity vector.
# ACTION_MAP = np.array([
#     [ 0.0,  0.0],   # 0 = stay
#     [ 0.0,  1.0],   # 1 = up
#     [ 0.0, -1.0],   # 2 = down
#     [-1.0,  0.0],   # 3 = left
#     [ 1.0,  0.0],   # 4 = right
# ], dtype=np.float32)

# N_ACTIONS = len(ACTION_MAP)


# def select_actions(
#     obs_array:    np.ndarray,        # [n_agents, 29]
#     agent_nets:   list,
#     epsilon:      float,
#     device:       torch.device,
# ) -> np.ndarray:
#     """
#     Epsilon-greedy action selection.
#     With probability epsilon: random action (exploration).
#     Otherwise: action with highest Q-value (exploitation).
#     Returns integer action indices, shape [n_agents].
#     """
#     n = len(agent_nets)
#     actions = np.zeros(n, dtype=np.int64)

#     for i, net in enumerate(agent_nets):
#         if np.random.random() < epsilon:
#             actions[i] = np.random.randint(N_ACTIONS)
#         else:
#             obs_t = torch.FloatTensor(obs_array[i]).unsqueeze(0).to(device)
#             with torch.no_grad():
#                 q_vals = net(obs_t)             # [1, n_actions]
#             actions[i] = q_vals.argmax(dim=1).item()

#     return actions


# def compute_qmix_loss(
#     batch:         dict,
#     agent_nets:    list,
#     target_nets:   list,
#     mixer:         QMIXMixer,
#     target_mixer:  QMIXMixer,
#     gamma:         float,
#     device:        torch.device,
# ) -> torch.Tensor:
#     """
#     Bellman update for QMIX.

#     1. Get current Q-values for chosen actions (one per agent)
#     2. Get target Q-values using target networks (max over actions)
#     3. Mix both sets using their respective mixer networks
#     4. TD error = (r + gamma * target_q_total - current_q_total)^2
#     """
#     obs      = torch.FloatTensor(batch["obs"]).to(device)       # [B, n, obs]
#     next_obs = torch.FloatTensor(batch["next_obs"]).to(device)  # [B, n, obs]
#     actions  = torch.LongTensor(batch["actions"]).to(device)    # [B, n]
#     rewards  = torch.FloatTensor(batch["rewards"]).to(device)   # [B, n]
#     dones    = torch.FloatTensor(batch["dones"]).to(device)     # [B]

#     B, n_agents, obs_dim = obs.shape

#     # ── Current Q-values ─────────────────────────────────────────────────────
#     current_qs = []
#     for i, net in enumerate(agent_nets):
#         q_all = net(obs[:, i, :])                   # [B, n_actions]
#         q_a   = q_all.gather(1, actions[:, i:i+1])  # [B, 1]  — Q of chosen action
#         current_qs.append(q_a)

#     current_qs = torch.cat(current_qs, dim=1)       # [B, n_agents]

#     # Global state = all agents' observations concatenated
#     states = obs.view(B, -1)                        # [B, n_agents * obs_dim]

#     q_total = mixer(current_qs, states)             # [B, 1]

#     # ── Target Q-values ───────────────────────────────────────────────────────
#     with torch.no_grad():
#         target_qs = []
#         for i, tnet in enumerate(target_nets):
#             q_all   = tnet(next_obs[:, i, :])       # [B, n_actions]
#             q_max   = q_all.max(dim=1, keepdim=True)[0]  # [B, 1]
#             target_qs.append(q_max)

#         target_qs    = torch.cat(target_qs, dim=1)  # [B, n_agents]
#         next_states  = next_obs.view(B, -1)
#         q_total_next = target_mixer(target_qs, next_states)  # [B, 1]

#         # Team reward = mean of individual rewards
#         team_reward = rewards.mean(dim=1, keepdim=True)      # [B, 1]

#         # Bellman target
#         targets = team_reward + gamma * q_total_next * (1 - dones.unsqueeze(1))

#     loss = nn.MSELoss()(q_total, targets)
#     return loss


# def save_checkpoint(
#     step:       int,
#     agent_nets: list,
#     mixer:      QMIXMixer,
#     optimizer:  optim.Optimizer,
#     save_dir:   str,
#     tag:        str = "",
# ):
#     os.makedirs(save_dir, exist_ok=True)
#     fname = f"qmix_step_{step}{tag}.pt"
#     torch.save({
#         "step":           step,
#         "agent_states":   [n.state_dict() for n in agent_nets],
#         "mixer_state":    mixer.state_dict(),
#         "optimizer_state": optimizer.state_dict(),
#     }, os.path.join(save_dir, fname))
#     print(f"  [checkpoint] saved {fname}")


# def train(config: dict = None):
#     """
#     Full QMIX training loop.

#     Pass a config dict to override defaults, e.g.:
#         train({"scenario": "easy", "n_agents": 5})
#     """

#     # ── Defaults ─────────────────────────────────────────────────────────────
#     cfg = {
#         # Environment
#         "scenario":           "easy",
#         "n_agents":           6,       # DO NOT increase beyond 6 on RTX 4050
#         "map_size":           50,
#         "n_targets":          3,
#         "n_obstacles":        5,
#         "max_steps":          300,
#         "comm_dropout":       0.0,
#         # Training
#         "total_steps":        200_000,
#         "batch_size":         64,
#         "lr":                 5e-4,
#         "gamma":              0.99,
#         "epsilon_start":      1.0,
#         "epsilon_end":        0.05,
#         "epsilon_decay_steps": 50_000,
#         "target_update_freq": 200,
#         "buffer_size":        5000,
#         "train_start":        500,     # don't train until buffer has this many
#         "checkpoint_freq":    10_000,
#         "checkpoint_dir":     "results/qmix/checkpoints",
#         "results_dir":        "results/qmix",
#         "obs_dim":            29,
#         "embed_dim":          32,
#     }
#     if config:
#         cfg.update(config)

#     # ── Device ───────────────────────────────────────────────────────────────
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"  Training on: {device}")
#     if device.type == "cuda":
#         print(f"  GPU: {torch.cuda.get_device_name(0)}")

#     # ── Environment ──────────────────────────────────────────────────────────
#     env = SwarmEnv({
#         "n_agents":     cfg["n_agents"],
#         "map_size":     cfg["map_size"],
#         "n_targets":    cfg["n_targets"],
#         "n_obstacles":  cfg["n_obstacles"],
#         "max_steps":    cfg["max_steps"],
#         "comm_dropout": cfg["comm_dropout"],
#     })

#     n       = cfg["n_agents"]
#     obs_dim = cfg["obs_dim"]
#     state_dim = obs_dim * n   # global state = all obs concatenated

#     # ── Networks ─────────────────────────────────────────────────────────────
#     agent_nets  = [AgentQNetwork(obs_dim, N_ACTIONS, hidden=64).to(device) for _ in range(n)]
#     target_nets = [AgentQNetwork(obs_dim, N_ACTIONS, hidden=64).to(device) for _ in range(n)]
#     mixer        = QMIXMixer(n, state_dim, cfg["embed_dim"]).to(device)
#     target_mixer = QMIXMixer(n, state_dim, cfg["embed_dim"]).to(device)

#     # Initialise target networks as copies of online networks
#     for i in range(n):
#         target_nets[i].load_state_dict(agent_nets[i].state_dict())
#     target_mixer.load_state_dict(mixer.state_dict())

#     # Single optimiser over all parameters
#     all_params = list(mixer.parameters())
#     for net in agent_nets:
#         all_params += list(net.parameters())
#     optimizer = optim.Adam(all_params, lr=cfg["lr"])

#     # ── Replay Buffer ────────────────────────────────────────────────────────
#     buffer = ReplayBuffer(cfg["buffer_size"], n, obs_dim, N_ACTIONS)

#     # ── Metrics ──────────────────────────────────────────────────────────────
#     metrics       = MetricsTracker()
#     episode_count = 0
#     total_reward  = 0.0
#     loss_log      = []

#     # ── Epsilon schedule ─────────────────────────────────────────────────────
#     epsilon_delta = (cfg["epsilon_start"] - cfg["epsilon_end"]) / cfg["epsilon_decay_steps"]
#     epsilon       = cfg["epsilon_start"]

#     # ── Reset environment ────────────────────────────────────────────────────
#     obs_array, _ = env.reset()   # [n_agents, 29]
#     metrics.episode_start()

#     print(f"\n  Starting QMIX training: {cfg['total_steps']:,} steps, {n} agents, scenario={cfg['scenario']}")
#     print(f"  Buffer warms up for {cfg['train_start']} steps before training begins.\n")

#     # ── Main loop ────────────────────────────────────────────────────────────
#     for step in range(1, cfg["total_steps"] + 1):

#         # Select actions (epsilon-greedy)
#         int_actions = select_actions(obs_array, agent_nets, epsilon, device)

#         # Convert integer actions → 2D velocity vectors for the env
#         vel_actions = ACTION_MAP[int_actions]   # [n_agents, 2]

#         # Step the environment
#         next_obs_array, rewards, done, _, info = env.step(vel_actions)

#         # Store transition
#         buffer.add(obs_array, next_obs_array, int_actions, rewards, done)

#         total_reward += rewards.mean()

#         # Update metrics tracker
#         collisions = int(info.get("collisions", 0))
#         metrics.update(step, info, collisions_this_step=collisions)

#         obs_array = next_obs_array

#         # ── Train ─────────────────────────────────────────────────────────────
#         if len(buffer) >= cfg["train_start"]:
#             batch = buffer.sample(cfg["batch_size"])
#             loss  = compute_qmix_loss(
#                 batch, agent_nets, target_nets,
#                 mixer, target_mixer, cfg["gamma"], device
#             )
#             optimizer.zero_grad()
#             loss.backward()
#             # Gradient clipping prevents exploding gradients
#             torch.nn.utils.clip_grad_norm_(all_params, max_norm=10.0)
#             optimizer.step()
#             loss_log.append(loss.item())

#         # ── Update target networks ─────────────────────────────────────────
#         if step % cfg["target_update_freq"] == 0:
#             for i in range(n):
#                 target_nets[i].load_state_dict(agent_nets[i].state_dict())
#             target_mixer.load_state_dict(mixer.state_dict())

#         # ── Epsilon decay ─────────────────────────────────────────────────
#         epsilon = max(cfg["epsilon_end"], epsilon - epsilon_delta)

#         # ── Episode end ───────────────────────────────────────────────────
#         if done:
#             episode_count += 1
#             metrics.episode_end(episode_count)

#             if episode_count % 10 == 0:
#                 recent = metrics.episodes[-10:]
#                 avg_cov = np.mean([e["coverage_pct"]   for e in recent])
#                 avg_det = np.mean([e["detection_rate"]  for e in recent])
#                 avg_col = np.mean([e["total_collisions"] for e in recent])
#                 avg_loss = np.mean(loss_log[-100:]) if loss_log else 0.0
#                 print(
#                     f"  Step {step:>7,} | Ep {episode_count:>4} | "
#                     f"ε={epsilon:.3f} | "
#                     f"cov={avg_cov:5.1f}% | "
#                     f"det={avg_det:.2f} | "
#                     f"col={avg_col:.1f} | "
#                     f"loss={avg_loss:.4f}"
#                 )

#             obs_array, _ = env.reset()
#             metrics.episode_start()
#             total_reward = 0.0

#         # ── Checkpoint ────────────────────────────────────────────────────
#         if step % cfg["checkpoint_freq"] == 0:
#             save_checkpoint(step, agent_nets, mixer, optimizer, cfg["checkpoint_dir"])

#     # ── Save final results ────────────────────────────────────────────────────
#     scenario = cfg["scenario"]
#     csv_path = os.path.join(cfg["results_dir"], f"metrics_{scenario}.csv")
#     metrics.save_csv(csv_path)
#     print(f"\n  Training complete. Results saved to {csv_path}")

#     return agent_nets, mixer





# algos/qmix/train_qmix.py

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from env.swarm_env import SwarmEnv
from utils.metrics import MetricsTracker
from algos.qmix.networks import AgentQNetwork, QMIXMixer
from algos.qmix.replay_buffer import ReplayBuffer


# ── Action map — 9 directions, step size 3.0 ─────────────────────────────────
# ACTION_MAP = np.array([
#     [ 0.0,  0.0],
#     [ 0.0,  3.0],
#     [ 0.0, -3.0],
#     [-3.0,  0.0],
#     [ 3.0,  0.0],
#     [ 2.1,  2.1],
#     [-2.1,  2.1],
#     [ 2.1, -2.1],
#     [-2.1, -2.1],
# ], dtype=np.float32)



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


# def select_actions(obs_array, agent_nets, epsilon, device):
#     n       = len(agent_nets)
#     actions = np.zeros(n, dtype=np.int64)
#     for i, net in enumerate(agent_nets):
#         if np.random.random() < epsilon:
#             actions[i] = np.random.randint(1, N_ACTIONS)  # never "stay" during explore
#         else:
#             obs_t = torch.FloatTensor(obs_array[i]).unsqueeze(0).to(device)
#             with torch.no_grad():
#                 actions[i] = net(obs_t).argmax(dim=1).item()
#     return actions



def select_actions(obs_array, agent_nets, epsilon, device):
    """
    Epsilon-greedy with two improvements:
    1. Never pick 'stay' (action 0) during random exploration
    2. During greedy phase, if Q-values of all movement actions
       are within 0.01 of each other (agent is confused),
       pick randomly from movement actions instead of defaulting
       to stay — prevents idle clustering
    """
    n       = len(agent_nets)
    actions = np.zeros(n, dtype=np.int64)

    for i, net in enumerate(agent_nets):
        if np.random.random() < epsilon:
            # Always pick a movement action during exploration
            actions[i] = np.random.randint(1, N_ACTIONS)
        else:
            obs_t = torch.FloatTensor(obs_array[i]).unsqueeze(0).to(device)
            with torch.no_grad():
                q_vals = net(obs_t).squeeze(0)  # shape: [N_ACTIONS]

            # If the best action is "stay", override with best movement action
            best = q_vals.argmax().item()
            if best == 0:
                best = q_vals[1:].argmax().item() + 1

            actions[i] = best

    return actions


def compute_qmix_loss(
    batch, agent_nets, target_nets,
    mixer, target_mixer, gamma, device
):
    obs      = torch.FloatTensor(batch["obs"]).to(device)
    next_obs = torch.FloatTensor(batch["next_obs"]).to(device)
    actions  = torch.LongTensor(batch["actions"]).to(device)
    rewards  = torch.FloatTensor(batch["rewards"]).to(device)
    dones    = torch.FloatTensor(batch["dones"]).to(device)

    B, n_agents, _ = obs.shape

    # ── Current Q-values ─────────────────────────────────────────────────────
    current_qs = []
    for i, net in enumerate(agent_nets):
        q_all = net(obs[:, i, :])
        q_a   = q_all.gather(1, actions[:, i:i+1])
        current_qs.append(q_a)
    current_qs = torch.cat(current_qs, dim=1)           # [B, n_agents]
    q_total    = mixer(current_qs, obs.view(B, -1))     # [B, 1]

    # ── Target Q-values (Double DQN) ─────────────────────────────────────────
    with torch.no_grad():
        target_qs = []
        for i, (net, tnet) in enumerate(zip(agent_nets, target_nets)):
            best_a  = net(next_obs[:, i, :]).argmax(dim=1, keepdim=True)
            q_max   = tnet(next_obs[:, i, :]).gather(1, best_a)
            target_qs.append(q_max)
        target_qs    = torch.cat(target_qs, dim=1)
        q_total_next = target_mixer(target_qs, next_obs.view(B, -1))

        # Mean team reward — keeps target scale bounded
        team_reward = rewards.mean(dim=1, keepdim=True)

        # Clamp targets to a reasonable range — prevents Q-value explosion
        targets = team_reward + gamma * q_total_next * (1.0 - dones.unsqueeze(1))
        targets = targets.clamp(-10.0, 10.0)

    return nn.SmoothL1Loss()(q_total, targets)   # Huber loss — far more stable than MSE


def save_checkpoint(step, agent_nets, mixer, optimizer, save_dir, tag=""):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"qmix_step_{step}{tag}.pt")
    torch.save({
        "step":            step,
        "agent_states":    [n.state_dict() for n in agent_nets],
        "mixer_state":     mixer.state_dict(),
        "optimizer_state": optimizer.state_dict(),
    }, path)
    print(f"  [checkpoint] saved qmix_step_{step}{tag}.pt")


def train(config: dict = None):
    cfg = {
        "scenario":            "easy",
        "n_agents":            6,
        "map_size":            50,
        "n_targets":           3,
        "n_obstacles":         5,
        "max_steps":           300,
        "comm_dropout":        0.0,
        # Training — conservative stable settings
        "total_steps":         500_000,
        "batch_size":          64,
        "lr":                  5e-5,       # KEY FIX: was 3e-4, now 10x lower
        "gamma":               0.99,
        "epsilon_start":       1.0,
        "epsilon_end":         0.05,
        "epsilon_decay_steps": 200_000,
        "target_update_freq":  500,
        "buffer_size":         20_000,
        "train_start":         2000,
        "checkpoint_freq":     50_000,
        "checkpoint_dir":      "results/qmix/checkpoints",
        "results_dir":         "results/qmix",
        "obs_dim":             29,
        "embed_dim":           64,
    }
    if config:
        cfg.update(config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Training on: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    env = SwarmEnv({
        "n_agents":     cfg["n_agents"],
        "map_size":     cfg["map_size"],
        "n_targets":    cfg["n_targets"],
        "n_obstacles":  cfg["n_obstacles"],
        "max_steps":    cfg["max_steps"],
        "comm_dropout": cfg["comm_dropout"],
    })

    n         = cfg["n_agents"]
    obs_dim   = cfg["obs_dim"]
    state_dim = obs_dim * n

    agent_nets   = [AgentQNetwork(obs_dim, N_ACTIONS, hidden=128).to(device) for _ in range(n)]
    target_nets  = [AgentQNetwork(obs_dim, N_ACTIONS, hidden=128).to(device) for _ in range(n)]
    mixer        = QMIXMixer(n, state_dim, cfg["embed_dim"]).to(device)
    target_mixer = QMIXMixer(n, state_dim, cfg["embed_dim"]).to(device)

    # for i in range(n):
    #     target_nets[i].load_state_dict(agent_nets[i].state_dict())
    # target_mixer.load_state_dict(mixer.state_dict())

    # all_params = list(mixer.parameters())
    # for net in agent_nets:
    #     all_params += list(net.parameters())

    # optimizer = optim.Adam(all_params, lr=cfg["lr"])


    for i in range(n):
        target_nets[i].load_state_dict(agent_nets[i].state_dict())
    target_mixer.load_state_dict(mixer.state_dict())

    all_params = list(mixer.parameters())
    for net in agent_nets:
        all_params += list(net.parameters())

    optimizer = optim.Adam(all_params, lr=cfg["lr"])

    # ── Resume from checkpoint if specified ───────────────────────────────────
    resume_step = 0
    if cfg.get("resume_checkpoint"):
        ckpt = torch.load(
            cfg["resume_checkpoint"],
            map_location=device,
            weights_only=True
        )
        for i, net in enumerate(agent_nets):
            net.load_state_dict(ckpt["agent_states"][i])
        mixer.load_state_dict(ckpt["mixer_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])

        # Sync target networks with loaded weights
        for i in range(n):
            target_nets[i].load_state_dict(agent_nets[i].state_dict())
        target_mixer.load_state_dict(mixer.state_dict())

        resume_step = ckpt.get("step", 0)
        print(f"  Resumed from step {resume_step:,}")
        print(f"  Epsilon will continue from where training left off.\n")

    buffer  = ReplayBuffer(cfg["buffer_size"], n, obs_dim, N_ACTIONS)
    metrics = MetricsTracker()

    episode_count = 0
    loss_log      = []
    epsilon_delta = (cfg["epsilon_start"] - cfg["epsilon_end"]) / cfg["epsilon_decay_steps"]
    # epsilon       = cfg["epsilon_start"]

    # If resuming, set epsilon to where it was at resume_step
    if resume_step > 0:
        steps_already_decayed = min(resume_step, cfg["epsilon_decay_steps"])
        epsilon = max(
            cfg["epsilon_end"],
            cfg["epsilon_start"] - (cfg["epsilon_start"] - cfg["epsilon_end"])
            * steps_already_decayed / cfg["epsilon_decay_steps"]
        )
        print(f"  Epsilon restored to: {epsilon:.4f}")
    else:
        epsilon = cfg["epsilon_start"]

    obs_array, _ = env.reset()
    metrics.episode_start()

    print(f"\n  Starting QMIX | {cfg['total_steps']:,} steps | "
          f"{n} agents | scenario={cfg['scenario']}\n")

    for step in range(1, cfg["total_steps"] + 1):

        int_actions = select_actions(obs_array, agent_nets, epsilon, device)
        vel_actions = ACTION_MAP[int_actions]

        next_obs_array, rewards, done, _, info = env.step(vel_actions)
        collisions = int(info.get("collisions", 0))


        # ── Stay penalty ──────────────────────────────────────────────────────
        # Agents that chose action 0 (stay) get a small extra punishment.
        # This discourages idle clustering which is the main collision cause.
        stay_mask = (int_actions == 0).astype(np.float32)
        rewards  -= 0.05 * stay_mask

        # ── Spread reward ─────────────────────────────────────────────────────
        # Reward each agent for being far from its nearest neighbour.
        # Uses the position component of the observation (first 2 dims, normalised).
        # This directly incentivises agents to spread across the map.
        positions = next_obs_array[:, :2]  # shape: [n_agents, 2]
        for i in range(len(positions)):
            dists = [
                np.linalg.norm(positions[i] - positions[j])
                for j in range(len(positions)) if j != i
            ]
            min_dist = min(dists) if dists else 0.0
            # Scale: min_dist is normalised (0-1 range), multiply by 0.02
            # so max spread bonus is ~0.02 — small but consistent signal
            rewards[i] += 0.02 * min_dist

        # ── Reward normalisation ──────────────────────────────────────────────
        rewards = rewards / 2.0


        buffer.add(obs_array, next_obs_array, int_actions, rewards, done)
        metrics.update(step, info, collisions_this_step=collisions)
        obs_array = next_obs_array

        # ── Train ─────────────────────────────────────────────────────────────
        if len(buffer) >= cfg["train_start"]:
            batch = buffer.sample(cfg["batch_size"])
            loss  = compute_qmix_loss(
                batch, agent_nets, target_nets,
                mixer, target_mixer, cfg["gamma"], device
            )
            optimizer.zero_grad()
            loss.backward()

            # KEY FIX: tight gradient clipping — was 10.0
            torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)

            optimizer.step()
            loss_log.append(loss.item())

        if step % cfg["target_update_freq"] == 0:
            for i in range(n):
                target_nets[i].load_state_dict(agent_nets[i].state_dict())
            target_mixer.load_state_dict(mixer.state_dict())

        epsilon = max(cfg["epsilon_end"], epsilon - epsilon_delta)

        if done:
            episode_count += 1
            metrics.episode_end(episode_count)

            if episode_count % 20 == 0:
                recent   = metrics.episodes[-20:]
                avg_cov  = np.mean([e["coverage_pct"]     for e in recent])
                avg_det  = np.mean([e["detection_rate"]   for e in recent])
                avg_col  = np.mean([e["total_collisions"] for e in recent])
                avg_loss = np.mean(loss_log[-200:]) if loss_log else 0.0
                print(
                    f"  Step {step:>7,} | Ep {episode_count:>5} | "
                    f"ε={epsilon:.3f} | "
                    f"cov={avg_cov:5.1f}% | "
                    f"det={avg_det:.2f} | "
                    f"col={avg_col:6.1f} | "
                    f"loss={avg_loss:.5f}"
                )

            obs_array, _ = env.reset()
            metrics.episode_start()

        if step % cfg["checkpoint_freq"] == 0:
            save_checkpoint(
                step, agent_nets, mixer, optimizer, cfg["checkpoint_dir"]
            )

    scenario = cfg["scenario"]
    csv_path = os.path.join(cfg["results_dir"], f"metrics_{scenario}.csv")
    metrics.save_csv(csv_path)
    print(f"\n  Training complete → {csv_path}")
    return agent_nets, mixer
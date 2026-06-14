# algos/mappo/train.py
import torch
import numpy as np
from algos.mappo.env_wrapper import SwarmParallelEnv
from algos.mappo.networks import Actor, CentralCritic

# ── Config ──────────────────────────────────────────
cfg = {
    'n_agents':    10,
    'map_size':    100,
    'n_targets':   8,
    'n_obstacles': 20,
    'comm_dropout': 0.2,
    'max_steps':   500,
    'scenario':    'hard',
    'obs_radius':  10,
    'coverage_mode': 'sensed',
}
TOTAL_STEPS = 1_000_000
DEVICE       = torch.device('cuda')
N_AGENTS     = cfg['n_agents']
OBS_DIM      = 29
ACTION_DIM   = 2
HIDDEN       = 128
LR           = 5e-5
GAMMA        = 0.99
LAM          = 0.95
EPS_CLIP     = 0.15
N_EPOCHS     = 4
BATCH_SIZE   = 128
N_STEPS      = 1000
TOTAL_STEPS  = 1_000_000
STAGE        = 3

# ── Init networks ────────────────────────────────────
actor  = Actor(OBS_DIM, ACTION_DIM, HIDDEN).to(DEVICE)
critic = CentralCritic(OBS_DIM, N_AGENTS, 256).to(DEVICE)
optimizer = torch.optim.Adam(
    list(actor.parameters()) + list(critic.parameters()), lr=LR
)

# ── Load Stage 1 checkpoint ──────────────────────────
ckpt = torch.load('results/mappo/ckpt_stage1_sensed_500000.pt', map_location=DEVICE)
actor.load_state_dict(ckpt['actor'])
print("Stage 1 sensed checkpoint loaded for Stage 3!")

# ── Init env ─────────────────────────────────────────
env = SwarmParallelEnv(cfg)

print("Stage 3 retraining ready! (sensed coverage)")

# ── Rollout Buffer ───────────────────────────────────
obs_buf        = []
act_buf        = []
logp_buf       = []
rew_buf        = []
done_buf       = []
global_obs_buf = []

# ── Training Loop ────────────────────────────────────
obs, _ = env.reset()
total_steps = 0
episode     = 0

print("\nStarting training...")

while total_steps < TOTAL_STEPS:

    obs_buf.clear(); act_buf.clear(); logp_buf.clear()
    rew_buf.clear(); done_buf.clear(); global_obs_buf.clear()

    # ── Collect rollout ──
    for _ in range(N_STEPS):
        obs_tensor = torch.tensor(
            np.stack([obs[a] for a in env.agents]),
            dtype=torch.float32
        ).to(DEVICE)

        global_obs = obs_tensor.flatten().unsqueeze(0)

        with torch.no_grad():
            actions, log_probs = actor(obs_tensor)

        actions_np = actions.cpu().numpy()
        actions_dict = {a: actions_np[i] for i, a in enumerate(env.agents)}

        next_obs, rews, dones, _, info = env.step(actions_dict)

        obs_buf.append(obs_tensor)
        act_buf.append(actions)
        logp_buf.append(log_probs)
        rew_buf.append(torch.tensor(
            [rews[a] for a in env.agents], dtype=torch.float32
        ).to(DEVICE))
        done_buf.append(dones[env.agents[0]])
        global_obs_buf.append(global_obs)

        obs = next_obs
        total_steps += 1

        if dones[env.agents[0]]:
            episode += 1
            obs, _ = env.reset()

    # ── Compute returns & advantages ──
    rewards  = torch.stack(rew_buf)           # [N_STEPS, N_AGENTS]
    obs_t    = torch.stack(obs_buf)           # [N_STEPS, N_AGENTS, OBS_DIM]
    acts_t   = torch.stack(act_buf)           # [N_STEPS, N_AGENTS, 2]
    logps_t  = torch.stack(logp_buf)          # [N_STEPS, N_AGENTS]
    gobs_t   = torch.cat(global_obs_buf)      # [N_STEPS, N_AGENTS*OBS_DIM]

    with torch.no_grad():
        values = critic(gobs_t).squeeze(-1)   # [N_STEPS]

    mean_rew = rewards.mean(dim=1)            # [N_STEPS]
    advantages = torch.zeros_like(mean_rew)
    gae = 0.0
    for t in reversed(range(N_STEPS)):
        delta = mean_rew[t] + (GAMMA * values[t] * (1 - float(done_buf[t]))) - values[t]
        gae   = delta + GAMMA * LAM * (1 - float(done_buf[t])) * gae
        advantages[t] = gae
    returns = advantages + values
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # ── PPO update ──
    for _ in range(N_EPOCHS):
        idx = torch.randperm(N_STEPS)
        for start in range(0, N_STEPS, BATCH_SIZE):
            mb = idx[start:start + BATCH_SIZE]
            if len(mb) < 2:
                continue

            obs_mb    = obs_t[mb].view(-1, OBS_DIM)
            acts_mb   = acts_t[mb].view(-1, ACTION_DIM)
            logps_mb  = logps_t[mb].view(-1)
            adv_mb    = advantages[mb].unsqueeze(1).expand(-1, N_AGENTS).reshape(-1)
            ret_mb    = returns[mb]
            gobs_mb   = gobs_t[mb]

            new_logps, entropy = actor.evaluate(obs_mb, acts_mb)
            ratio  = (new_logps - logps_mb).exp()
            surr1  = ratio * adv_mb
            surr2  = ratio.clamp(1 - EPS_CLIP, 1 + EPS_CLIP) * adv_mb

            actor_loss  = -torch.min(surr1, surr2).mean()
            critic_loss = (ret_mb - critic(gobs_mb).squeeze(-1)).pow(2).mean()
            entropy_loss = -entropy.mean()
            loss = actor_loss + 0.5 * critic_loss + 0.05 * entropy_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(actor.parameters()) + list(critic.parameters()), 0.5
            )
            optimizer.step()

    if total_steps % 10_000 == 0:
        print(f"Steps: {total_steps:>7,} | Episodes: {episode:>4} | "
              f"Loss: {loss.item():.4f} | "
              f"Coverage: {info['coverage_pct']:.1f}% | "
              f"Detected: {info['detection_rate']:.2f}")
        

import os
os.makedirs('results/mappo', exist_ok=True)
torch.save({
    'step':      total_steps,
    'actor':     actor.state_dict(),
    'critic':    critic.state_dict(),
    'optimizer': optimizer.state_dict(),
    'config':    cfg,
}, 'results/mappo/ckpt_stage3_sensed_1000000.pt')
print("Stage 3 sensed checkpoint saved!")
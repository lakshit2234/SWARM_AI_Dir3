# scripts/train_mappo_all_configs.py
# Train MAPPO on SWARN-S1 through S10
import sys, os
sys.path.insert(0, '.')
import torch
import numpy as np
from env.benchmark_configs import get_config, SWARN_CONFIGS
from algos.mappo.env_wrapper import SwarmParallelEnv
from algos.mappo.networks import Actor, CentralCritic

DEVICE     = torch.device('cuda')
OBS_DIM    = 29
ACTION_DIM = 2
HIDDEN     = 128
LR         = 5e-5
GAMMA      = 0.99
LAM        = 0.95
EPS_CLIP   = 0.2
N_EPOCHS   = 4
BATCH_SIZE = 128
N_STEPS    = 500

# Steps per config based on swarm size
STEPS_MAP = {
    'SWARN-S1':  300_000,
    'SWARN-S2':  300_000,
    'SWARN-S3':  500_000,
    'SWARN-S4':  500_000,
    'SWARN-S5':  500_000,
    'SWARN-S6':  500_000,
    'SWARN-S7':  700_000,
    'SWARN-S8':  700_000,
    'SWARN-S9':  800_000,
    'SWARN-S10': 800_000,
}

def train_mappo(config_code, total_steps):
    cfg      = get_config(config_code)
    n_agents = cfg['n_agents']
    env      = SwarmParallelEnv(cfg)

    actor  = Actor(OBS_DIM, ACTION_DIM, HIDDEN).to(DEVICE)
    critic = CentralCritic(OBS_DIM, n_agents, 256).to(DEVICE)
    optimizer = torch.optim.Adam(
        list(actor.parameters()) + list(critic.parameters()), lr=LR)

    obs_buf, act_buf, logp_buf = [], [], []
    rew_buf, done_buf, gobs_buf = [], [], []

    obs, _ = env.reset()
    step = 0
    episode = 0

    while step < total_steps:
        obs_buf.clear(); act_buf.clear(); logp_buf.clear()
        rew_buf.clear(); done_buf.clear(); gobs_buf.clear()

        for _ in range(N_STEPS):
            obs_tensor = torch.tensor(
                np.stack([obs[a] for a in env.agents]),
                dtype=torch.float32).to(DEVICE)
            global_obs = obs_tensor.flatten().unsqueeze(0)

            with torch.no_grad():
                actions, log_probs = actor(obs_tensor)

            actions_np   = actions.cpu().numpy()
            actions_dict = {a: actions_np[i] for i, a in enumerate(env.agents)}
            next_obs, rews, dones, _, info = env.step(actions_dict)

            obs_buf.append(obs_tensor)
            act_buf.append(actions)
            logp_buf.append(log_probs)
            rew_buf.append(torch.tensor(
                [rews[a] for a in env.agents],
                dtype=torch.float32).to(DEVICE))
            done_buf.append(dones[env.agents[0]])
            gobs_buf.append(global_obs)

            obs = next_obs
            step += 1

            if dones[env.agents[0]]:
                episode += 1
                obs, _ = env.reset()

        rewards  = torch.stack(rew_buf)
        obs_t    = torch.stack(obs_buf)
        acts_t   = torch.stack(act_buf)
        logps_t  = torch.stack(logp_buf)
        gobs_t   = torch.cat(gobs_buf)

        with torch.no_grad():
            values = critic(gobs_t).squeeze(-1)

        mean_rew   = rewards.mean(dim=1)
        advantages = torch.zeros_like(mean_rew)
        gae = 0.0
        for t in reversed(range(N_STEPS)):
            delta = mean_rew[t] + (GAMMA * values[t] *
                    (1 - float(done_buf[t]))) - values[t]
            gae   = delta + GAMMA * LAM * (1 - float(done_buf[t])) * gae
            advantages[t] = gae
        returns     = advantages + values
        advantages  = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(N_EPOCHS):
            idx = torch.randperm(N_STEPS)
            for start in range(0, N_STEPS, BATCH_SIZE):
                mb = idx[start:start + BATCH_SIZE]
                if len(mb) < 2:
                    continue
                obs_mb   = obs_t[mb].view(-1, OBS_DIM)
                acts_mb  = acts_t[mb].view(-1, ACTION_DIM)
                logps_mb = logps_t[mb].view(-1)
                adv_mb   = advantages[mb].unsqueeze(1).expand(
                           -1, n_agents).reshape(-1)
                ret_mb   = returns[mb]
                gobs_mb  = gobs_t[mb]

                new_logps, entropy = actor.evaluate(obs_mb, acts_mb)
                ratio  = (new_logps - logps_mb).exp()
                surr1  = ratio * adv_mb
                surr2  = ratio.clamp(1-EPS_CLIP, 1+EPS_CLIP) * adv_mb
                loss   = (-torch.min(surr1, surr2).mean()
                          + 0.5 * (ret_mb - critic(gobs_mb).squeeze(-1)).pow(2).mean()
                          - 0.05 * entropy.mean())

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(actor.parameters()) + list(critic.parameters()), 0.5)
                optimizer.step()

        if step % 50_000 == 0:
            print(f"  [{config_code}] Step {step:>7,} | "
                  f"Ep {episode:>4} | "
                  f"Cov: {info['coverage_pct']:.1f}% | "
                  f"Det: {info['detection_rate']:.2f}")

    ckpt_dir = f'results/mappo/{config_code}'
    os.makedirs(ckpt_dir, exist_ok=True)
    torch.save({'actor': actor.state_dict(),
                'critic': critic.state_dict(),
                'config': cfg},
               f'{ckpt_dir}/checkpoint.pt')
    print(f"  Checkpoint saved: {ckpt_dir}/checkpoint.pt")
    return actor


if __name__ == '__main__':
    for code, steps in STEPS_MAP.items():
        print(f"\n{'='*55}")
        print(f"Training MAPPO — {code} ({steps:,} steps)")
        print(f"  {SWARN_CONFIGS[code]['description']}")
        print('='*55)
        train_mappo(code, steps)
# scripts/train_qmix_10agents.py
import sys, os
sys.path.insert(0, '.')
from algos.qmix.train_qmix import train

print("Training QMIX with 10 agents, sensed coverage...")

train({
    "scenario":            "medium",
    "n_agents":            10,
    "map_size":            100,
    "n_targets":           5,
    "n_obstacles":         15,
    "max_steps":           500,
    "comm_dropout":        0.0,
    "obs_radius":          10,
    "coverage_mode":       "sensed",
    "total_steps":         1_000_000,
    "batch_size":          64,
    "lr":                  5e-5,
    "gamma":               0.99,
    "epsilon_start":       1.0,
    "epsilon_end":         0.05,
    "epsilon_decay_steps": 200_000,
    "target_update_freq":  500,
    "buffer_size":         20_000,
    "train_start":         2000,
    "checkpoint_freq":     50_000,
    "checkpoint_dir":      "results/qmix/checkpoints_10agents",
    "results_dir":         "results/qmix",
    "obs_dim":             29,
    "embed_dim":           64,
})
# env/benchmark_configs.py
# SWARN Benchmark — 12 Canonical Configurations
# Frozen v1.0 — DO NOT modify after baseline runs begin

SWARN_CONFIGS = {

    # ── SCALE TIER: Small ─────────────────────────────────────────────────
    "SWARN-S1": {
        "description": "Micro swarm, open terrain, static targets",
        "n_agents": 5, "map_size": 50, "n_targets": 3,
        "n_obstacles": 5, "comm_dropout": 0.0, "max_steps": 300,
        "obs_radius": 10, "coverage_mode": "sensed", "seed": 42,
    },
    "SWARN-S2": {
        "description": "Micro swarm, dense obstacles, static targets",
        "n_agents": 5, "map_size": 50, "n_targets": 3,
        "n_obstacles": 20, "comm_dropout": 0.0, "max_steps": 300,
        "obs_radius": 10, "coverage_mode": "sensed", "seed": 42,
    },

    # ── SCALE TIER: Medium ────────────────────────────────────────────────
    "SWARN-S3": {
        "description": "Standard swarm, open terrain, perfect comms",
        "n_agents": 10, "map_size": 100, "n_targets": 5,
        "n_obstacles": 10, "comm_dropout": 0.0, "max_steps": 500,
        "obs_radius": 10, "coverage_mode": "sensed", "seed": 42,
    },
    "SWARN-S4": {
        "description": "Standard swarm, dense obstacles, perfect comms",
        "n_agents": 10, "map_size": 100, "n_targets": 5,
        "n_obstacles": 25, "comm_dropout": 0.0, "max_steps": 500,
        "obs_radius": 10, "coverage_mode": "sensed", "seed": 42,
    },
    "SWARN-S5": {
        "description": "Standard swarm, moderate obstacles, lossy comms",
        "n_agents": 10, "map_size": 100, "n_targets": 5,
        "n_obstacles": 15, "comm_dropout": 0.2, "max_steps": 500,
        "obs_radius": 10, "coverage_mode": "sensed", "seed": 42,
    },
    "SWARN-S6": {
        "description": "Standard swarm, dense obstacles, lossy comms, more targets",
        "n_agents": 10, "map_size": 100, "n_targets": 8,
        "n_obstacles": 25, "comm_dropout": 0.2, "max_steps": 500,
        "obs_radius": 10, "coverage_mode": "sensed", "seed": 42,
    },

    # ── SCALE TIER: Large ─────────────────────────────────────────────────
    "SWARN-S7": {
        "description": "Large swarm, open terrain, perfect comms",
        "n_agents": 25, "map_size": 150, "n_targets": 8,
        "n_obstacles": 15, "comm_dropout": 0.0, "max_steps": 600,
        "obs_radius": 10, "coverage_mode": "sensed", "seed": 42,
    },
    "SWARN-S8": {
        "description": "Large swarm, dense obstacles, degraded comms",
        "n_agents": 25, "map_size": 150, "n_targets": 8,
        "n_obstacles": 35, "comm_dropout": 0.3, "max_steps": 600,
        "obs_radius": 10, "coverage_mode": "sensed", "seed": 42,
    },

    # ── SCALE TIER: XLarge ────────────────────────────────────────────────
    "SWARN-S9": {
        "description": "XL swarm, open terrain, perfect comms",
        "n_agents": 50, "map_size": 200, "n_targets": 10,
        "n_obstacles": 20, "comm_dropout": 0.0, "max_steps": 700,
        "obs_radius": 10, "coverage_mode": "sensed", "seed": 42,
    },
    "SWARN-S10": {
        "description": "XL swarm, dense obstacles, heavy comm dropout",
        "n_agents": 50, "map_size": 200, "n_targets": 10,
        "n_obstacles": 50, "comm_dropout": 0.5, "max_steps": 700,
        "obs_radius": 10, "coverage_mode": "sensed", "seed": 42,
    },

    # ── SCALE TIER: XXLarge ───────────────────────────────────────────────
    "SWARN-S11": {
        "description": "Mega swarm, moderate terrain, lossy comms",
        "n_agents": 100, "map_size": 250, "n_targets": 15,
        "n_obstacles": 40, "comm_dropout": 0.2, "max_steps": 1000,
        "obs_radius": 10, "coverage_mode": "sensed", "seed": 42,
    },
    "SWARN-S12": {
        "description": "Mega swarm, hardest — dense obstacles, severe dropout",
        "n_agents": 100, "map_size": 250, "n_targets": 15,
        "n_obstacles": 80, "comm_dropout": 0.5, "max_steps": 1000,
        "obs_radius": 10, "coverage_mode": "sensed", "seed": 42,
    },
}

# 10 fixed seeds per config for reproducibility
BENCHMARK_SEEDS = [42, 123, 256, 512, 1024, 2048, 4096, 7777, 8888, 9999]


def get_config(config_code: str) -> dict:
    if config_code not in SWARN_CONFIGS:
        raise ValueError(f"Unknown config: {config_code}. "
                        f"Choose from {list(SWARN_CONFIGS.keys())}")
    cfg = SWARN_CONFIGS[config_code].copy()
    cfg.pop("description")
    cfg["detect_radius"] = 2.0   # frozen v2.1 benchmark value
    return cfg


def list_configs():
    print(f"\n{'Code':<12} {'Agents':>7} {'Map':>6} {'Targets':>8} "
          f"{'Obstacles':>10} {'Dropout':>8} {'Steps':>7}")
    print("-" * 65)
    for code, cfg in SWARN_CONFIGS.items():
        print(f"{code:<12} {cfg['n_agents']:>7} {cfg['map_size']:>6} "
              f"{cfg['n_targets']:>8} {cfg['n_obstacles']:>10} "
              f"{cfg['comm_dropout']:>8.1f} {cfg['max_steps']:>7}")


if __name__ == "__main__":
    list_configs()
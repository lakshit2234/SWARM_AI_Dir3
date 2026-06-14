import gymnasium as gym
import numpy as np
from gymnasium import spaces


class SwarmEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, config=None):
        defaults = {
            "n_agents": 10,
            "map_size": 100,
            "n_targets": 5,
            "n_obstacles": 15,
            "obs_radius": 10,
            "comm_radius": 20,
            "max_steps": 500,
            "scenario": "medium",
            "comm_dropout": 0.0,
            "seed": 42,
        }

        self.cfg = {**defaults, **(config or {})}

        np.random.seed(self.cfg["seed"])

        self.observation_space = spaces.Box(
            low=-1,
            high=1,
            shape=(29,),
            dtype=np.float32,
        )

        self.action_space = spaces.Box(
            low=-1,
            high=1,
            shape=(2,),
            dtype=np.float32,
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        cfg = self.cfg
        m = cfg["map_size"]

        self.agent_pos = np.random.uniform(
            5, m - 5, (cfg["n_agents"], 2)
        )

        self.agent_vel = np.zeros((cfg["n_agents"], 2))

        self.battery = np.ones(cfg["n_agents"])

        self.targets = np.random.uniform(
            0, m, (cfg["n_targets"], 2)
        )

        self.detected = np.zeros(
            cfg["n_targets"],
            dtype=bool
        )

        self.obstacles = [
            (
                np.random.uniform(10, m - 10, 2),
                np.random.uniform(2, 6, 2),
            )
            for _ in range(cfg["n_obstacles"])
        ]

        self.visited = np.zeros((m, m), dtype=bool)

        self.step_count = 0

        return self._get_obs(), {}

    def step(self, actions):
        cfg = self.cfg

        rewards = np.zeros(cfg["n_agents"])

        collision_count = 0

        for i in range(cfg["n_agents"]):

            vel = np.clip(actions[i], -1, 1)

            new_pos = self.agent_pos[i] + vel

            new_pos = np.clip(
                new_pos,
                0,
                cfg["map_size"] - 1
            )

            if self._collides_obstacle(new_pos):
                rewards[i] -= 1.0
                collision_count += 1
            else:
                self.agent_pos[i] = new_pos
                self.agent_vel[i] = vel

            speed = np.linalg.norm(vel)

            self.battery[i] -= (
                0.001 + 0.0005 * speed**2
            )

            self.battery[i] = max(
                0,
                self.battery[i]
            )

            gx, gy = int(new_pos[0]), int(new_pos[1])

            if not self.visited[gx, gy]:
                self.visited[gx, gy] = True
                rewards[i] += 0.01

            for t, tpos in enumerate(self.targets):

                if not self.detected[t]:

                    dist = np.linalg.norm(
                        self.agent_pos[i] - tpos
                    )

                    if dist < 2.0:
                        self.detected[t] = True
                        rewards[i] += 1.0

            for j in range(cfg["n_agents"]):

                if i != j:

                    dist = np.linalg.norm(
                        self.agent_pos[i]
                        - self.agent_pos[j]
                    )

                    if dist < 1.5:
                        rewards[i] -= 0.5
                        collision_count += 1

            rewards[i] -= 0.001

        self.step_count += 1

        done = bool(
            self.detected.all()
            or self.step_count >= cfg["max_steps"]
        )

        info = {
            "coverage_pct": (
                100.0
                * self.visited.sum()
                / self.visited.size
            ),
            "detection_rate": (
                self.detected.sum()
                / len(self.detected)
            ),
            "step": self.step_count,
            "collisions": collision_count,
        }

        return (
            self._get_obs(),
            rewards,
            done,
            False,
            info,
        )

    def _get_obs(self):

        cfg = self.cfg

        observations = []

        m = cfg["map_size"]

        for i in range(cfg["n_agents"]):

            pos = self.agent_pos[i] / m

            vel = self.agent_vel[i]

            nearest_agents = []

            for j in range(cfg["n_agents"]):

                if i == j:
                    continue

                rel = (
                    self.agent_pos[j]
                    - self.agent_pos[i]
                ) / m

                nearest_agents.extend(rel.tolist())

            nearest_agents = (
                nearest_agents[:10]
            )

            while len(nearest_agents) < 10:
                nearest_agents.append(0.0)

            target_feats = []

            for t in self.targets:

                rel = (
                    t - self.agent_pos[i]
                ) / m

                target_feats.extend(rel.tolist())

            target_feats = target_feats[:6]

            while len(target_feats) < 6:
                target_feats.append(0.0)

            obstacle_feats = []

            for centre, half in self.obstacles:

                rel = (
                    centre - self.agent_pos[i]
                ) / m

                obstacle_feats.extend(rel.tolist())

            obstacle_feats = obstacle_feats[:8]

            while len(obstacle_feats) < 8:
                obstacle_feats.append(0.0)

            obs = np.array(
                [
                    pos[0],
                    pos[1],
                    vel[0],
                    vel[1],
                    *nearest_agents,
                    *target_feats,
                    *obstacle_feats,
                    self.battery[i],
                ],
                dtype=np.float32,
            )

            observations.append(obs)

        return np.array(observations)

    def _collides_obstacle(self, pos):

        for centre, half in self.obstacles:

            if (
                abs(pos[0] - centre[0]) < half[0]
                and abs(pos[1] - centre[1]) < half[1]
            ):
                return True

        return False
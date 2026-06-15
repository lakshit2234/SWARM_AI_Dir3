import gymnasium as gym
import numpy as np
from gymnasium import spaces


class SwarmEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"]}

    # 8 compass unit vectors: N, NE, E, SE, S, SW, W, NW
    _COMPASS = np.array([
        [0.0, 1.0], [0.7071, 0.7071], [1.0, 0.0], [0.7071, -0.7071],
        [0.0, -1.0], [-0.7071, -0.7071], [-1.0, 0.0], [-0.7071, 0.7071],
    ])

    def __init__(self, config=None):
        defaults = {
            "n_agents": 10, "map_size": 100, "n_targets": 5, "n_obstacles": 15,
            "obs_radius": 10, "comm_radius": 20, "max_steps": 500,
            "scenario": "medium", "comm_dropout": 0.0, "seed": 42,
            # --- WORLD SETTINGS — frozen at v2.1 (2026-06-01) ---
            # detect_radius: distance at which an agent detects a target.
            # Set to 5.0 ("discriminative challenge") or 10.0 ("sensor-realistic").
            # DO NOT change after the team freeze — all five algorithms must share
            # the same value for the paper comparison to be valid.
            "detect_radius": 5.0,      # FROZEN v2.1 default — change only before freeze
            "coverage_mode": "sensed", # FROZEN v2.1 — "sensed" = sensor-footprint coverage
        }
        self.cfg = {**defaults, **(config or {})}
        self.observation_space = spaces.Box(-1, 1, shape=(29,), dtype=np.float32)
        self.action_space = spaces.Box(-1, 1, shape=(2,), dtype=np.float32)
        self.reset(seed=self.cfg["seed"])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        cfg, m, rng = self.cfg, self.cfg["map_size"], self.np_random
        self.agent_pos = rng.uniform(5, m - 5, (cfg["n_agents"], 2))
        self.agent_vel = np.zeros((cfg["n_agents"], 2))
        self.battery = np.ones(cfg["n_agents"])
        self.targets = rng.uniform(0, m, (cfg["n_targets"], 2))
        self.detected = np.zeros(cfg["n_targets"], dtype=bool)
        self.obstacles = [(rng.uniform(10, m - 10, 2), rng.uniform(2, 6, 2))
                          for _ in range(cfg["n_obstacles"])]
        self.obs_centres = (np.array([c for c, h in self.obstacles])
                            if self.obstacles else np.zeros((0, 2)))
        self.obs_halfs = (np.array([h for c, h in self.obstacles])
                          if self.obstacles else np.zeros((0, 2)))
        self.visited = np.zeros((m, m), dtype=bool)   # cells entered
        self.sensed = np.zeros((m, m), dtype=bool)    # sensor footprint
        self.step_count = 0
        return self._get_obs(), {}

    def step(self, actions):
        cfg, m, n = self.cfg, self.cfg["map_size"], self.cfg["n_agents"]
        rewards = np.zeros(n)
        collisions = 0
        actions = np.asarray(actions, dtype=float)

        for i in range(n):
            vel = np.clip(actions[i], -1, 1)
            proposed = np.clip(self.agent_pos[i] + vel, 0, m - 1)
            if self._collides_obstacle(proposed):
                rewards[i] -= 1.0
                collisions += 1
                self.agent_vel[i] = 0.0
            else:
                self.agent_pos[i] = proposed
                self.agent_vel[i] = vel

            speed = np.linalg.norm(self.agent_vel[i])
            self.battery[i] = max(0.0, self.battery[i] - (0.001 + 0.0005 * speed ** 2))

            # Coverage always stamped as sensed footprint (frozen v2.1)
            self._stamp_sensed(self.agent_pos[i])

            # Also track visited cells for backward-compat info key
            gx, gy = int(self.agent_pos[i][0]), int(self.agent_pos[i][1])
            if not self.visited[gx, gy]:
                self.visited[gx, gy] = True
                rewards[i] += 0.01

            for t, tpos in enumerate(self.targets):
                if not self.detected[t] and \
                        np.linalg.norm(self.agent_pos[i] - tpos) < cfg["detect_radius"]:
                    self.detected[t] = True
                    rewards[i] += 1.0
            rewards[i] -= 0.001

        # Drone-drone collisions
        for i in range(n):
            for j in range(i + 1, n):
                if np.linalg.norm(self.agent_pos[i] - self.agent_pos[j]) < 1.5:
                    rewards[i] -= 0.5
                    rewards[j] -= 0.5
                    collisions += 1

        self.step_count += 1
        done = bool(self.detected.all() or self.step_count >= cfg["max_steps"])
        info = {
            "coverage_pct": 100.0 * self.sensed.sum() / (m * m),
            "detection_rate": self.detected.sum() / len(self.detected),
            "step": self.step_count,
            "collisions": collisions,
        }
        return self._get_obs(), rewards, done, False, info

    def _get_obs(self):
        cfg, m = self.cfg, self.cfg["map_size"]
        n, obs_r, comm_r = cfg["n_agents"], cfg["obs_radius"], cfg["comm_radius"]
        dropout, rng = cfg["comm_dropout"], self.np_random
        out = np.zeros((n, 29), dtype=np.float32)

        for i in range(n):
            pos_i = self.agent_pos[i]
            feat = list(pos_i / m) + list(self.agent_vel[i])

            rel = self.agent_pos - pos_i
            dist = np.linalg.norm(rel, axis=1)
            within = [j for j in np.argsort(dist) if j != i and dist[j] <= comm_r][:5]
            af = []
            for j in within:
                if dropout > 0 and rng.random() < dropout:
                    af += [0.0, 0.0]
                else:
                    af += list(rel[j] / m)
            while len(af) < 10:
                af.append(0.0)
            feat += af[:10]

            tf = []
            tidx = [t for t in range(len(self.targets)) if not self.detected[t]]
            if tidx:
                trel = self.targets[tidx] - pos_i
                tdist = np.linalg.norm(trel, axis=1)
                cnt = 0
                for k in np.argsort(tdist):
                    if cnt >= 3 or tdist[k] > obs_r:
                        break
                    tf += list(trel[k] / m)
                    cnt += 1
            while len(tf) < 6:
                tf.append(0.0)
            feat += tf[:6]

            feat += self._obstacle_rays(pos_i)
            feat.append(float(self.battery[i]))
            out[i] = np.asarray(feat, dtype=np.float32)
        return out

    def _obstacle_rays(self, pos):
        m, obs_r = self.cfg["map_size"], self.cfg["obs_radius"]
        out = []
        for d in self._COMPASS:
            hit, r = obs_r, 1.0
            while r <= obs_r:
                p = pos + d * r
                if (p[0] < 0 or p[0] > m - 1 or p[1] < 0 or p[1] > m - 1
                        or self._collides_obstacle(p)):
                    hit = r
                    break
                r += 1.0
            out.append(hit / obs_r)
        return out

    def _stamp_sensed(self, pos):
        m, r = self.cfg["map_size"], int(self.cfg["obs_radius"])
        cx, cy = int(pos[0]), int(pos[1])
        x0, x1 = max(0, cx - r), min(m, cx + r + 1)
        y0, y1 = max(0, cy - r), min(m, cy + r + 1)
        gx, gy = np.meshgrid(np.arange(x0, x1), np.arange(y0, y1), indexing="ij")
        self.sensed[x0:x1, y0:y1] |= ((gx - cx) ** 2 + (gy - cy) ** 2 <= r * r)

    def _collides_obstacle(self, pos):
        if len(self.obs_centres) == 0:
            return False
        return bool(np.any(np.all(np.abs(pos - self.obs_centres) < self.obs_halfs, axis=1)))
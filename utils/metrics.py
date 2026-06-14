import csv
import os
import numpy as np


class MetricsTracker:
    def __init__(self):
        self.episodes = []
        self._ep = {}

    def episode_start(self):
        self._ep = {
            "coverage_pct":           0.0,
            "detection_rate":         0.0,
            "total_collisions":       0,
            "steps_to_first_target":  -1,
            "steps_to_90pct_coverage":-1,
            "time_to_detect_all":     -1,
            "episode_length":         0,
            "total_path_length":      0.0,
            "_first_target_found":    False,
            "_90pct_reached":         False,
            "_all_detected":          False,
            "_prev_positions":        None,
        }

    def update(self, step, info, collisions_this_step=0,
               agent_positions=None):
        self._ep["coverage_pct"]     = info.get("coverage_pct", 0)
        self._ep["detection_rate"]   = info.get("detection_rate", 0)
        self._ep["total_collisions"] += collisions_this_step
        self._ep["episode_length"]   = step

        # Steps to first target
        if (info.get("detection_rate", 0) > 0
                and not self._ep["_first_target_found"]):
            self._ep["steps_to_first_target"] = step
            self._ep["_first_target_found"]   = True

        # Steps to 90% coverage
        if (info.get("coverage_pct", 0) >= 90
                and not self._ep["_90pct_reached"]):
            self._ep["steps_to_90pct_coverage"] = step
            self._ep["_90pct_reached"]          = True

        # Time to detect ALL targets
        if (info.get("detection_rate", 0) >= 1.0
                and not self._ep["_all_detected"]):
            self._ep["time_to_detect_all"] = step
            self._ep["_all_detected"]      = True

        # Total path length — sum of agent displacements
        if agent_positions is not None:
            prev = self._ep["_prev_positions"]
            if prev is not None:
                dists = np.linalg.norm(
                    agent_positions - prev, axis=1
                )
                self._ep["total_path_length"] += float(dists.sum())
            self._ep["_prev_positions"] = agent_positions.copy()

    def episode_end(self, episode_number):
        self.episodes.append({
            "episode":                episode_number,
            "coverage_pct":           round(self._ep["coverage_pct"], 2),
            "detection_rate":         round(self._ep["detection_rate"], 3),
            "total_collisions":       self._ep["total_collisions"],
            "steps_to_first_target":  self._ep["steps_to_first_target"],
            "steps_to_90pct_coverage":self._ep["steps_to_90pct_coverage"],
            "time_to_detect_all":     self._ep["time_to_detect_all"],
            "episode_length":         self._ep["episode_length"],
            "total_path_length":      round(self._ep["total_path_length"], 2),
        })

    def save_csv(self, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        cols = [
            "episode", "coverage_pct", "detection_rate",
            "total_collisions", "steps_to_first_target",
            "steps_to_90pct_coverage", "time_to_detect_all",
            "episode_length", "total_path_length",
        ]
        with open(filepath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=cols)
            writer.writeheader()
            writer.writerows(self.episodes)
        print(f"Saved {len(self.episodes)} episodes to {filepath}")
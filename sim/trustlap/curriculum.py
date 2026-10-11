"""Automatic difficulty curriculum (SPEC 7.4), inspired by Automatic Domain Randomization
(OpenAI, 2019). One global difficulty d ∈ [0, 1] scales every condition range.

Every 10 updates: success = share of finished episodes (at the curriculum difficulty) that
ended in a lap or the time limit, not a crash. Above 0.8 twice in a row → d += 0.05; below
0.5 → d −= 0.05. 20% of envs always train at d = 0 (no forgetting), 10% get one parameter
at the edge of the current range (stretch exposure).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from trustlap.config import CONFIG, CurriculumConfig


@dataclass
class Curriculum:
    cfg: CurriculumConfig = field(default_factory=lambda: CONFIG.curriculum)
    difficulty: float = 0.0
    consecutive_up: int = 0
    episodes: float = 0.0
    successes: float = 0.0
    updates_since_check: int = 0
    last_success: float = float("nan")

    def record(self, episodes: float, successes: float) -> None:
        """Add one update's finished-episode counts."""
        self.episodes += episodes
        self.successes += successes
        self.updates_since_check += 1

    def step(self) -> bool:
        """Check after every `check_interval` updates. Returns True if d changed."""
        if self.updates_since_check < self.cfg.check_interval_updates:
            return False
        changed = False
        if self.episodes > 0:
            success = self.successes / self.episodes
            self.last_success = success
            if success > self.cfg.up_threshold:
                self.consecutive_up += 1
                if self.consecutive_up >= self.cfg.consecutive_up:
                    changed = self._move(+self.cfg.step)
                    self.consecutive_up = 0
            else:
                self.consecutive_up = 0
                if success < self.cfg.down_threshold:
                    changed = self._move(-self.cfg.step)
        self.episodes = self.successes = 0.0
        self.updates_since_check = 0
        return changed

    def _move(self, delta: float) -> bool:
        new = float(np.clip(round(self.difficulty + delta, 4), 0.0, 1.0))
        changed = new != self.difficulty
        self.difficulty = new
        return changed

    def env_arrays(self, n_envs: int) -> tuple[np.ndarray, np.ndarray]:
        """Per-env difficulty and stretch flags: first 20% easy (d = 0), next 10% stretched."""
        n_easy = int(round(self.cfg.easy_frac * n_envs))
        n_stretch = int(round(self.cfg.boundary_frac * n_envs))
        difficulty = np.full(n_envs, self.difficulty, np.float32)
        difficulty[:n_easy] = 0.0
        stretch = np.zeros(n_envs, bool)
        stretch[n_easy : n_easy + n_stretch] = self.difficulty > 0
        return difficulty, stretch

    def state_dict(self) -> dict:
        return {
            "difficulty": self.difficulty,
            "consecutive_up": self.consecutive_up,
            "episodes": self.episodes,
            "successes": self.successes,
            "updates_since_check": self.updates_since_check,
        }

    def load_state_dict(self, d: dict) -> None:
        for k, v in d.items():
            setattr(self, k, v)

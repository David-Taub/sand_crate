from collections import defaultdict
from typing import TYPE_CHECKING

import numpy as np
import yaml

if TYPE_CHECKING:
    from src.crate.crate import Crate

DECAY = 0.95


class ForceMonitor:
    def __init__(self, crate: "Crate") -> None:
        self.reset()
        self.crate = crate
        self.ticks: int = 0
        self.context_to_velocity = defaultdict(lambda: 0)

    def __call__(self, context: str = "") -> "ForceMonitor":
        self.context = context
        return self

    def __enter__(self) -> "ForceMonitor":
        self.preforce_velocity = self.crate.particle_velocities.copy()
        return self

    def __exit__(self, *args) -> None:
        self.postforce_velocity = self.crate.particle_velocities
        if self.postforce_velocity.shape[0] == 0:
            return
        velocity_diff = self.postforce_velocity - self.preforce_velocity
        avg_velocity_normal = np.mean(np.hypot(velocity_diff[:, 0], velocity_diff[:, 1]))
        self.context_to_velocity[self.context] *= DECAY
        self.context_to_velocity[self.context] += (1 - DECAY) * avg_velocity_normal

    def reset(self):
        self.context_to_velocity = defaultdict(lambda: 0)

    def report(self) -> str:
        rounded = {context: float(f"{100 * velocity:.1f}") for context, velocity in self.context_to_velocity.items()}
        self.reset()
        return yaml.dump(
            {
                "Forces": rounded,
            }
        )

from collections import defaultdict
from typing import TYPE_CHECKING

import numpy as np
import yaml

if TYPE_CHECKING:
    from src.crate.crate import Crate

DECAY = 0.80


class ForceMonitor:
    def __init__(self, crate: "Crate") -> None:
        self.crate = crate
        self.ticks: int = 0
        self.context_to_velocity = defaultdict(lambda: 0)

    def __call__(self, context: str) -> "ForceMonitor":
        self.context = context
        return self

    def __enter__(self) -> "ForceMonitor":
        self.preforce_velocity = self.crate.particle_velocities.copy()
        return self

    def __exit__(self, *args) -> None:
        postforce_velocity = self.crate.particle_velocities
        if postforce_velocity.shape[0] == 0:
            return
        velocity_diff = postforce_velocity - self.preforce_velocity
        self.context_to_velocity[self.context] *= DECAY
        self.context_to_velocity[self.context] += (1 - DECAY) * np.mean(np.linalg.norm(velocity_diff, axis=1))

    def report(self) -> str:
        rounded = {context: float(f"{1000 * velocity:.1f}") for context, velocity in self.context_to_velocity.items()}
        return yaml.dump({"Forces": rounded})

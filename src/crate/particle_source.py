from dataclasses import dataclass
from typing import Optional

import numpy as np
from nptyping import NDArray


@dataclass
class ParticleSource:
    radius: float
    position: list[float]
    velocity: list[float]
    flow: float
    active_ticks: int
    noise: float = 0.05

    def generate_particles(self, dt: float, max_particles: int) -> tuple[Optional[NDArray], Optional[NDArray]]:
        particle_count = min(np.round(np.random.binomial(self.flow, dt)), max_particles)
        if particle_count == 0:
            return None, None
        particles = (np.random.rand(particle_count, 2) - 0.5) * self.radius + np.array(self.position)
        particle_velocities = np.ones_like(particles) * np.array(self.velocity)[None]
        particle_velocities += (np.random.rand(particle_count, 2) - 0.5) * self.noise
        return particles, particle_velocities


def build_particle_sources(particle_source_configs):
    return [ParticleSource(**config) for config in particle_source_configs]

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
    particle_mass: float = 1.0
    noise: float = 0.05

    def generate_particles(
            self, dt: float, max_particles: int
    ) -> tuple[Optional[NDArray], Optional[NDArray], Optional[NDArray]]:
        particle_count = min(np.round(np.random.binomial(self.flow, dt)), max_particles)
        if particle_count == 0:
            return None, None, None
        particles = (np.random.rand(particle_count, 2) - 0.5) * self.radius + np.array(self.position)
        particle_velocities = np.ones_like(particles) * np.array(self.velocity)[None]
        particle_velocities += (np.random.rand(particle_count, 2) - 0.5) * self.noise
        new_particle_masses = np.ones(particle_count) * self.particle_mass
        return particles, particle_velocities, new_particle_masses

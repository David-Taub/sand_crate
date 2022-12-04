import numpy as np

from geometry_utils import points_to_segments_distance
from neighbor_detector import detect_particle_neighbors
from typings import Particles

DT = 0.005
PARTICLE_RADIUS = 0.01
DIAMETER = PARTICLE_RADIUS * 2
PARTICLE_MASS = 0.3
PRESSURE_AMPLIFIER = 7
IGNORED_PRESSURE = 0.1
VISCOSITY = 2
TENSILE_ALPHA = 5
TENSILE_BETA = 3  # droplets factor
TARGET_FRAME_RATE = 120
PARTICLE_COUNT = 500
NOISE_LEVEL = 0.2
ONTOP_FAKE_DISTANCE = 0.9
WALL_FAKE_OVERLAP = 0.9
WALL_INTERACTION_DISTANCE = PARTICLE_RADIUS


class Crate:
    gravity = np.array([0.0, 9.81])

    def __init__(self) -> None:
        self.particles: Particles
        self.colliders: Particles
        # N x 2 x 2
        self.segments = np.array([
            [[0.0, 0.0], [0.0, 1.0]],
            [[0.0, 0.0], [1.0, 0.0]],
            [[1.0, 0.0], [1.0, 1.0]],
            [[0.0, 1.0], [1.0, 1.0]],
        ])
        self.colliders_indices = [[]] * PARTICLE_COUNT
        self.gen_particles()
        self.particles = np.random.rand(PARTICLE_COUNT, 2)
        self.particle_velocities = np.zeros((PARTICLE_COUNT, 2))
        self.particles_weights = np.ones(PARTICLE_COUNT)

    def gen_particles(self) -> None:
        self.particles = np.random.rand(PARTICLE_COUNT, 2)
        self.particle_velocities = np.zeros((PARTICLE_COUNT, 2))

    def physics_tick(self):
        self.colliders_indices = detect_particle_neighbors(particles=self.particles, diameter=DIAMETER)
        self.populate_colliders()
        self.compute_particle_pressures()

        self.apply_pressure()
        self.apply_viscocity()
        self.apply_velocity()

    def populate_colliders(self):
        self.colliders = []
        self.collider_weights = []
        self.collider_velocities = []
        for particle_index in range(self.particles.shape[0]):
            collider_indices = self.colliders_indices[particle_index]
            particle_colliders = self.particles[collider_indices]
            # particle_colliders += (np.random.rand(self.colliders[i].shape[0], 2) - 0.5) * DIAMETER * NOISE_LEVEL
            self.colliders.append(self.particles[particle_index] - particle_colliders)
            self.collider_weights.append(self.particles_weights[collider_indices])
            self.collider_velocities.append(self.particle_velocities[collider_indices])

        self.add_wall_virtual_colliders()

    def add_wall_virtual_colliders(self):
        segment_closest, distances = points_to_segments_distance(self.particles, self.segments)
        for particle_index, particle in enumerate(self.particles):
            """
              segment
                  +
                  | 
            *-----|-----* virtual p 
            p     | 
                  | 
                  | 
                  +   
            """
            particle_segment_closest = segment_closest[
                particle_index, distances[particle_index] <= WALL_INTERACTION_DISTANCE]
            virtual_collider = (particle - particle_segment_closest) * 2
            self.colliders[particle_index] = np.vstack(
                [self.colliders[particle_index], virtual_collider])

    # def copy_to_colliders_mats(self):
    #     for i in range(PARTICLE_COUNT):
    #         self.colliders[i][0: len(self.colliders_indices[i]), 3: 9] = self.particles[self.colliders_indices[i], 2: 8]

    def compute_particle_pressures(self):
        particles_pressure = []
        for i in range(self.particles.shape[0]):
            if self.colliders[i].shape[0] == 0:
                continue
            # ontop_mask = np.logical_and(self.colliders[i][:, 0] == 0, self.colliders[i][:, 1] == 0)
            # if np.sum(ontop_mask) > 0:
            #     self.colliders[i][ontop_mask, 0: 2] = (np.random.rand(1, 2) - 0.5) * ONTOP_FAKE_DISTANCE * DIAMETER
            collider_distances = np.hypot(
                self.colliders[i][:, 0] + self.colliders[i][:, 1]).reshape(-1, 1)
            normalized_collider_distances = 1 - np.clip(collider_distances / DIAMETER, 0, 1)
            particle_pressure = np.sum(normalized_collider_distances * self.collider_weights, 0)  # total overlap
            # particle_pressure = np.maximum(0, particle_pressure - IGNORED_PRESSURE)
            particles_pressure[i] = particle_pressure
        self.particles_pressure = np.array(particles_pressure)

    def apply_pressure(self):
        for i in range(self.particles.shape[0]):
            collider_pressures = self.particles_pressure[self.colliders_indices[i]]
            self.particle_velocities[i] += DT * PRESSURE_AMPLIFIER * np.sum(
                self.particles_pressure[i] + collider_pressures, 0) * \
                                           self.particles_weights[i] * self.colliders[i]

    def apply_gravity(self):
        self.particle_velocities += DT * self.gravity * PARTICLE_MASS

    def apply_viscocity(self):
        for i in range(self.particles.shape[0]):
            self.particle_velocities[i] += DT * VISCOSITY * np.sum(
                self.collider_velocities[i] - self.particle_velocities[i], 0)
            # viscosity
            # # surface tension
            # a = TENSILE_ALPHA * (self.particles[i, 4] + self.colliders[i][:, 5: 6] - 2 * IGNORED_PRESSURE)
            # b = TENSILE_BETA * (
            #     np.sum((self.colliders[i][:, 7: 9] - self.particles[i, 6: 8]) * self.colliders[i][:, 0: 2], 1)[:, None])
            # self.particles_velocities[i] += DT * np.sum(self.colliders[i][:, 0: 2] * (a + b), 0)

    def apply_velocity(self):
        self.particles += DT * self.particle_velocities
        self.particles = np.maximum(self.particles, PARTICLE_RADIUS)
        self.particles = np.minimum(self.particles, 1 - PARTICLE_RADIUS)

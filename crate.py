import numpy as np

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


class Crate:
    gravity = np.array([0.0, 9.81])

    def __init__(self) -> None:
        self.particles: Particles
        self.colliders: Particles
        self.colliders_indices = [[]] * PARTICLE_COUNT
        self.gen_particles()
        self.particles = np.random.rand(PARTICLE_COUNT, 2)
        self.particles_velocities = np.zeros((PARTICLE_COUNT, 2))

    def gen_particles(self) -> None:
        self.particles = np.random.rand(PARTICLE_COUNT, 2)
        self.particles_velocities = np.zeros((PARTICLE_COUNT, 2))

    def physics_tick(self):
        self.colliders_indices = detect_particle_neighbors(particles=self.particles, diameter=DIAMETER)
        self.precalc_colliders_interaction()
        # self.copy_to_colliders_mats()
        self.apply_velocities_updates()
        self.apply_positions_updates()

    def add_wall_virtual_colliders(self, i):
        WALL_FAKE_OVERLAP = 0.9
        if self.particles[i, 0] <= PARTICLE_RADIUS:
            virtual_particle = np.zeros((1, 9))
            virtual_particle[0, 0] = DIAMETER * WALL_FAKE_OVERLAP
            virtual_particle[0, 5: 9] = [1 - WALL_FAKE_OVERLAP,
                                         (1 - WALL_FAKE_OVERLAP - IGNORED_PRESSURE) * PRESSURE_AMPLIFIER,
                                         DIAMETER * WALL_FAKE_OVERLAP * WALL_FAKE_OVERLAP * (1 - WALL_FAKE_OVERLAP), 0]
            self.colliders[i] = np.vstack([self.colliders[i], virtual_particle])

        if self.particles[i, 0] >= 1 - PARTICLE_RADIUS:
            virtual_particle = np.zeros((1, 9))
            virtual_particle[0, 0] = -DIAMETER * WALL_FAKE_OVERLAP
            virtual_particle[0, 5: 9] = [1 - WALL_FAKE_OVERLAP,
                                         (1 - WALL_FAKE_OVERLAP - IGNORED_PRESSURE) * PRESSURE_AMPLIFIER,
                                         -DIAMETER * WALL_FAKE_OVERLAP * WALL_FAKE_OVERLAP * (1 - WALL_FAKE_OVERLAP), 0]
            self.colliders[i] = np.vstack([self.colliders[i], virtual_particle])

        if self.particles[i, 1] <= PARTICLE_RADIUS:
            virtual_particle = np.zeros((1, 9))
            virtual_particle[0, 1] = DIAMETER * WALL_FAKE_OVERLAP
            virtual_particle[0, 5: 9] = [1 - WALL_FAKE_OVERLAP,
                                         (1 - WALL_FAKE_OVERLAP - IGNORED_PRESSURE) * PRESSURE_AMPLIFIER, 0,
                                         DIAMETER * WALL_FAKE_OVERLAP * WALL_FAKE_OVERLAP * (1 - WALL_FAKE_OVERLAP)]
            self.colliders[i] = np.vstack([self.colliders[i], virtual_particle])

        if self.particles[i, 1] >= 1 - PARTICLE_RADIUS:
            virtual_particle = np.zeros((1, 9))
            virtual_particle[0, 1] = -DIAMETER * WALL_FAKE_OVERLAP
            virtual_particle[0, 5: 9] = [1 - WALL_FAKE_OVERLAP,
                                         (1 - WALL_FAKE_OVERLAP - IGNORED_PRESSURE) * PRESSURE_AMPLIFIER, 0,
                                         -DIAMETER * WALL_FAKE_OVERLAP * WALL_FAKE_OVERLAP * (1 - WALL_FAKE_OVERLAP)]
            self.colliders[i] = np.vstack([self.colliders[i], virtual_particle])

    def populate_collider_positions(self):
        for i in range(self.particles.shape[0]):
            self.colliders[i] = np.zeros((len(self.colliders_indices[i]), 2))
            self.colliders[i] = self.particles[i] - self.particles[self.colliders_indices[i]]
            self.colliders[i] += (np.random.rand(self.colliders[i].shape[0], 2) - 0.5) * DIAMETER * NOISE_LEVEL
            self.add_wall_virtual_colliders(i)

    def precalc_colliders_interaction(self):
        for i in range(self.particles.shape[0]):
            if self.colliders[i].shape[0] == 0:
                continue
            # ontop_mask = np.logical_and(self.colliders[i][:, 0] == 0, self.colliders[i][:, 1] == 0)
            # if np.sum(ontop_mask) > 0:
            #     self.colliders[i][ontop_mask, 0: 2] = (np.random.rand(1, 2) - 0.5) * ONTOP_FAKE_DISTANCE * DIAMETER

            colliders_distance = np.sqrt(self.colliders[i][:, 0] ** 2 + self.colliders[i][:, 1] ** 2)
            self.colliders[i] /= colliders_distance  # normalized repel from collider
            colliders_distance_normalized = np.minimum(1, np.maximum(0,
                                                                     1 - colliders_distance / DIAMETER))  # collider particle overlap
            overlap = np.sum(colliders_distance_normalized, 0)  # total overlap
            perssure = np.maximum(0, (overlap - IGNORED_PRESSURE) * PRESSURE_AMPLIFIER)  # pressure
            # mid-range pressure
            self.particles[i, 6: 8] = np.sum(
                self.colliders[i][:, 0: 2] * (1 - self.colliders[i][:, 2: 3]) * (self.colliders[i][:, 2: 3]), 0)

    # def copy_to_colliders_mats(self):
    #     for i in range(PARTICLE_COUNT):
    #         self.colliders[i][0: len(self.colliders_indices[i]), 3: 9] = self.particles[self.colliders_indices[i], 2: 8]

    def apply_velocities_updates(self):
        # gravity
        self.particles_velocities += DT * self.gravity * PARTICLE_MASS
        for i in range(PARTICLE_COUNT):
            if self.colliders[i].shape[0] == 0:
                continue

            # # pressure - dt * (particle_presure + collider_presure) * collider_relative_overlap * normalized_repel
            # self.particles_velocities[i] += DT * np.sum(
            #     (self.particles[i, 5: 6] + self.colliders[i][:, 6: 7]) * self.colliders[i][:, 2: 3] * self.colliders[i][
            #                                                                                           :, 0: 2], 0)
            # viscosity
            self.particles_velocities[i] += DT * VISCOSITY * np.sum(
                self.colliders[i][:, 3: 5] - self.particles_velocities[i], 0)
            # # surface tension
            # a = TENSILE_ALPHA * (self.particles[i, 4] + self.colliders[i][:, 5: 6] - 2 * IGNORED_PRESSURE)
            # b = TENSILE_BETA * (
            #     np.sum((self.colliders[i][:, 7: 9] - self.particles[i, 6: 8]) * self.colliders[i][:, 0: 2], 1)[:, None])
            # self.particles_velocities[i] += DT * np.sum(self.colliders[i][:, 0: 2] * (a + b), 0)

    def apply_positions_updates(self):
        self.particles += DT * self.particles_velocities
        self.particles = np.maximum(self.particles, PARTICLE_RADIUS)
        self.particles = np.minimum(self.particles, 1 - PARTICLE_RADIUS)

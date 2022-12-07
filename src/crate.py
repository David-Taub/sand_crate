import numpy as np

from geometry_utils import points_to_segments_distance
from neighbor_detector import detect_particle_neighbors
from timer import Timer
from typings import Particles

DT = 0.005
PARTICLE_RADIUS = 0.005
DIAMETER = PARTICLE_RADIUS * 2
WALL_COLLISION_DECAY = 0.2
PARTICLE_MASS = 0.5
SPRING_OVERLAP_BALANCE = 0.1
SPRING_AMPLIFIER = 5000
PRESSURE_AMPLIFIER = 2000
IGNORED_PRESSURE = 0.0
NOISE_LEVEL = 0.1
VISCOSITY = 8

TARGET_FRAME_RATE = 120


class Crate:
    gravity = np.array([0.0, 9.81])

    def __init__(self, particle_count: int) -> None:
        self.particles: Particles
        self.colliders: Particles
        # N x 2 x 2
        self.segments = np.array([
            [[0.0, 0.0], [0.0, 1.0]],
            [[0.0, 0.0], [1.0, 0.0]],
            [[1.0, 0.0], [1.0, 1.0]],
            [[0.0, 1.0], [1.0, 1.0]],
        ])
        self.colliders_indices = [[]] * particle_count
        self.particles = np.random.rand(particle_count, 2)
        self.particle_velocities = np.zeros((particle_count, 2))
        self.particles_pressure = np.zeros((particle_count, 1))
        self.particles_weights = np.ones(particle_count) * PARTICLE_MASS
        self.colliders = []
        self.collider_weights = []
        self.colliders_indices = []
        self.collider_overlaps = []
        self.collider_velocities = []
        self.virtual_colliders = []
        self.collider_pressures = []
        self.debug_prints = ""
        self.timer = Timer()

    def physics_tick(self):
        with self.timer("Collisions"):
            self.colliders_indices = detect_particle_neighbors(particles=self.particles, diameter=DIAMETER)

        with self.timer("Colliders"):
            self.populate_colliders()
            self.add_wall_virtual_colliders()

        with self.timer("Pressure"):
            self.compute_particle_pressures()
            self.compute_collider_pressures()

        with self.timer("Forces"):
            self.apply_wall_bounce()
            self.apply_gravity()
            self.apply_pressure()
            self.apply_viscosity()

        with self.timer("Velocities"):
            self.apply_velocity()

        self.debug_prints = self.timer.report()

    def populate_colliders(self):
        self.colliders = []
        self.collider_weights = []
        self.collider_velocities = []
        for particle_index in range(self.particles.shape[0]):
            collider_indices = self.colliders_indices[particle_index]
            particle_colliders = self.particles[collider_indices]
            particle_colliders += (np.random.rand(len(collider_indices), 2) - 0.5) * DIAMETER * NOISE_LEVEL
            self.colliders.append(self.particles[particle_index] - particle_colliders)
            self.collider_weights.append(self.particles_weights[collider_indices])
            self.collider_velocities.append(self.particle_velocities[collider_indices])

    def add_wall_virtual_colliders(self):
        self.virtual_colliders = []
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
                particle_index, distances[particle_index] <= PARTICLE_RADIUS]
            virtual_colliders_count = particle_segment_closest.shape[0]
            if virtual_colliders_count:
                virtual_colliders = particle - particle_segment_closest
                self.colliders[particle_index] = np.vstack(
                    [self.colliders[particle_index], virtual_colliders * 2])
                self.collider_weights[particle_index] = np.hstack([self.collider_weights[particle_index], [
                    self.particles_weights[particle_index]] * virtual_colliders_count])
                self.collider_velocities[particle_index] = np.vstack(
                    [self.collider_velocities[particle_index], [[0.0, 0.0]] * virtual_colliders_count])
                self.virtual_colliders.append(virtual_colliders)
            else:
                self.virtual_colliders.append([])

    def apply_wall_bounce(self):
        for particle_index in range(self.particles.shape[0]):
            if len(self.virtual_colliders[particle_index]) == 0:
                continue
            wall_ortho = np.mean(self.virtual_colliders[particle_index], 0)
            wall_velocity_dot = np.dot(self.particle_velocities[particle_index], wall_ortho)
            if wall_velocity_dot < 0:
                wall_counter_component = wall_velocity_dot * wall_ortho / np.dot(wall_ortho, wall_ortho)
                self.particle_velocities[particle_index] -= wall_counter_component * 2 * (1 - WALL_COLLISION_DECAY)

    def compute_particle_pressures(self):
        particles_pressure = []
        self.collider_overlaps = []
        for i in range(self.particles.shape[0]):
            if self.colliders[i].shape[0] == 0:
                particles_pressure.append(0)
                self.collider_overlaps.append([])
                continue
            # K
            collider_distances = np.hypot(
                self.colliders[i][:, 0], self.colliders[i][:, 1])
            # K
            collider_overlaps = 1 - np.clip(collider_distances / DIAMETER, 0, 1)
            self.collider_overlaps.append(collider_overlaps)
            particle_pressure = np.sum(collider_overlaps * self.collider_weights[i], 0)  # total overlap
            particle_pressure = np.maximum(0, particle_pressure - IGNORED_PRESSURE)
            particles_pressure.append(particle_pressure)
        assert all(n >= 0 for n in particles_pressure)
        self.particles_pressure = np.array(particles_pressure)

    def compute_collider_pressures(self):
        self.collider_pressures = []
        for particle_index in range(self.particles.shape[0]):
            if self.colliders[particle_index].shape[0] == 0:
                self.collider_pressures.append([])
                continue
            particle_collider_pressures = self.particles_pressure[self.colliders_indices[particle_index]]
            # add virtual particle pressure
            particle_collider_pressures = np.append(particle_collider_pressures,
                                                    [0] * (self.colliders[particle_index].shape[0] - len(
                                                        self.colliders_indices[particle_index])))
            self.collider_pressures.append(particle_collider_pressures)

    def apply_pressure(self):
        for i in range(self.particles.shape[0]):
            if self.colliders[i].shape[0] == 0:
                continue
            # K
            weighted_pressures = (self.particles_pressure[i] + self.collider_pressures[i]) * self.particles_weights[i]
            # K x 2
            weighted_colliders = self.colliders[i] * weighted_pressures[:, None]
            self.particle_velocities[i] += DT * PRESSURE_AMPLIFIER * np.sum(weighted_colliders, 0)

    def apply_gravity(self):
        self.particle_velocities += DT * self.gravity[None]

    def apply_viscosity(self):
        for i in range(self.particles.shape[0]):
            self.particle_velocities[i] += DT * VISCOSITY * np.sum(
                self.collider_velocities[i] - self.particle_velocities[i], 0)

    def apply_velocity(self):
        self.particles += DT * self.particle_velocities
        self.particles = np.clip(self.particles, PARTICLE_RADIUS / 2, 1 - PARTICLE_RADIUS / 2)

    def apply_spring(self):
        for i in range(self.particles.shape[0]):
            if self.colliders[i].shape[0] == 0:
                continue
            spring_pull = SPRING_OVERLAP_BALANCE - self.collider_overlaps[i]
            self.particle_velocities[i] += DT * SPRING_AMPLIFIER * spring_pull * self.colliders[i]

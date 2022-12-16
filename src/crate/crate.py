import numpy as np
import yaml
from nptyping import NDArray

from .collision_detector import detect_particle_collisions
from .load_config import WorldConfig, load_config
from .rigid_body import FixedRigidBody, MotoredRigidBody
from .utils.force_monitor import ForceMonitor
from .utils.geometry_utils import points_to_segments_distance, segments_crossings, calc_cross_coefficient, pad_segments
from .utils.timer import Timer


class Crate:
    # TODO: make colliders into a 3d array, with zeros
    def __init__(self, world_config: WorldConfig) -> None:
        self.particles = np.zeros((0, 2))
        self.particle_velocities = np.zeros((0, 2))
        self.particles_pressure = np.zeros((0, 1))
        self.real_colliders = []
        self.colliders_indices = []
        self.collider_overlaps = []
        self.collider_velocities = []
        self.collider_pressures = []
        self.virtual_colliders = []
        self.virtual_colliders_velocity = []
        self.debug_prints = ""
        self.debug_timer = Timer()
        self.force_monitor = ForceMonitor(self)

        self.rigid_bodies = world_config.rigid_bodies
        self.particle_sources = world_config.particle_sources
        self.particle_radius = None
        self.dt = None
        self.wall_collision_decay = None
        self.spring_overlap_balance = None
        self.spring_amplifier = None
        self.pressure_amplifier = None
        self.surface_smoothing = None
        self.target_pressure = None
        self.ignored_pressure = None
        self.collider_noise_level = None
        self.viscosity = None
        self.max_particles = None

        for coefficient_name in self.editable_coefficients():
            setattr(self, coefficient_name, world_config.coefficients[coefficient_name])
        self.gravity = np.array(world_config.coefficients["gravity"])

    def editable_coefficients(self) -> list[str]:
        return list(load_config().world_config.coefficients.keys())

    def colliders_count(self, particle_index: int) -> int:
        return self.real_colliders[particle_index].shape[0]

    def colliders(self, particle_index: int) -> NDArray:
        return np.concatenate((self.real_colliders[particle_index], self.virtual_colliders[particle_index]))

    @property
    def diameter(self) -> float:
        return self.particle_radius * 2

    @property
    def segments(self) -> NDArray:
        # segments x dots(2) x dims(2)
        return np.vstack(rigid_body.segments for rigid_body in self.rigid_bodies)

    @property
    def segment_velocities(self) -> NDArray:
        return np.vstack(
            rigid_body.segment_velocities() for rigid_body in self.rigid_bodies
        )

    @property
    def particle_count(self) -> int:
        return self.particles.shape[0]

    def physics_tick(self) -> None:
        self.create_new_particles()
        self.remove_particles()
        self.apply_bodies_velocity()
        with self.debug_timer("Collisions"):
            self.colliders_indices = detect_particle_collisions(particles=self.particles, diameter=self.diameter)

        with self.debug_timer("Colliders"):
            self.populate_colliders()
        with self.debug_timer("Virtual Colliders"):
            self.add_wall_virtual_colliders()

        with self.debug_timer("Pressure"):
            self.compute_particle_pressures()
            self.compute_collider_pressures()
            # self.add_virtual_collider_pressure_and_overlap()

        with self.debug_timer("gravity"), self.force_monitor("gravity"):
            self.apply_gravity()
        with self.debug_timer("pressure"), self.force_monitor("pressure"):
            self.apply_pressure()
        with self.debug_timer("spring"), self.force_monitor("spring"):
            self.apply_spring()
        with self.debug_timer("viscosity"), self.force_monitor("viscosity"):
            self.apply_viscosity()
        with self.debug_timer("tension"), self.force_monitor("tension"):
            self.apply_tension()
        with self.debug_timer("wall_bounce"), self.force_monitor("wall_bounce"):
            self.apply_wall_bounce()

        self.apply_particles_velocity()

        self.debug_prints = self.debug_timer.report()
        self.debug_prints += f"\n\n{self.force_monitor.report()}"
        self.debug_prints += f"\n\n{self.get_coefficient_debug()}"
        self.debug_timer.reset()

    def create_new_particles(self) -> None:
        for particle_source in self.particle_sources:
            new_particles, new_particle_velocities = particle_source.generate_particles(
                dt=self.dt, max_particles=self.max_particles - self.particle_count
            )
            if new_particles is not None:
                self.particles = np.vstack((self.particles, new_particles))
                self.particle_velocities = np.vstack((self.particle_velocities, new_particle_velocities))

    def remove_particles(self) -> None:
        self.particle_velocities = np.delete(
            self.particle_velocities,
            np.where((self.particles < -self.particle_radius) | (self.particles > 1 + self.particle_radius))[0],
            0,
        )
        self.particles = np.delete(
            self.particles,
            np.where((self.particles < -self.particle_radius) | (self.particles > 1 + self.particle_radius))[0],
            0,
        )

    def populate_colliders(self) -> None:
        self.real_colliders = []
        self.collider_distances = []
        self.collider_velocities = []
        for particle_index in range(self.particle_count):
            collider_indices = self.colliders_indices[particle_index]
            particle_colliders = self.particles[collider_indices]
            particle_colliders += (
                    (np.random.rand(len(collider_indices), 2) - 0.5) * self.diameter * self.collider_noise_level
            )
            relative_colliders = self.particles[particle_index] - particle_colliders
            collider_distances = np.hypot(relative_colliders[:, 0], relative_colliders[:, 1])
            self.collider_distances.append(collider_distances)
            self.real_colliders.append(relative_colliders / collider_distances[:, None])
            self.collider_velocities.append(self.particle_velocities[collider_indices])

    def calc_continuous_collision_fix_factors(self) -> NDArray:
        particle_fix_factors = np.ones(self.particle_count)
        if self.particle_count == 0:
            return particle_fix_factors
        padded_segments = pad_segments(self.segments, self.particle_radius)
        particle_movements = np.concatenate(
            (self.particles[:, None], self.particles[:, None] + self.particle_velocities[:, None] * self.dt), 1)
        particle_segment_collisions = segments_crossings(particle_movements, padded_segments)
        particle_indices, segments_indices = np.where(particle_segment_collisions)
        if len(particle_indices) == 0:
            return particle_fix_factors
        a = padded_segments[segments_indices, 0, :]
        b = padded_segments[segments_indices, 1, :]
        cross_coefficients = calc_cross_coefficient(self.particles[particle_indices],
                                                    self.particle_velocities[particle_indices] * self.dt,
                                                    a, b - a)
        for i, particle_index in enumerate(particle_indices):
            particle_fix_factors[particle_index] = min(particle_fix_factors[particle_index],
                                                       cross_coefficients[i])
        return particle_fix_factors
        # self.particle_velocities[list(particle_fix_factors.keys())] *= \
        #     np.array(list(particle_fix_factors.values()))[:, None]

    def add_wall_virtual_colliders(self) -> None:
        self.virtual_colliders = []
        self.virtual_colliders_velocity = []
        nearest_point_in_segment, distances = points_to_segments_distance(self.particles, self.segments)
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
            touching_segments_mask = distances[particle_index] <= self.particle_radius
            particle_segment_contacts = nearest_point_in_segment[particle_index, touching_segments_mask]
            virtual_colliders_count = particle_segment_contacts.shape[0]
            if virtual_colliders_count:
                virtual_colliders = (particle - particle_segment_contacts) * 2
                # self.collider_velocities[particle_index] = np.vstack(
                #     (self.collider_velocities[particle_index], self.segment_velocities[touching_segments_mask])
                # )
                self.virtual_colliders.append(virtual_colliders)
                self.virtual_colliders_velocity.append(self.segment_velocities[touching_segments_mask])
            else:
                self.virtual_colliders.append(np.array([]))
                self.virtual_colliders_velocity.append(np.array([]))

    def apply_wall_bounce(self) -> None:
        for particle_index in range(self.particle_count):
            if len(self.virtual_colliders[particle_index]) == 0:
                continue
            wall_ortho = np.mean(self.virtual_colliders[particle_index], 0)
            wall_velocity = np.mean(self.virtual_colliders_velocity[particle_index], 0)
            wall_particle_velocity_dot = np.dot(self.particle_velocities[particle_index] - wall_velocity, wall_ortho)
            if wall_particle_velocity_dot < 0:
                # particle moves toward the wall
                normalized_wall_ortho = wall_ortho / np.dot(wall_ortho, wall_ortho)
                wall_counter_component = -2 * wall_particle_velocity_dot * normalized_wall_ortho
                self.particle_velocities[particle_index] += (
                        wall_counter_component * (1 - self.wall_collision_decay) + wall_velocity
                )

    def compute_particle_pressures(self) -> None:
        particles_pressure = []
        self.collider_overlaps = []
        for particle_index in range(self.particle_count):
            if self.colliders_count(particle_index) == 0:
                particles_pressure.append(0)
                self.collider_overlaps.append(np.array([]))
                continue
            # C
            collider_overlaps = 1 - np.clip(self.collider_distances[particle_index] / self.diameter, 0, 1)
            self.collider_overlaps.append(collider_overlaps)
            particle_pressure = np.sum(collider_overlaps, 0)
            particle_pressure = np.maximum(0, particle_pressure - self.ignored_pressure)
            particles_pressure.append(particle_pressure)
        self.particles_pressure = np.array(particles_pressure)

    def compute_collider_pressures(self) -> None:
        self.collider_pressures = []
        for particle_index in range(self.particle_count):
            if self.colliders_count(particle_index) == 0:
                self.collider_pressures.append(np.array([]))
                continue
            particle_collider_pressures = self.particles_pressure[self.colliders_indices[particle_index]]
            self.collider_pressures.append(particle_collider_pressures)

    def add_virtual_collider_pressure_and_overlap(self):
        for particle_index in range(self.particle_count):
            self.collider_overlaps[particle_index] = np.append(self.collider_overlaps[particle_index],
                                                               [0] * len(self.virtual_colliders[particle_index]))
            self.collider_pressures[particle_index] = np.append(self.collider_pressures[particle_index],
                                                                [0] * len(self.virtual_colliders[particle_index]))

    def apply_pressure(self) -> None:
        for particle_index in range(self.particle_count):
            if self.colliders_count(particle_index) == 0:
                continue

            # C x 2
            weighted_colliders = (
                    self.real_colliders[particle_index]
                    * (self.particles_pressure[particle_index] + self.collider_pressures[particle_index])[:, None]
            )

            self.particle_velocities[particle_index] += self.dt * self.pressure_amplifier * np.sum(weighted_colliders,
                                                                                                   0)

    def apply_gravity(self) -> None:
        self.particle_velocities += self.dt * self.gravity[None]
        for rigid_body in self.rigid_bodies:
            if isinstance(rigid_body, (FixedRigidBody, MotoredRigidBody)):
                continue
            rigid_body.center_velocity += self.dt * self.gravity

    def apply_viscosity(self) -> None:

        for particle_index in range(self.particle_count):
            self.particle_velocities[particle_index] += (
                    self.dt
                    * self.viscosity
                    * np.sum(self.collider_velocities[particle_index] - self.particle_velocities[particle_index], 0)
            )

    def apply_spring(self) -> None:
        for particle_index in range(self.particle_count):
            if self.colliders_count(particle_index) == 0:
                continue
            # C
            spring_pull = self.spring_overlap_balance - self.collider_overlaps[particle_index]
            self.particle_velocities[particle_index] += np.mean(
                self.dt * self.spring_amplifier * spring_pull[:, None] * self.real_colliders[particle_index], 0
            )

    def apply_tension(self) -> None:
        # P x 2
        surface_normals = np.zeros((self.particle_count, 2))
        for i in range(self.particle_count):
            if self.colliders_count(i) == 0:
                continue
            surface_normals[i] = np.sum(
                ((1 - self.collider_overlaps[i]) * self.collider_overlaps[i])[:, None] * self.real_colliders[i], 0)
        for i in range(self.particle_count):
            if self.colliders_count(i) == 0:
                continue
            # C x 2
            normal_deltas = surface_normals[i][None] - surface_normals[self.colliders_indices[i]]
            # C
            normals_alignment = np.sum(normal_deltas * self.real_colliders[i], 1) * self.surface_smoothing
            # C
            target_pressure_fix = self.collider_pressures[i] + self.particles_pressure[i] - 2 * self.target_pressure
            self.particle_velocities[i] += self.dt * np.sum(
                (normals_alignment + target_pressure_fix)[:, None] * self.real_colliders[i], 0)

        # s[i] = sum((1 - w[i, j]) * w[i, j] * n[i, j], j)
        # A[i] = a * (w[i] + w[j] - 2 * w0)
        # B[i] = b * dot(s[j] - s[i], n[i, j]))
        # v[i] ← v[i] - Δt * (A[i] + B[i]) * n[i, j]

    def apply_particles_velocity(self) -> None:
        fix_factors = self.calc_continuous_collision_fix_factors()
        self.particles += self.dt * self.particle_velocities * fix_factors[:, None]

    def apply_bodies_velocity(self) -> None:
        for rigid_body in self.rigid_bodies:
            rigid_body.apply_velocity(self.dt)

    def get_coefficient_debug(self) -> str:
        coefficients_list = [(name, getattr(self, name)) for name in self.editable_coefficients()]
        coefficients_list = [{name: val.tolist() if isinstance(val, np.ndarray) else val} for name, val in
                             coefficients_list]
        return yaml.dump(coefficients_list)

import numpy as np
import yaml
from nptyping import NDArray

from collision_detector import detect_particle_collisions
from geometry_utils import points_to_segments_distance
from load_config import WorldConfig
from rigid_body import FixedRigidBody, MotoredRigidBody
from timer import Timer


class Crate:
    # TODO: make colliders into a 3d array, with zeros
    def __init__(self, world_config: WorldConfig) -> None:
        self.particles = np.zeros((0, 2))
        self.particle_velocities = np.zeros((0, 2))
        self.particles_pressure = np.zeros((0, 1))
        self.colliders = []
        self.colliders_indices = []
        self.collider_overlaps = []
        self.collider_velocities = []
        self.collider_pressures = []
        self.virtual_colliders = []
        self.virtual_colliders_velocity = []
        self.debug_prints = ""
        self.debug_timer = Timer()

        self.rigid_bodies = world_config.rigid_bodies
        self.particle_sources = world_config.particle_sources
        self.particle_radius = world_config.consts["particle_radius"]
        self.dt = world_config.consts["dt"]
        self.wall_collision_decay = world_config.consts["wall_collision_decay"]
        self.spring_overlap_balance = world_config.consts["spring_overlap_balance"]
        self.spring_amplifier = world_config.consts["spring_amplifier"]
        self.pressure_amplifier = world_config.consts["pressure_amplifier"]
        self.surface_smoothing = world_config.consts["surface_smoothing"]
        self.target_pressure = world_config.consts["target_pressure"]
        self.ignored_pressure = world_config.consts["ignored_pressure"]
        self.collider_noise_level = world_config.consts["collider_noise_level"]
        self.viscosity = world_config.consts["viscosity"]
        self.max_particles = world_config.consts["max_particles"]
        self.gravity = np.array(world_config.consts["gravity"])

    def colliders_count(self, particle_index: int) -> int:
        return self.colliders[particle_index].shape[0]

    @property
    def diameter(self) -> float:
        return self.particle_radius * 2

    @property
    def segments(self) -> NDArray:
        return np.vstack(rigid_body.segments for rigid_body in self.rigid_bodies)

    @property
    def segment_velocities(self) -> NDArray:
        return np.vstack(
            [rigid_body.center_velocity] * rigid_body.segments.shape[0] for rigid_body in self.rigid_bodies
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
            self.add_virtual_collider_pressure_and_overlap()

        with self.debug_timer("Forces"):
            self.apply_gravity()
            self.apply_pressure()
            # self.apply_spring()
            self.apply_viscosity()
            self.apply_tension()
            self.apply_wall_bounce()
            self.apply_particles_velocity()

        self.debug_prints = self.debug_timer.report()
        self.debug_prints += f"\n\n{self.collect_debug_metrics()}"
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
        self.colliders = []
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
            self.colliders.append(relative_colliders / collider_distances[:, None])
            self.collider_velocities.append(self.particle_velocities[collider_indices])

    def fix_clipping_particles(self, distances: NDArray, nearest_point_in_segment: NDArray) -> None:
        """
            fix this:
                     segment
                     |
                p    |
            x   *----+
        radius       |
                     |
            to this:

                     segment
                     |
            p        |
            *--------+
        radius       |
                     |
        """
        # P
        clipping_particles_mask = np.any(distances < self.particle_radius, 1)
        if not any(clipping_particles_mask):
            return
        # L (clipped)
        clipped_segments = np.argmin(distances[clipping_particles_mask, :], 1)
        # L x 2
        contact_point = nearest_point_in_segment[clipping_particles_mask, clipped_segments, :]
        # L
        distance_to_contact = distances[clipping_particles_mask, clipped_segments]
        # L x 2
        correction_direction = self.particles[clipping_particles_mask] - contact_point

        self.particles[clipping_particles_mask] += (
                correction_direction * (self.particle_radius - distance_to_contact[:, None]) / distance_to_contact[:,
                                                                                               None]
        )

    def add_wall_virtual_colliders(self) -> None:
        self.virtual_colliders = []
        self.virtual_colliders_velocity = []
        nearest_point_in_segment, distances = points_to_segments_distance(self.particles, self.segments)
        self.fix_clipping_particles(distances=distances, nearest_point_in_segment=nearest_point_in_segment)
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
                virtual_colliders = particle - particle_segment_contacts
                self.colliders[particle_index] = np.vstack([self.colliders[particle_index], virtual_colliders * 2])
                self.collider_velocities[particle_index] = np.vstack(
                    (self.collider_velocities[particle_index], self.segment_velocities[touching_segments_mask])
                )
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
        assert all(n >= 0 for n in particles_pressure)
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
                    self.colliders[particle_index]
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
                self.dt * self.spring_amplifier * spring_pull[:, None] * self.colliders[particle_index], 0
            )

    def apply_tension(self) -> None:
        # P x 2
        surface_normals = np.zeros((self.particle_count, 2))
        for i in range(self.particle_count):
            if len(self.collider_overlaps[i]) == 0:
                continue
            surface_normals[i] = np.sum(
                ((1 - self.collider_overlaps[i]) * self.collider_overlaps[i])[:, None] * self.colliders[i], 0)
        for i in range(self.particle_count):
            if len(self.collider_overlaps[i]) == 0:
                continue
            # C x 2
            normal_deltas = surface_normals[i][None] - surface_normals[self.colliders_indices[i]]
            # C
            normals_alignment = np.sum(normal_deltas * self.colliders[i], 1) * self.surface_smoothing
            # C
            target_pressure_fix = self.collider_pressures[i] + self.particles_pressure[i] - 2 * self.target_pressure
            self.particle_velocities[i] += self.dt * np.sum(
                (normals_alignment + target_pressure_fix)[:, None] * self.colliders[i], 0)

        # s[i] = sum((1 - w[i, j]) * w[i, j] * n[i, j], j)
        # A[i] = a * (w[i] + w[j] - 2 * w0)
        # B[i] = b * dot(s[j] - s[i], n[i, j]))
        # v[i] ← v[i] - Δt * (A[i] + B[i]) * n[i, j]

    def apply_particles_velocity(self) -> None:
        self.particles += self.dt * self.particle_velocities

    def apply_bodies_velocity(self) -> None:
        for rigid_body in self.rigid_bodies:
            rigid_body.apply_velocity(self.dt)

    def collect_debug_metrics(self) -> str:
        return yaml.dump(
            [
                {"particle_radius": self.particle_radius},
                {"dt": self.dt},
                {"wall_collision_decay": self.wall_collision_decay},
                {"spring_overlap_balance": self.spring_overlap_balance},
                {"spring_amplifier": self.spring_amplifier},
                {"pressure_amplifier": self.pressure_amplifier},
                {"ignored_pressure": self.ignored_pressure},
                {"collider_noise_level": self.collider_noise_level},
                {"viscosity": self.viscosity},
                {"max_particles": self.max_particles},
                {"gravity": str(self.gravity.tolist())},
            ]
        )

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from nptyping import NDArray

Segment = tuple[float, float]

# Position = Union[tuple[float, float], list[float, float], NDArray[float]]


@dataclass
class RigidBody:
    segments: NDArray  # segments x dots(2) x dims(2)
    name: str = ""
    center_velocity: NDArray = np.array([0.0, 0.0])
    angular_velocity: float = 0.00
    scale: list[float] = field(default_factory=lambda: [1.0, 1.0])
    position: list[float] = field(default_factory=lambda: [0.0, 0.0])

    def segment_velocities(self) -> NDArray:
        # K x 2
        segment_positions_to_center = np.mean(self.segments, 1) - self.center_velocity[None]
        segment_distances_to_center = np.hypot(segment_positions_to_center[:, 0], segment_positions_to_center[:, 1])
        segment_positions_to_velocity_direction = segment_positions_to_center[:, [1, 0]] * np.array([[1, -1]])
        return (
                self.center_velocity
                + segment_positions_to_velocity_direction * segment_distances_to_center[None] * self.angular_velocity
        )

    def place_in_world(self):
        self.segments *= np.array(self.scale)[None]
        self.segments += np.array(self.position)[None]

    @property
    def center(self):
        return np.mean(self.segments, (0, 1))

    def apply_velocity(self, dt: float) -> None:
        self.segments += dt * self.center_velocity[None, None]
        self.rotate_object_around_center(dt * self.angular_velocity)

    def rotate_object_around_center(self, theta) -> None:
        center = self.center
        new_segments = np.zeros_like(self.segments)
        new_segments[:, :, 0] = (
                np.cos(theta) * (self.segments[:, :, 0] - center[None, 0])
                - np.sin(theta) * (self.segments[:, :, 1] - center[None, 1])
                + center[None, 0]
        )

        new_segments[:, :, 1] = (
                np.sin(theta) * (self.segments[:, :, 0] - center[None, 0])
                + np.cos(theta) * (self.segments[:, :, 1] - center[None, 1])
                + center[None, 1]
        )
        self.segments = new_segments


@dataclass
class FixedRigidBody(RigidBody):
    def apply_velocity(self, dt: float) -> None:
        ...


@dataclass
class MotoredRigidBody(RigidBody):
    velocity_func: Callable[[float], NDArray] = lambda t: np.array([0.0, 0.0])
    angular_velocity_func: Callable[[float], float] = lambda t: 0
    time_from_start: float = 0.0

    def apply_velocity(self, dt: float) -> None:
        self.time_from_start += dt
        self.velocity = self.velocity_func(self.time_from_start)  # noqa
        self.angular_velocity = self.angular_velocity_func(self.time_from_start)  # noqa
        super(MotoredRigidBody, self).apply_velocity(dt)

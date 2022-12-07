from dataclasses import dataclass
from typing import Callable

import numpy as np
from nptyping import NDArray

Segment = tuple[float, float]


# Position = Union[tuple[float, float], list[float, float], NDArray[float]]


@dataclass
class RigidBody:
    segments: NDArray  # segments x dots(2) x dims(2)
    name: str = ""
    mass: float = 1.0
    velocity: NDArray = np.array([0.0, 0.0])
    angular_velocity: float = 0.00

    @property
    def center(self):
        return np.mean(self.segments, (0, 1))

    def apply_velocity(self, dt: float) -> None:
        self.segments += dt * self.velocity[None, None]
        center = self.center
        new_segments = np.zeros_like(self.segments)
        new_segments[:, :, 0] = (
                np.cos(self.angular_velocity) * (self.segments[:, :, 0] - center[None, 0])
                - np.sin(self.angular_velocity) * (self.segments[:, :, 1] - center[None, 1])
                + center[None, 0]
        )

        new_segments[:, :, 1] = (
                np.sin(self.angular_velocity) * (self.segments[:, :, 0] - center[None, 0])
                + np.cos(self.angular_velocity) * (self.segments[:, :, 1] - center[None, 1])
                + center[None, 1]
        )

        self.segments = new_segments + dt * self.velocity[None, None]


@dataclass
class FixedRigidBody(RigidBody):
    def apply_velocity(self, dt: float) -> None:
        ...


@dataclass
class MotoredRigidBody(RigidBody):
    velocity_func: Callable[[float], NDArray] = lambda x: np.array([0.0, 0.0])
    angular_velocity_func: Callable[[float], float] = lambda x: 0
    start_time: float = 0.0

    def apply_velocity(self, dt: float) -> None:
        self.start_time += dt
        self.velocity = self.velocity_func(self.start_time)  # noqa
        self.angular_velocity = self.angular_velocity_func(self.start_time)  # noqa
        super(MotoredRigidBody, self).apply_velocity(dt)

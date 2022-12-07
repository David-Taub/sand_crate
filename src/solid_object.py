from dataclasses import dataclass

import numpy as np
from nptyping import NDArray

Segment = tuple[float, float]


# Position = Union[tuple[float, float], list[float, float], NDArray[float]]


@dataclass
class SolidObject:
    segments: NDArray  # segments x dots(2) x dims(2)
    name: str = ""
    is_fixed: bool = True
    mass: float = 1.0
    velocity: NDArray = np.array([0.0, 0.0])
    angular_velocity: float = 0.01

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

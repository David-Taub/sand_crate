from dataclasses import dataclass

import numpy as np
import yaml

from particle_source import ParticleSource
from rigid_body import MotoredRigidBody, FixedRigidBody, RigidBody, BODIES_CONFIG_FILE_PATH

BODY_TYPE_TO_CLASS = {"motored": MotoredRigidBody, "fixed": FixedRigidBody, "free": RigidBody}


@dataclass
class WorldConfig:
    rigid_bodies: list[RigidBody]
    particle_sources: list[ParticleSource]
    consts: dict


def load_world_config() -> WorldConfig:
    with open(BODIES_CONFIG_FILE_PATH, "r") as f:
        world_config = yaml.safe_load(f)
    return WorldConfig(
        rigid_bodies=build_rigid_bodies(world_config.get("rigid_bodies", [])),
        particle_sources=build_particle_sources(world_config.get("particle_sources")),
        consts=world_config.get("consts"),
    )


def build_particle_sources(particle_source_configs):
    return [ParticleSource(**config) for config in particle_source_configs]


def build_rigid_bodies(body_configs: list) -> list[RigidBody]:
    rigid_bodies = []
    for body_config in body_configs:
        body_type, kwargs = next(iter(body_config.items()))
        body_class = BODY_TYPE_TO_CLASS[body_type]
        if "segments" in kwargs:
            kwargs["segments"] = np.array(kwargs["segments"])
        if "velocity" in kwargs:
            kwargs["velocity"] = np.array(kwargs["velocity"])
        if "velocity_func" in kwargs:
            kwargs["velocity_func"] = eval(kwargs["velocity_func"])
        if "angular_velocity_func" in kwargs:
            kwargs["angular_velocity_func"] = eval(kwargs["angular_velocity_func"])
        rigid_body = body_class(**kwargs)
        rigid_body.place_in_world()
        rigid_bodies.append(rigid_body)

    return rigid_bodies

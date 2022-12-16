import numpy as np
from nptyping import NDArray

from typings import Particles, ParticlesNeighbors

MAX_ALLOWED_NEIGHBORS = 20


def detect_particle_collisions(particles: Particles, diameter: float) -> list[ParticlesNeighbors]:
    """
    Example:
    +------------------------------------------------+
    |       o                                        |
    |                  diameter                      | strip 0
    |-----------------+--------+---------------------|
    |            o   >X<   x   |                 o   |
    |     o           |  x    x|         o           | strip 1  <- current strip x
    |--------+--------+--------+---------------------|
    |        |      x |       x|     o               |
    |        |x   x   |  x     |   o            o    | strip 2  <- next strip x
    |--------+--------+--------+---------------------|
    |    o                o                    o     |
    |                o             o                 | strip 3
    +------------------------------------------------+

    Legend
    --------
    >X<  Particle to which collisions are searched
    x    Neighbor
    o    Non collision
    """
    # calc particles indices that are less than DIAMETER apart
    particles, y_floored, indices = strip_sort_particles(particles=particles, diameter=diameter)
    unique_ys, strip_start_indices = np.unique(y_floored, return_index=True)
    strip_start_indices = np.append(strip_start_indices, len(y_floored))
    strip_start_indices = np.append(strip_start_indices, len(y_floored))
    collisions: list[ParticlesNeighbors] = []
    for strip_start_index, next_strip_start_index, next_strip_end_index in zip(
            strip_start_indices, strip_start_indices[1:], strip_start_indices[2:]
    ):
        detect_collisions_for_particles_in_strip(
            particles, collisions, strip_start_index, next_strip_start_index, next_strip_end_index, diameter
        )
    add_reverse_collisions(collisions)
    trim_collisions(collisions, MAX_ALLOWED_NEIGHBORS)
    collisions = np.array(collisions)
    reversed_indices_collisions = [
        [indices[i] for i in particle_collisions] for particle_collisions in collisions[np.argsort(indices)]
    ]
    return reversed_indices_collisions


def detect_collisions_for_particles_in_strip(
        particles: Particles,
        collisions: list[ParticlesNeighbors],
        strip_start_index: int,
        next_strip_start_index: int,
        next_strip_end_index: int,
        diameter: float,
) -> None:
    strip_x = particles[strip_start_index:next_strip_start_index, 0]
    next_strip_x = particles[next_strip_start_index:next_strip_end_index, 0]
    for particle_index_in_strip, particle_x in enumerate(strip_x):
        particle_collisions = find_collisions_of_particle(
            particle_x,
            particle_index_in_strip,
            strip_x,
            next_strip_x,
            strip_start_index,
            next_strip_start_index,
            diameter,
        )
        # TODO - tradeoff, slower collision detection, for fewer collisions. see if commenting out these lines
        # is faster
        #############################################################
        particle_collisions = np.array(particle_collisions)
        if len(particle_collisions) > 0:
            d = particles[particle_collisions, :] - particles[particle_index_in_strip + strip_start_index, :]
            distances = np.hypot(d[:, 0], d[:, 1])
            particle_collisions = particle_collisions[distances <= diameter]
        particle_collisions = particle_collisions.tolist()
        #############################################################
        collisions.append(particle_collisions)


def add_reverse_collisions(collisions: list[ParticlesNeighbors]) -> None:
    for particle_index, particle_collisions in reversed(list(enumerate(collisions))):
        for collision_index in reversed(particle_collisions):
            collisions[collision_index].append(particle_index)


def trim_collisions(collisions: list[ParticlesNeighbors], max_allowed_collisions: int) -> None:
    for particle_index, particle_collisions in enumerate(collisions):
        collisions[particle_index] = particle_collisions[:max_allowed_collisions]


def find_collisions_of_particle(
        particle_x: float,
        particle_index_in_strip: int,
        strip_x: list[float],
        next_strip_x: list[float],
        strip_start_index: int,
        next_strip_start_index: int,
        diameter: float,
) -> ParticlesNeighbors:
    # current strip
    end_of_collisions_index_in_strip = np.searchsorted(strip_x, particle_x + diameter, side="right")
    collisions_in_strip = list(
        range(strip_start_index + particle_index_in_strip + 1, strip_start_index + end_of_collisions_index_in_strip)
    )

    # next strip
    start_of_collisions_index_in_next_strip = np.searchsorted(next_strip_x, particle_x - diameter, side="left")
    end_of_collisions_index_in_next_strip = np.searchsorted(next_strip_x, particle_x + diameter, side="right")
    collisions_in_next_strip = list(
        range(
            next_strip_start_index + start_of_collisions_index_in_next_strip,
            next_strip_start_index + end_of_collisions_index_in_next_strip,
        )
    )

    return collisions_in_strip + collisions_in_next_strip


def strip_sort_particles(particles: Particles, diameter: float) -> tuple[Particles, list[float], NDArray]:
    # sorts first by X, then by Y_floored
    y_floored = np.floor(particles[:, 1] / diameter).astype(int)
    sorted_indices = np.lexsort((particles[:, 0], y_floored))
    return particles[sorted_indices, :], y_floored[sorted_indices], sorted_indices

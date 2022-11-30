import numpy as np

from crate import DIAMETER
from typings import Particles, ParticlesNeighbors


def stip_sort_particles(particles: Particles, diameter: float) -> tuple[Particles, list[float]]:
    # sorts first by X, then by Y_floored
    y_floored = np.floor(particles[:, 1] / diameter)
    sorted_indices = np.lexsort((y_floored, particles[:, 0]))
    y_floored = y_floored[sorted_indices]
    return particles[sorted_indices, :], y_floored


def detect_particle_collision(particles: Particles, diameter: float = DIAMETER) -> list[ParticlesNeighbors]:
    """
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
    """
    # calc particles indices that are less than DIAMETER apart
    particles, y_floored = stip_sort_particles(particles=particles, diameter=diameter)
    unique_ys, strip_start_indices = np.unique(y_floored, return_index=True)
    strip_start_indices.append(len(y_floored))
    collisions: list[ParticlesNeighbors] = []
    for strip_start_index, next_strip_start_index, next_strip_end_index in zip(strip_start_indices,
                                                                               strip_start_indices[1:],
                                                                               strip_start_indices[2:]):
        strip_x = particles[strip_start_index:, next_strip_start_index, 0]
        next_strip_x = particles[next_strip_start_index:next_strip_end_index, 0]
        for particle_index_in_strip, particle_x in strip_x:
            particle_neighbors = find_neighbors_of_particle(particle_x, particle_index_in_strip, strip_x,
                                                            next_strip_x, strip_start_index, next_strip_start_index,
                                                            diameter)
            collisions.append(particle_neighbors)
    add_reverse_colllisions(collisions)
    return collisions


def add_reverse_colllisions(collisions: list[ParticlesNeighbors]) -> None:
    for particle_index, particle_neighbors in collisions[::-1]:
        for neighbor_index in particle_neighbors[::-1]:
            collisions[neighbor_index].append(particle_index)


def find_neighbors_of_particle(particle_x: float, particle_index_in_strip: int, strip_x: list[float],
                               next_strip_x: list[float], strip_start_index: int, next_strip_start_index: int,
                               diameter: float) -> ParticlesNeighbors:
    # current strip
    end_of_neighbors_index_in_strip = np.searchsorted(strip_x, particle_x + diameter, side='right')
    neighbors_in_strip = list(range(strip_start_index + particle_index_in_strip + 1,
                                    strip_start_index + end_of_neighbors_index_in_strip))

    # next strip
    start_of_neighbors_index_in_next_strip = np.searchsorted(next_strip_x, particle_x - diameter, side='left')
    end_of_neighbors_index_in_next_strip = np.searchsorted(next_strip_x, particle_x + diameter, side='right')
    neighbors_in_next_strip = list(range(next_strip_start_index + start_of_neighbors_index_in_next_strip,
                                         next_strip_start_index + end_of_neighbors_index_in_next_strip))
    return neighbors_in_strip + neighbors_in_next_strip

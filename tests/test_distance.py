import itertools
from math import ceil, floor

import numpy as np
import pytest
from collision_detector import detect_particle_neighbors

from src.crate.utils.geometry_utils import points_to_segments_distance

PARTICLES_COUNT = 35
SEGMENTS_COUNT = 5

np.random.seed(0)


def test_row_distance():
    p = np.array([[i, 0] for i in range(PARTICLES_COUNT)])
    segments = np.array([[[i, -1], [i, 1]] for i in range(SEGMENTS_COUNT)])

    points, distances = points_to_segments_distance(p, segments)

    assert distances.shape == (PARTICLES_COUNT, SEGMENTS_COUNT)
    for i in range(SEGMENTS_COUNT):
        for j in range(PARTICLES_COUNT):
            assert distances[j, i] == abs(j - i)


@pytest.fixture(params=[(0.5, 0, 0), (1, 1, 2), (2, 2, 4)])
def diameter_and_neighbors_row(request):
    return request.param


@pytest.fixture(params=[(0.5, 0, 0), (1, 2, 4), (2, 5, 12)])
def diameter_and_neighbors_grid(request):
    return request.param


def test_collider_particles_row(diameter_and_neighbors_row):
    diameter, min_neighbors, max_neighbors = diameter_and_neighbors_row
    p = np.array([[i, 0] for i in range(PARTICLES_COUNT)])
    particles_neighbors = detect_particle_neighbors(p, diameter)
    for i, n in enumerate(particles_neighbors):
        for j in range(max(0, ceil(i - diameter)), min(floor(i + diameter), PARTICLES_COUNT - 1)):
            assert j in n or j == i
    assert len(particles_neighbors) == p.shape[0]
    assert all(min_neighbors <= len(n) <= max_neighbors for n in particles_neighbors)
    assert any(min_neighbors == len(n) for n in particles_neighbors)
    assert any(max_neighbors == len(n) for n in particles_neighbors)


def test_collider_particles_grid(diameter_and_neighbors_grid):
    diameter, min_neighbors, max_neighbors = diameter_and_neighbors_grid
    p = np.array([[i, j] for i, j in itertools.product(range(PARTICLES_COUNT), range(PARTICLES_COUNT))])
    particles_neighbors = detect_particle_neighbors(p, diameter)
    assert len(particles_neighbors) == p.shape[0]
    assert all(min_neighbors <= len(n) <= max_neighbors for n in particles_neighbors)
    assert any(min_neighbors == len(n) for n in particles_neighbors)
    assert any(max_neighbors == len(n) for n in particles_neighbors)


def test_collider_random_space():
    diameter = 0.1
    ps = np.random.rand(PARTICLES_COUNT, 2)
    particles_neighbors = detect_particle_neighbors(ps, diameter)
    for i, p in enumerate(ps):
        if len(particles_neighbors[i]) == 0:
            continue
        colliders = ps[particles_neighbors[i]] - p
        distances = np.hypot(colliders[:, 0], colliders[:, 1])
        assert all(d <= diameter * 3 for d in distances), f"{distances} exceed {diameter * 3}"

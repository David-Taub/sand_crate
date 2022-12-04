import numpy as np

from neighbor_detector import detect_particle_neighbors
from src.geometry_utils import points_to_segments_distance

PARTICLES_COUNT = 5
SEGMENTS_COUNT = 3


def test_lin_distance():
    p = np.array([[i, 0] for i in range(PARTICLES_COUNT)])
    a = np.array([[i, 1] for i in range(SEGMENTS_COUNT)])
    b = np.array([[i, -1] for i in range(SEGMENTS_COUNT)])

    distances = points_to_segments_distance(p, a, b)

    assert distances.shape == (PARTICLES_COUNT, SEGMENTS_COUNT)
    for i in range(SEGMENTS_COUNT):
        for j in range(PARTICLES_COUNT):
            assert distances[j, i] == abs(j - i)


def test_collider_sparse_row():
    p = np.array([[i, 0] for i in range(PARTICLES_COUNT)])
    particles_neighbors = detect_particle_neighbors(p, 0.5)
    assert all(len(n) == 0 for n in particles_neighbors)


def test_collider_singular_row():
    p = np.array([[i, 0] for i in range(PARTICLES_COUNT)])
    particles_neighbors = detect_particle_neighbors(p, 1)
    assert all(len(n) in [1, 2] for n in particles_neighbors)


def test_collider_dense_row():
    p = np.array([[i, 0] for i in range(PARTICLES_COUNT)])
    particles_neighbors = detect_particle_neighbors(p, 2)
    assert all(len(n) in [1, 2, 3, 4] for n in particles_neighbors)

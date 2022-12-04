import numpy as np

from src.geometry_utils import lineseg_dists

PARTICLES_COUNT = 5
SEGMENTS_COUNT = 3


def test_lin_distance():
    p = np.array([[i, 0] for i in range(PARTICLES_COUNT)])
    a = np.array([[i, 1] for i in range(SEGMENTS_COUNT)])
    b = np.array([[i, -1] for i in range(SEGMENTS_COUNT)])

    distances = lineseg_dists(p, a, b)

    assert distances.shape == (PARTICLES_COUNT, SEGMENTS_COUNT)
    for i in range(SEGMENTS_COUNT):
        for j in range(PARTICLES_COUNT):
            assert distances[j, i] == abs(j - i)

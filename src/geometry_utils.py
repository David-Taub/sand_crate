import numpy as np

from typings import Particles, Distances


def points_to_segments_distance(p: Particles, a: Particles, b: Particles) -> Distances:
    """Cartesian distance from point to line segment

    Edited to support arguments as series, from:
    https://stackoverflow.com/a/54442561/11208892

    a *-----------------------* b
       \                 ___/
        \           ___/
         \     ___/
          \  /
           *
            P

    Args:
        - p: np.array of shape (K, 2)
        - a: np.array of shape (N, 2)
        - b: np.array of shape (N, 2)
    """
    K = p.shape[0]
    N = a.shape[0]
    # normalized tangent vectors
    # N x 2
    d_ba = b - a
    # N x 2
    d = np.divide(d_ba, (np.hypot(d_ba[:, 0], d_ba[:, 1]).reshape(-1, 1)))

    # signed parallel distance components
    # rowwise dot products of 2D vectors
    # N x K x 2
    p_r = np.repeat(p[:, :, np.newaxis], N, axis=2).transpose((0, 2, 1))
    a_r = np.repeat(a[:, :, np.newaxis], K, axis=2).transpose((2, 0, 1))
    b_r = np.repeat(b[:, :, np.newaxis], K, axis=2).transpose((2, 0, 1))
    d_r = np.repeat(d[:, :, np.newaxis], K, axis=2).transpose((2, 0, 1))

    # N x K
    s = ((a_r - p_r) * d_r).sum(axis=2)
    t = ((p_r - b_r) * d_r).sum(axis=2)

    # N x K
    # clamped parallel distance
    h = np.maximum.reduce([s, t, np.zeros(s.shape)])

    # perpendicular distance component
    # rowwise cross products of 2D vectors
    # N x K x 2
    d_pa = p_r - a_r
    # N x K x 2
    c = d_pa[:, :, 0] * d_r[:, :, 1] - d_pa[:, :, 1] * d_r[:, :, 0]

    return np.hypot(h, c)

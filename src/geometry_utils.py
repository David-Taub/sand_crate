import numpy as np

from typings import Particles


def points_to_segments_distance(p: Particles, segments):
    """Cartesian distance from point to line segment

    Edited to support arguments as series, from:
    https://stackoverflow.com/a/54442561/11208892
           d
    a *----+------------------* b
       \   |             ___/
        \  | dist   ___/
         \ |   ___/
          \| /
           *
            P

    Args:
        - p: np.array of shape (K, 2)
        - a: np.array of shape (N, 2)
        - b: np.array of shape (N, 2)
    """
    a = segments[:, 0, :]
    b = segments[:, 1, :]
    # 1 x N x 2
    ab = (b - a)[None]
    # K x N x 2
    ap = p[:, None] - a[None]
    projected_p = ap * ab
    # K x N
    rate_projected_p = projected_p.sum(2) / (ab * ab).sum(2)
    nx = np.clip(rate_projected_p, 0, 1)
    # K x N x 2
    d = ab * nx[:, :, None] + a[None]
    pd = d - p[:, None]
    distance = np.hypot(pd[:, :, 0], pd[:, :, 1])
    return d, distance

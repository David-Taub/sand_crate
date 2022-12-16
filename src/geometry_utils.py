import numpy as np
from nptyping import NDArray

from typings import Particles


def points_to_segments_distance(p: Particles, segments):
    """Cartesian distance from point to line segment
           d
    a *----+------------------* b
       \   |             ___/
        \  | dist   ___/
         \ |   ___/
          \| /
           *
            P

    Args:
        - p: np.array of shape (P, 2)
        - a: np.array of shape (S, 2)
        - b: np.array of shape (S, 2)
    Return:
     P x S matrix of distances
    """
    a = segments[:, 0, :]
    b = segments[:, 1, :]
    # 1 x S x 2
    ab = (b - a)[None]
    # P x S x 2
    ap = p[:, None] - a[None]
    projected_p = ap * ab
    # P x S
    rate_projected_p = projected_p.sum(2) / (ab * ab).sum(2)
    nx = np.clip(rate_projected_p, 0, 1)
    # P x S x 2
    d = ab * nx[:, :, None] + a[None]
    pd = d - p[:, None]
    distance = np.hypot(pd[:, :, 0], pd[:, :, 1])
    return d, distance


def segments_to_segments_distance(a, b, c, d):
    """Given two lines defined by numpy.array pairs (a0,a1,b0,b1)
    Return the closest points on each segment and their distance
    """

    # If clampAll=True, set all clamps to True
    # Calculate denominator
    ab = b - a
    ac = c - a
    ad = d - a
    bc = c - b
    cd = d - c
    bd = d - b
    ab_norm = np.linalg.norm(ab)
    cd_norm = np.linalg.norm(cd)

    ab_normalized = ab / ab_norm
    cd_normalized = cd / cd_norm

    ab_cd_cross = np.cross(ab_normalized, cd_normalized)
    ab_cd_sin_square = np.linalg.norm(ab_cd_cross) ** 2

    # If they don't overlap then there is a closest point solution.
    # If they do overlap, there are infinite closest positions, but there is a closest distance
    if ab_cd_sin_square == 0:
        # ac and cd are on the same line
        ab_norm_dot_ac = np.dot(ab_normalized, ac)
        ab_norm_dot_ad = np.dot(ab_normalized, ad)
        if ab_norm_dot_ac <= 0 >= ab_norm_dot_ad:
            # angles BAC and BAD are more than 90 degrees
            if np.absolute(ab_norm_dot_ac) < np.absolute(ab_norm_dot_ad):
                # dc->ab
                return a, c, np.linalg.norm(ac)
            else:
                # cd->ab
                return a, d, np.linalg.norm(ad)

        elif ab_norm_dot_ac >= ab_norm <= ab_norm_dot_ad:
            # angles BAC and BAD are less than 90 degrees, and cd is after ab
            if np.absolute(ab_norm_dot_ac) < np.absolute(ab_norm_dot_ad):
                # ab->cd
                return b, c, np.linalg.norm(bc)
            else:
                # ab->dc
                return b, d, np.linalg.norm(bd)

        else:
            # ab and bd overlap,
            ac_projected_on_ab = ab_norm_dot_ac * ab_normalized
            return None, None, np.linalg.norm(a + ac_projected_on_ab - c)

    # TODO: convert to 2d
    # non parallel segments
    detA = np.linalg.det([ac, cd_normalized, ab_cd_cross])
    detB = np.linalg.det([ac, ab_normalized, ab_cd_cross])

    t0 = detA / ab_cd_sin_square
    t1 = detB / ab_cd_sin_square

    pA = a + (ab_normalized * t0)  # Projected closest point on segment A
    pB = c + (cd_normalized * t1)  # Projected closest point on segment B

    if t0 < 0:
        pA = a
    elif t0 > ab_norm:
        pA = b

    if t1 < 0:
        pB = c
    elif t1 > cd_norm:
        pB = d

    # Clamp projection A
    if t0 < 0 or t0 > ab_norm:
        dot = np.dot(cd_normalized, (pA - c))
        if dot < 0:
            dot = 0
        elif dot > cd_norm:
            dot = cd_norm
        pB = c + (cd_normalized * dot)

    # Clamp projection B
    if t1 < 0 or t1 > cd_norm:
        dot = np.dot(ab_normalized, (pB - a))
        if dot < 0:
            dot = 0
        elif dot > ab_norm:
            dot = ab_norm
        pA = a + (ab_normalized * dot)

    return pA, pB, np.linalg.norm(pA - pB)


def cross_2d(v1: NDArray, v2: NDArray) -> NDArray:
    # N x 2, N x 2 -> N
    return v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]


def calc_cross_coefficient(a, ab, c, cd):
    # all - N x 2 -> N
    return cross_2d(a - c, cd) / cross_2d(cd, ab)


def pad_segments(segments: NDArray, pad_distance: float) -> NDArray:
    # K x 2 x 2 -> 2K x 2 x 2
    a = segments[:, 0, :]
    b = segments[:, 1, :]
    ab = b - a
    prep_ab = np.hstack((ab[:, 1, None], -ab[:, 0, None]))
    offset = prep_ab * pad_distance / np.hypot(prep_ab[:, 0], prep_ab[:, 1])[:, None]
    padded_segments1 = np.hstack(((a + offset)[:, None], (b + offset)[:, None]))
    padded_segments2 = np.hstack(((a - offset)[:, None], (b - offset)[:, None]))
    return np.vstack((padded_segments1, padded_segments2))


def segments_crossings(segments1: NDArray, segments2: NDArray) -> NDArray:
    """
         * C
         |
    A    |     B
    *----+-----*
         |
         |
         * D
    """
    # N x 2
    a = segments1[:, 0, :]
    b = segments1[:, 1, :]
    # K x 2
    c = segments2[:, 0, :]
    d = segments2[:, 1, :]

    # Find the 4 orientations required for
    # the general and special cases
    return np.logical_and(orientation(a, b, c) != orientation(a, b, d),
                          (orientation(c, d, a) != orientation(c, d, b)).T)


def orientation(p: NDArray, q: NDArray, r: NDArray) -> NDArray:
    # -1 -> ccw, 1 -> cw, 0 -> on line
    # p - N x 2
    # q - N x 2
    # r - K x 2
    # return - N x K
    return np.sign(((q[:, 1, None] - p[:, 1, None]) * (r[None, :, 0] - q[:, 0, None])) - \
                   ((q[:, 0, None] - p[:, 0, None]) * (r[None, :, 1] - q[:, 1, None])))

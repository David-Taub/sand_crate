import numpy as np
from nptyping import NDArray

from .types import Particles, Segments, Points, Vectors


def points_to_segments_distance(p: Particles, segments: Segments) -> tuple[NDArray, NDArray]:
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
    distance = np.linalg.norm(pd, axis=2)
    return d, distance


#
# def segments_to_segments_distance(a, b, c, d):
#     """Given two lines defined by numpy.array pairs (a0,a1,b0,b1)
#     Return the closest points on each segment and their distance
#     """
#
#     # If clampAll=True, set all clamps to True
#     # Calculate denominator
#     ab = b - a
#     ac = c - a
#     ad = d - a
#     bc = c - b
#     cd = d - c
#     bd = d - b
#     ab_norm = np.linalg.norm(ab)
#     cd_norm = np.linalg.norm(cd)
#
#     ab_normalized = ab / ab_norm
#     cd_normalized = cd / cd_norm
#
#     ab_cd_cross = np.cross(ab_normalized, cd_normalized)
#     ab_cd_sin_square = np.linalg.norm(ab_cd_cross) ** 2
#
#     # If they don't overlap then there is a closest point solution.
#     # If they do overlap, there are infinite closest positions, but there is a closest distance
#     if ab_cd_sin_square == 0:
#         # ac and cd are on the same line
#         ab_norm_dot_ac = np.dot(ab_normalized, ac)
#         ab_norm_dot_ad = np.dot(ab_normalized, ad)
#         if ab_norm_dot_ac <= 0 >= ab_norm_dot_ad:
#             # angles BAC and BAD are more than 90 degrees
#             if np.absolute(ab_norm_dot_ac) < np.absolute(ab_norm_dot_ad):
#                 # dc->ab
#                 return a, c, np.linalg.norm(ac)
#             else:
#                 # cd->ab
#                 return a, d, np.linalg.norm(ad)
#
#         elif ab_norm_dot_ac >= ab_norm <= ab_norm_dot_ad:
#             # angles BAC and BAD are less than 90 degrees, and cd is after ab
#             if np.absolute(ab_norm_dot_ac) < np.absolute(ab_norm_dot_ad):
#                 # ab->cd
#                 return b, c, np.linalg.norm(bc)
#             else:
#                 # ab->dc
#                 return b, d, np.linalg.norm(bd)
#
#         else:
#             # ab and bd overlap,
#             ac_projected_on_ab = ab_norm_dot_ac * ab_normalized
#             return None, None, np.linalg.norm(a + ac_projected_on_ab - c)
#
#     # TODO: convert to 2d
#     # non parallel segments
#     detA = np.linalg.det([ac, cd_normalized, ab_cd_cross])
#     detB = np.linalg.det([ac, ab_normalized, ab_cd_cross])
#
#     t0 = detA / ab_cd_sin_square
#     t1 = detB / ab_cd_sin_square
#
#     pA = a + (ab_normalized * t0)  # Projected closest point on segment A
#     pB = c + (cd_normalized * t1)  # Projected closest point on segment B
#
#     if t0 < 0:
#         pA = a
#     elif t0 > ab_norm:
#         pA = b
#
#     if t1 < 0:
#         pB = c
#     elif t1 > cd_norm:
#         pB = d
#
#     # Clamp projection A
#     if t0 < 0 or t0 > ab_norm:
#         dot = np.dot(cd_normalized, (pA - c))
#         if dot < 0:
#             dot = 0
#         elif dot > cd_norm:
#             dot = cd_norm
#         pB = c + (cd_normalized * dot)
#
#     # Clamp projection B
#     if t1 < 0 or t1 > cd_norm:
#         dot = np.dot(ab_normalized, (pB - a))
#         if dot < 0:
#             dot = 0
#         elif dot > ab_norm:
#             dot = ab_norm
#         pA = a + (ab_normalized * dot)
#
#     return pA, pB, np.linalg.norm(pA - pB)


def cross_2d(v1: Vectors, v2: Vectors) -> NDArray:
    # N x 2, N x 2 -> N
    return v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]


def calc_collision_point(a: Particles, ab: Vectors, c: Particles, cd: Vectors) -> NDArray:
    # all - N x 2 -> N
    return cross_2d(a - c, cd) / cross_2d(cd, ab)


def pad_segments(segments: Segments, pad_distance: float) -> NDArray:
    r"""
     duplicate segments, each segments becomes two parallel ones, with +\- the pad_distance relative to the original

                  orig+pad
             orig     *
     orig-pad   * b    \
          *      \      \
           \      \      \
            \      \      *
             \      * a
              *
    """
    # K x 2
    a: Points = segments[:, 0, :]
    b: Points = segments[:, 1, :]
    # K x 2
    ab: Vectors = b - a
    # K x 2 CW 90 deg
    ab_normal = rotate_vectors_clockwise_90_deg(ab)
    # K x 2
    offset = ab_normal * pad_distance / np.linalg.norm(ab_normal, axis=1)[:, None]
    # K x 2 x 2
    padded_segments1 = np.hstack(((a + offset)[:, None], (b + offset)[:, None]))
    padded_segments2 = np.hstack(((b - offset)[:, None], (a - offset)[:, None]))
    # K x 2 x 2 -> 2K x 2 x 2
    return np.vstack((padded_segments1, padded_segments2))



def rotate_vectors_clockwise_90_deg(vectors: Vectors) -> Vectors:
    # (x, y)->(y, -x)
    # points - N x 2
    return np.hstack((vectors[:, 1, None], -vectors[:, 0, None]))


def segments_crossings(segments1: Segments, segments2: Segments) -> NDArray:
    """
    segments1 - N x 2 x 2
    segments2 - K x 2 x 2

    calculates N x K binary map of whether the segments are crossing or not
    note: intersections that are overlaps on the same line are not considered crossing here

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
    # N x K x 2 -> N x K
    opposite_direction_map = np.sum(rotate_vectors_clockwise_90_deg(d - c)[None] * (b - a)[:, None], axis=2) < 0
    # N x K
    crossing_map = np.logical_and(orientation(a, b, c) != orientation(a, b, d),
                                  (orientation(c, d, a) != orientation(c, d, b)).T)
    return np.logical_and(crossing_map, opposite_direction_map)


def orientation(p: Points, q: Points, r: Points) -> NDArray:
    # p - N x 2
    # q - N x 2
    # r - K x 2
    # returns a sign map of N x K where
    # -1 -> counter clock wise
    #  1 -> clock wise
    #  0 -> on single line

    return np.sign(((q[:, 1, None] - p[:, 1, None]) * (r[None, :, 0] - q[:, 0, None])) - \
                   ((q[:, 0, None] - p[:, 0, None]) * (r[None, :, 1] - q[:, 1, None])))

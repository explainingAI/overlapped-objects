# -*- coding: utf-8 -*-
from typing import List, Iterable

from scipy import spatial
import numpy as np
import cv2


def get_nearest_point(contour, point):
    """ Get the nearest point of the contour to the point passed as parameter.

    Args:
        contour (Iterable): Set of two integers defining a coordinate. The point to find the
                          nearest one on the contour.
        point

    Returns:

    """
    mat = spatial.distance_matrix(contour, point)

    return np.argmin(mat[:, 0])


def discriminate_interest_points(interest_point, k: int, contour, img_shape: tuple, search_concave: bool = True):
    """ Discriminate between concave and convex points.

    To discriminate between concave and convex

    Args:
        interest_point:
        k:
        contour:
        search_concave:

    Returns:

    """
    concaves = []
    mask = np.zeros(img_shape, dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 1, -1)

    if len(interest_point) > 0:
        points = middle_points(contour, k, interest_point)
        types = np.zeros(len(interest_point))

        for i in range(len(interest_point)):
            types[i] = mask[int(points[i][1])][int(points[i][0])] == 0

        interest_point = np.array(interest_point)
        concaves = interest_point[types == int(search_concave)]

    return concaves


def middle_points(contour, displacement: int, middles: List[int]):
    """ Calculate the middle point.

    The midle point of a point in a contour is calculated, getting the nearest point to the left
    and to the right, and gets the middle point of them. It can be used to detect if a point is
    a concavity or a convexity.
    Args:
        contour:
        displacement (int): Displacement (to the left and to the right) of the point to get the
                            middle point.
        middles (int) : Point to calculate the middle point

    Returns:
        Numpy array with the middle points.
    """
    middles_points = []
    for middle_pint in middles:
        before_point = int(middle_pint - displacement)
        after_point = int(middle_pint + displacement)

        middles_points.append([int(np.abs(contour[after_point % len(contour)][0] +
                                          contour[before_point % len(contour)][0]) / 2),
                               int(np.abs(contour[after_point % len(contour)][1] +
                                          contour[before_point % len(contour)][1]) / 2)])

    return np.asarray(middles_points).astype(np.uint64)

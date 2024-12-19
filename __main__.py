# -*- coding: utf-8 -*-
""" Method for concave point detection from Miró et al. (2022)

Writen by: Miquel Miró Nicolau (UIB), 2020
"""
import cv2
import numpy as np

from concave import curvature as curv
from concave import regions, points


def weighted_median(data):
    """ Get the interest point inside the region.

    A regions has only one interest point. The interest point is the point is calculated with an
    approximation to a weighted median. The calculation is done with the cumulative sum of the
    curvature values. Our goal is the homogenous division in two classes of this cumulative sum.

    Args:
        data: Array with the curvature values.
    Returns:

    """
    cum_sum = np.cumsum(data)
    search = cum_sum[-1] / 2
    distances = np.abs(cum_sum - search)

    min_dist = np.argmin(distances)
    return min_dist


def concave_point_detector(contour, k: int, l_min: int, l_max: int, epsilon: float, img_shape: tuple):
    """ Calculate the concave point.

    A concave point is a point with maximum negative curvature. In this case we calculate the
    concavity of the point with the original k-slope formulation. The k-slope it depens on the axis
    that is calculated and is as follows: k-slope_x = yi - y ( i + k) / xi - x ( i + k)

    And in the other axis is the inverse division. Once we get all the slopes of every point we
    need to calculate the curvature, that is the difference betwen the slope of the point n with
    the point n+k.

    Once we have the curvature we need to binarize the data. To do it we use a recursive method
    the method has an increasing threshold on each recusion. The stop condition is that the binary
    segment from the recursion is the lenght of the segment. After that to get the concave point
    we use the weighted median to get the interest points. Finally we check if are concave or
    convex.


    Args:
        contour (contour): Object of the class contour containing all the information about the
            clump.
        k (int): Distance used to calculate the k-slope.
        l_min (int): Minimum longitude of the segment, used in the dynamic method.
        l_max (int): Maximum longitude of the segment.
        epsilon (int): Parameter for RDP approximation method.
        img_shape (int, int): Shape of the original image.

    Returns:
        A list containing the index of every concave points in the contour.
    """
    interests_points = []

    contour_org = contour[:, 0, :]
    contour = np.copy(contour)

    # Simplification of the contour with RDP algorithm
    contour = cv2.approxPolyDP(contour, epsilon, True)
    contour = contour.reshape(-1, 2)

    curvature = curv.k_curvature(contour, k)

    threshold = int(np.percentile(curvature, 25)) + 1
    curvature_binary = regions.threshold_data(curvature, threshold)

    # We check if there are at least one pixel of a region
    if curvature_binary.max() > 0:
        positions, length = regions.regions_of_interest(curvature_binary, l_min)
        positions, length = regions.refine_regions(positions, length, curvature, threshold, l_min,
                                                   l_max)

        for seg_pos, seg_length in zip(positions, length):
            interest_point = weighted_median(curvature[int(seg_pos): int(seg_pos + seg_length)])
            interest_point += seg_pos

            interests_points.append(interest_point)

    concave = points.discriminate_interest_points(interests_points, k, contour, img_shape=img_shape).astype(int)
    punts = [contour[c_idx] for c_idx in concave]

    concave = [points.get_nearest_point(contour_org, punts) for concave_point in concave]

    return concave


if __name__ == '__main__':
    img = cv2.imread("0.jpg", cv2.IMREAD_GRAYSCALE)

    img[img < 127] = 0
    img[img >= 127] = 1

    contours, _ = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    punts = concave_point_detector(contours[0], k=7, l_min=2, l_max=11, epsilon=0.2, img_shape=img.shape)
    print(punts)


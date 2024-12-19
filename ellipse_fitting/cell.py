# -*- coding: utf-8 -*-
"""
This module  contains all the classes and method to manipulate a cell image. Includes
the splitting method, the fitting and the call to the concave points.
"""

from typing import Tuple, Union, List
from enum import Enum

import cv2
import numpy as np


Num = Union[int, float]


class Tipus(Enum):
    """ Enumerate, type of cell

    """
    ELONGATED = 0
    CIRCULAR = 1
    OTHER = 2
    ERROR = 3


def find_cells(contour, concave_points: List[int], shape: Tuple[int, int], orientation: Num = 0,
               overlapped=0):
    """
    Find the ellipses in a cluster.

    A cluster is a set of cells that are overlapped and form one single convex element. From
    that convex element we try to find the ellipses. We get the concaves points as a paramater.
    Every concave point is an splitting point where two object collide- We use it to detect
    multiples segments to check if can create a ellipse.

    The segments are the basis for the creation of the ellipse. We use the coordinates in an
    algebraic system to get the parameters that define the ellipse passin though that point.

    In the case that the ellipse is not a good fit we try to concatenate other segments to
    increase the number of points for the system. The segment added to the original information,
    is circumscribed by the pairs of concave adjacent points j and j+1, where j and j+1 are not
    points of the original arc.

    Args:
        contour: Contour that defines an object.
        concave_points:
        shape: Tuple[int, int]
        orientation:
        overlapped:

    Returns:

    """
    cells = []
    cell = None

    if len(concave_points) == 0:
        cell = fit_ellipse(contour.points[:, 0], contour.points[:, 1], orientation)
        if cell is not None:
            cells.append(cell)
    else:

        segments = list(contour.iterate_over_segments(concave_points))
        segments = [{"segment": seg, "used": False} for seg in segments]

        for initial_index in range(len(concave_points)):
            final_index = ((initial_index + 1) % len(concave_points))

            segment = contour.build_segment(concave_points[initial_index],
                                            concave_points[final_index])
            cell = build_ellipse(segment, contour, orientation, overlapped)
            if cell is not None:
                cells.append(cell)
                if final_index > initial_index:
                    segments[initial_index]["used"] = True
                else:
                    segments[((final_index - 1) % len(segments))]['used'] = True
        segments = [seg for seg in segments if not seg["used"]]
        for segment in segments:
            finished = False
            i = 0
            while i < len(segments) and not finished:
                if not segments[i]['used']:
                    extra_segment = segments[i]['segment']
                    segment_conc = np.concatenate((segment['segment'], extra_segment))
                    cell = build_ellipse(segment_conc, contour, orientation, overlapped)
                    finished = cell is not None
                i = i + 1
            if cell is not None:
                cells.append(cell)

    mask_cnt = np.zeros(shape, dtype=np.uint8)
    cv2.drawContours(mask_cnt, [contour], -1, 1, -1)

    cells_type = []
    for c in cells:
        cells_type.append(_discover_type(c, mask_cnt))

    return cells, cells_type


def _discover_type(cell, contour_mask) -> Tipus:
    """

    Args:
        cell:

    Returns:

    """
    center, angle, semi_axis = cell

    ellipse_mask = np.zeros_like(contour_mask)
    ellipse_mask = cv2.ellipse(ellipse_mask, center=center,
                               axes=(int(max(semi_axis)), int(min(semi_axis))),
                               angle=np.rad2deg(-angle), startAngle=0.0, endAngle=360,
                               color=(1, 1, 1),
                               thickness=-1)

    intersection = cv2.bitwise_and(contour_mask, ellipse_mask)
    _, contours, _ = cv2.findContours(intersection, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    tipus = None
    if len(contours) > 0:
        intersection = contours[0]
        perimeter = cv2.arcLength(intersection, True)
        area = cv2.contourArea(intersection)

        ell_coeff = min(semi_axis) / max(semi_axis)
        circ_coeff = (4 * np.pi * area) / np.power(perimeter, 2)

        if ell_coeff >= 0.6 and circ_coeff < 0.8:
            tipus = Tipus.OTHER
        elif ell_coeff >= 0.6 and circ_coeff >= 0.8:
            tipus = Tipus.CIRCULAR
        elif ell_coeff < 0.6:
            tipus = Tipus.ELONGATED

    if tipus is None:
        tipus = Tipus.ERROR

    return tipus



def _correct_orientation(means: List[Num, Num], equation: Tuple[Num, Num, Num, Num, Num]):
    """ Correct the orientation of the ellipse.

    Args:
        means (List): 2D list, containing coordinates.
        equation: Values of the general equation of the ellipse.

    Returns:
        The equation corrected, the angle and the means corrected.
    """
    a, b, c, d, e = equation
    # Corrects the orientation
    phi = (1 / 2) * np.arctan(b / (c - a))
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)

    new_equation = (
        (a * np.power(cos_phi, 2)) - (b * cos_phi * sin_phi) + (c * np.power(sin_phi, 2)),
        0,
        (a * np.power(sin_phi, 2)) + (b * cos_phi * sin_phi) + (c * np.power(cos_phi, 2)),
        (d * cos_phi - e * sin_phi), (d * sin_phi + e * cos_phi))

    new_means = cos_phi * means[0] - sin_phi * means[1], sin_phi * means[
        0] + cos_phi * means[1]

    return new_equation, phi, new_means


def solve_equation(equation, means, phi=0):
    """
    Generate all the paramater from the conical equation of the ellipse

    Args:
        equation:
        means:
        phi:

    Returns:
        center (Tuple[float, float]): Center of the cell, where intersects both axis.
        angle (float): Angle between the major axis of the ellipse and the horizontal coordinate.
        semi_axis (Tuple[])
    """
    a, _, c, d, e = equation

    centers = means[0] - d / (2 * a), means[1] - e / (2 * c)
    F = 1 + (np.power(d, 2)) / (4 * a) + np.power(e, 2) / (4 * c)
    semi_axis = np.sqrt(np.abs(F / a)), np.sqrt(np.abs(F / c))

    R = np.array([[np.cos(phi), np.sin(phi)], [- np.sin(phi), np.cos(phi)]])

    center = np.squeeze(np.matmul(R, np.asarray(centers)))

    if (np.inf in center) or (-np.inf in center) or (center[0] != center[0]) or (
            center[1] != center[1]):
        return None
    else:
        return center, phi, semi_axis


def fit_ellipse(coordinates_horizontal, coordinates_vertical, orientation_tolerance: Num):
    """ Creates an ellipse from multiples coordinates.

    The general equation of the ellipse is as follow:
        ax² + bxy + cy² + dx + ey + f = 0
    With multiples coordinates, and their X and Y values we can get the values of the parameters
    a, b, c, d, e and f with simple algebraic operations. We have 6 unknown variable so we need,
    at least, 6 points to calculate it.

    Args:
        coordinates_horizontal (Iterable[int]): The horizontal points of the coordinates.
        coordinates_vertical (Iterable[int]): The vertical points of the coordinates
        orientation_tolerance (Number): Parameter from González-Hidalgo et al.
    Raises:
        ValueError if coordinates_horizontal and coordinates_vertical are not of the same lengths
    Returns:
        The cell that corresponds to the coordinates passed as parameters.
            center (Tuple[float, float]): Center of the cell, where intersects both axis.
            angle (float): Angle between the major axis of the ellipse and the horizontal.
            semi_axis (Tuple[])
    """

    if len(coordinates_vertical) != len(coordinates_horizontal):
        raise ValueError("Length of the coordinates vertical and horizontal are different")

    means = [float(np.mean(coordinates_horizontal)), float(np.mean(coordinates_vertical))]

    coordinates_horizontal = coordinates_horizontal - means[0]
    coordinates_vertical = coordinates_vertical - means[1]

    # The estimator of the conic equation of an ellipse
    equations = [np.power(coordinates_horizontal, 2),
                 np.multiply(coordinates_horizontal, coordinates_vertical),
                 np.power(coordinates_vertical, 2), coordinates_horizontal,
                 coordinates_vertical]
    equations = np.transpose(equations)
    equations = np.linalg.lstsq(np.dot(np.transpose(equations), equations),
                                np.sum(equations, axis=0),
                                rcond=None)
    equation = equations[0]
    a, b, c, _, _ = equations[0]

    phi = 0
    if min(np.abs(b / a), np.abs(b / c)) > orientation_tolerance:
        equation, phi, means = _correct_orientation(means, equation)

    if a * c > 0:
        cell = solve_equation(equation, means, phi)
    else:
        cell = None

    return cell


def build_ellipse(coordinates: np.ndarray, contours, orientation: Num, overlapped: Num):
    """

    Args:
        coordinates:
        contours:
        orientation:
        overlapped:

    Returns:
        Cell
    """
    ellipse = None
    mask = contours.mask
    if len(coordinates) > 6:
        coordinates_horizontal = coordinates[:, 0]
        coordinates_vertical = coordinates[:, 1]

        ellipse = fit_ellipse(coordinates_horizontal, coordinates_vertical, orientation)

        if ellipse is not None and not ellipse.check_fit(mask, overlapped):
            ellipse = None

    return ellipse

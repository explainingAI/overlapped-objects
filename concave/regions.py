# -*- coding: utf-8 -*-
from typing import Iterable, Union

import numpy as np

Num = Union[int, float]


def threshold_data(data, thresh):
    """ Binarize the data between 0 and 1 to the value of the threshold.


    Args:
        data:
        thresh:

    Returns:

    """

    data_binary = np.copy(data)
    data_binary[data < thresh] = 0
    data_binary[data >= thresh] = 1
    data_binary[np.isnan(data_binary)] = 0

    return data_binary


def remove_small_regions(initial_positions, lengths, min_long: int):
    """ Handles the small regions.

    A small region is a set of points smaller than the min_long. We combine these regions. If the
    regions are closer enough ( a distance lower than the min_long ) can  be combined in a new
    bigger region.

    Args:
        initial_positions:
        lengths:
        min_long:

    Returns:

    """
    for i, seg_long in enumerate(lengths):
        if (i + 1) < len(lengths):
            dist = initial_positions[i + 1] - (initial_positions[i] + seg_long)

            if dist < min_long:
                ini = initial_positions[i]
                long = ini - (initial_positions[i + 1] + lengths[i + 1])

                initial_positions[i] = np.NAN
                initial_positions[i + 1] = ini

                lengths[i] = np.NAN
                lengths[i + 1] = long

    initial_positions = initial_positions[~np.isnan(initial_positions)]
    lengths = lengths[~np.isnan(lengths)]

    aux = [(position, length) for position, length in zip(initial_positions, lengths) if
           length > min_long]

    initial_positions, lengths = list(zip(*aux))

    return initial_positions, lengths


def regions_of_interest(curvature_binary: Iterable[int], l_min):
    """ Calculate the curve segment

    A curve segment is a bunch of points that are considered curves. That curves are defined for
    the first point and for the total lenght of the curve.

    Args:
        curvature_binary (iterable) : A collection of points. 0 it means that is not curve 1 that it
        is.
        l_min(int): Minimum long of the segments.

    Returns:
        Tuple with the initial position of the curves and the longitude
    """
    in_segment = False
    segments = []

    for i, point in enumerate(curvature_binary):
        if not in_segment and point == 1:
            in_segment = True
            segments.append({"Pos": i* 1.0, "Long": 1.0})
        elif in_segment and point == 1:
            segments[-1]["Long"] += 1
        elif in_segment and point == 0:
            in_segment = False

    if in_segment:
        point = curvature_binary[0]
        while point == 1:
            segments[-1]["Long"] += 1
            point += 1
        del segments[0]

    initial_positions = np.asarray([seq['Pos'] for seq in segments])
    length = np.asarray([seq['Long'] for seq in segments])

    return remove_small_regions(initial_positions, length, l_min)


def refine_regions(position, length, data, threshold: Num, l_min, l_max):
    """ Refine the regions to have a size between the a minimum and maximum longitude.

    A region of interest is defined as a set of contiguous points, where all points have a
    curvature greater than a certain threshold, and it is defined by its start and end points.
    The process to determine the regions of interest is a recursive procedure.

    Let C and t be the set of contour points and an initial threshold for the curvature value,
    respectively. We add two thresholds for the length of the regions of interest, namely l min
    and l max . The l min value aims to avoid having an excessive number of regions and reduce
    the noise effect, and the l max value is useful to prevent that the point of interest is located
    in an excessively large region.

    Args:
        position:
        length:
        data:
        threshold:
        l_max:
        l_min:

    Returns:

    """
    position_out = []
    lenght_out = []
    for pos, long in zip(position, length):
        if long > l_max:
            seg_bin = threshold_data(data, threshold)
            seg_pos, seg_long = regions_of_interest(seg_bin, l_min)

            aux = refine_regions(seg_pos, seg_long, data, threshold + 1, l_min, l_max)

            position_out = position_out + aux[0]
            lenght_out = lenght_out + aux[1]
        else:
            position_out.append(pos)
            lenght_out.append(long)

    return position_out, lenght_out

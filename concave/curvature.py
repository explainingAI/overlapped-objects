import numpy as np


def k_slope(contour, k: int, axis: int):
    """ Calculate the slope.

    The k-slope is the derivative with k distance between the points instead of the minimum.
    The values has range between -k and +k. Is defined by the next equation:
            slope_x (i, k) = y_i - y_(i + k) / (x_i - x_ ( i + k )

    Args:
        k (int): Distance to calculate the slope.
        axis (int): Axis where it should done the calculation.

    Returns:
        Numpy array with the slope of every point of the contour.
    """

    res = []
    long_contour = len(contour)
    for i in range(long_contour):
        j = (i + k) % long_contour

        pendent = (contour[i][axis] - contour[j][axis]) / (
                contour[i][(axis + 1) % 2] - contour[j][(axis + 1) % 2])

        if np.isnan(pendent) or np.abs(pendent) == np.inf:
            pendent = k

        res.append(pendent)

    return np.asarray(res)


def k_curvature(contour, k):
    """ Calculate the curvature of a contour in a 2D problem.

    Using the k-curvature method we calculate the curvature of all the point from a contour.
    The k-curvature is calculate with the next equation::
        $ curvature_total = | curvature_horizontal | * |curvature_vertical|

    Exists another way to calculate the curvature, using the tangent. This method is an
    aproximation of the general method describe above.

    Args:
        contour:
        k (int): Distance to calculate the slope

    Returns:
        Return one numpy array with the curvature value of each point.
    """

    slope_by_axis = [k_slope(contour, k, axis) for axis in range(0, 2)]

    curvature = []
    for id_curvature, slope in enumerate(slope_by_axis):
        curvature_by_axis = []
        for i, slope_i in enumerate(slope):
            j = (i - k) % len(slope)
            pendent = (slope_i - slope[j])
            curvature_by_axis.append(pendent)
        curvature.append(curvature_by_axis)

    curvature = np.multiply(np.absolute(curvature[0]), np.absolute(curvature[1]))

    return curvature

import numpy as np
from scipy.linalg import sqrtm
from shapely.geometry import Polygon

from contour_eot.utility.utils import params_to_matrix
from contour_eot.utility.contour_sampling import state_to_ellipse_contour_pts


def gwd(s1, s2, return_parts=False):
    """
    Returns the squared Gauss Wasserstein distance for two elliptical extended object.
    Each object is parameterized by a 7D state in the form:
        [loc_x, loc_y, orientation, semi-axis 1, semi-axis 2]
    :param s1: 5D state of object 1
    :param s2: 5D state of object 2
    :param return_parts:
    :return: Squared Gauss Wasserstein distance
    """
    m1 = np.array(s1[:2])
    m2 = np.array(s1[:2])
    X1 = np.array(params_to_matrix(s1[2:]))
    X2 = np.array(params_to_matrix(s2[2:]))

    X1sqrt = sqrtm(X1.astype(float))
    C = sqrtm(X1sqrt @ X2 @ X1sqrt)

    d_center = np.linalg.norm(m1 - m2) ** 2
    d_shape = np.trace(X1 + X2 - 2 * C)
    squared_gwd = d_center + d_shape
    # for sgw == 0, rounding errors might cause a minimally negative result. set to 0 to avoid issues
    if squared_gwd < 0:
        squared_gwd = 0
    if return_parts:
        return squared_gwd, d_center, d_shape
    else:
        return squared_gwd


def iou(s1, s2):
    """
    Returns the IoU between two elliptical extended objects.
    Each object is parameterized by a 7D state in the form:
        [loc_x, loc_y, orientation, semi-axis 1, semi-axis 2]

    :param s1: 5D state of object 1
    :param s2: 5D state of object 2
    :return: IoU between objects
    """
    # prepare for "normal" IoU in an explicit manner for reuse below
    points_1 = state_to_ellipse_contour_pts(s1)
    points_2 = state_to_ellipse_contour_pts(s2)
    p1 = Polygon(points_1)
    p2 = Polygon(points_2)
    intersection = p1.intersection(p2).area
    union = p1.union(p2).area
    iou = intersection / union

    return iou

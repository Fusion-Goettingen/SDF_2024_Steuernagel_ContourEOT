import numpy as np
from scipy.optimize import root
from scipy.special import ellipeinc

from contour_eot.utility.utils import rot


def equidistant_angles_in_ellipse(num, a, b, rng=None):
    """
    Get angles of num points on an ellipse contour with minor axis length a and major axis length b
    https://stackoverflow.com/a/52062369
    """
    # Note on randomization:
    #   the sampling code always produces deterministic samples, even if 'angles' is randomly sampled from [0, 2pi]
    #   instead, we sample a lot more points, and randomly subsample back down

    # optional random prep
    if rng is not None:
        num_to_get = num
        num = num_to_get * 20

    # initial asserts
    assert (num > 0)
    assert (a <= b), "Semi Axis length a needs to be smaller than b!"

    # sampling
    angles = 2 * np.pi * np.arange(num) / num
    if a != b:
        e2 = (1.0 - a ** 2.0 / b ** 2.0)
        tot_size = ellipeinc(2.0 * np.pi, e2)
        arc_size = tot_size / num
        arcs = np.arange(num) * arc_size
        res = root(lambda x: (ellipeinc(x, e2) - arcs), angles)
        angles = res.x

    # optional random finish
    if rng is not None:
        angles = rng.choice(angles, size=num_to_get)
    return angles


def state_to_ellipse_contour_pts(s, n_pts=100):
    m = s[:2]
    p = s[2:]
    ellipse_angle_array = np.linspace(0.0, 2.0 * np.pi, n_pts)
    pts = (m[:, None] + rot(p[0]) @ np.diag([p[1], p[2]]) @ np.array(
        [np.cos(ellipse_angle_array), np.sin(ellipse_angle_array)])).T
    return np.array(pts)


def get_deterministic_ellipse_contour_measurements(state,
                                                   number_of_points=100,
                                                   equidistant_on_contour=True,
                                                   R=None,
                                                   rng=None):
    """
    Get equally spaced points on the contour of an ellipse

    :param state: 5D state [x, y, theta, semi-axis1, semi-axis2] of the ellipse
    :param number_of_points: Number of points to sample
    :param R: covariance matrix of Gaussian noise to add, or None to return exact points on the contour
    :param equidistant_on_contour: If True, points will be uniformly spaced along the contour of the ellipse. If False,
    the angles of points will be uniformly, resulting in higher density towards the outer parts of the ellipse.
    For a circle, the two are equivalent, and passing False is preferable for computational speed.
    :param rng: If R=True, use this np RNG to generate measurement noise. Set to None to create a new seeded(!) rng
    instead.
    :return: Points with equidistant spacing on ellipse contour as ndarray of shape
    """
    # prepare
    m = np.array(state[:2]).astype(float)
    p = np.array(state[2:]).astype(float)

    # angles_in_ellipse requires p[1] < p[2]
    # if this is not the case, flip them, which corresponds to a 90° rotation
    if p[1] >= p[2]:
        # flip
        p[1:] = p[1:][::-1]
        # mark in angle as rotation
        p[0] = (p[0] + np.pi / 2) % (2 * np.pi)

    # calculate angles
    if equidistant_on_contour:
        theta = equidistant_angles_in_ellipse(number_of_points, *p[1:])
    else:
        theta = np.linspace(0, 2 * np.pi, num=number_of_points, endpoint=False)
    # get points for angles
    points = m[:, None] + rot(p[0]) @ np.diag([p[1], p[2]]) @ np.array([np.cos(theta), np.sin(theta)])
    points = points.T

    # optionally, add random noise
    if R is not None:
        rng = np.random.default_rng(42) if rng is None else rng
        noise = rng.multivariate_normal(mean=[0, 0], cov=R, size=len(points))
        points += noise

    return points


def get_ellipse_contour_measurements(state,
                                     rng: np.random.Generator,
                                     number_of_points=100,
                                     equidistant_on_contour=True,
                                     R=None):
    """
    Get equally spaced points on the contour of an ellipse

    :param state: 5D state [x, y, theta, semi-axis1, semi-axis2] of the ellipse
    :param rng: np rng to be used
    :param number_of_points: Number of points to sample
    :param R: covariance matrix of Gaussian noise to add, or None to return exact points on the contour
    :param equidistant_on_contour: If True, points will be uniformly spaced along the contour of the ellipse. If False,
    the angles of points will be uniformly, resulting in higher density towards the outer parts of the ellipse.
    For a circle, the two are equivalent, and passing False is preferable for computational speed.
    :return: Points with equidistant spacing on ellipse contour as ndarray of shape
    """
    # prepare
    m = np.array(state[:2]).astype(float)
    p = np.array(state[2:]).astype(float)

    # angles_in_ellipse requires p[1] < p[2]
    # if this is not the case, flip them, which corresponds to a 90° rotation
    if p[1] >= p[2]:
        # flip
        p[1:] = p[1:][::-1]
        # mark in angle as rotation
        p[0] = (p[0] + np.pi / 2) % (2 * np.pi)

    # calculate angles
    if equidistant_on_contour:
        theta = equidistant_angles_in_ellipse(number_of_points, *p[1:], rng=rng)
    else:
        theta = rng.uniform(low=0, high=2 * np.pi, size=number_of_points)
    # get points for angles
    points = m[:, None] + rot(p[0]) @ np.diag([p[1], p[2]]) @ np.array([np.cos(theta), np.sin(theta)])
    points = points.T

    # optionally, add random noise
    if R is not None:
        noise = rng.multivariate_normal(mean=[0, 0], cov=R, size=len(points))
        points += noise

    return points

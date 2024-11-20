"""
Contains general utility functions.
"""
import numpy as np


def rot(theta):
    """
    Constructs a rotation matrix for given angle alpha.
    :param theta: angle of orientation
    :return: Rotation matrix in 2D around theta (2x2)
    """
    r = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])
    return r.reshape((2, 2))


def matrix_to_params(X):
    """Convert shape matrix X to parameter form [alpha, l1, l2] with semi-axis length"""
    assert X.shape == (2, 2), "X is not a 2x2 matrix"
    val, vec = np.linalg.eig(X)  # eigenvalue decomposition
    alpha = np.arctan2(vec[1][0], vec[0][0])  # calculate angle of orientation
    alpha = (alpha + 2 * np.pi) % (2 * np.pi)  # just in case alpha was negative
    p = [alpha, *np.sqrt(val)]
    if p[1] < p[2]:
        p[1], p[2] = p[2], p[1]
        p[0] -= np.pi / 2
    return np.array(p)


def params_to_matrix(p):
    """
    Convert parameters [alpha, l1, l2] to shape matrix X (2x2)
    """
    X = rot(p[0]) @ np.diag(np.array(p[1:]) ** 2) @ rot(p[0]).T
    return X


def scatter_matrix(Z, normalize=True):
    """
    Compute the scatter matrix for a set of measurements Z
    :param Z: measurements as Nx2 ndarray
    :param normalize: True to return normalized (division by N-1) scatter matrix, False to return non-normalized one.
    :return: Scatter matrix of Z
    """
    normalization = len(Z) - 1 if normalize else 1
    m = np.average(Z, axis=0).reshape((-1, 2))
    Z_bar = Z - m
    Z_bar = (Z_bar.T @ Z_bar) / normalization
    return Z_bar


def get_numerical_mm_factors_rotationfree(s, Z, sub=None):
    """

    :param s: 5D state
    :param Z: Set of N measurements as Nx2 array
    :param sub: None or 2x2 matrix to subtract from the resulting scatter matrix before computation of the scaling
    factors
    :return: [f1, f2] scaling factors for the two semi-axis
    """
    Z_bar = scatter_matrix(Z, normalize=True)
    if sub is not None:
        Z_bar -= sub
    return np.diag(Z_bar) / np.diag(params_to_matrix(s[2:]))

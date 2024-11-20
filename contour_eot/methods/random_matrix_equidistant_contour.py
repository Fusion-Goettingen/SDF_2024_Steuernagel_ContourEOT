"""
Extends the Random Matrix tracker with adaptive estimation of the scaling factor for contour measurements from prior

Please refer to
    A Tutorial on Multiple Extended Object Tracking
    K. Granström and M. Baum, 2022
    https://www.techrxiv.org/articles/preprint/ATutorialonMultipleExtendedObjectTracking/19115858/1
based on which this implementation has been done.

Furthermore, refer (for example) to
    Tracking of Extended Objects and Group Targets Using Random Matrices
    M. Feldmann, D. Fränken, W. Koch, 2011
    https://ieeexplore.ieee.org/document/5672614
for the RM model specifically.
"""
import numpy as np
from scipy.integrate import quad

from contour_eot.methods.random_matrix import TrackerRM
from contour_eot.utility.utils import rot


class TrackerRMEC(TrackerRM):
    """
    Implements a tracker based on the Random Matrix (RM) model
    """

    def __init__(self,
                 m,
                 P,
                 v,
                 R,
                 H=None,
                 Q=None,
                 time_step_length=1,
                 tau=10,
                 scaling_factors=None):
        """

        :param m: Initial kinematic state
        :param P: Initial kinematic state uncertainty
        :param v: Extent uncertainty
        :param R: Measurement noise
        :param H: Measurement model
        :param Q: Process noise
        :param time_step_length: time between discrete time steps
        :param tau: hyperparameter that determines decay of v. use large values if you know the object shape barely
        changes over time
        :param scaling_factors: None to adaptively estimate the semi axis scaling factors over time. Else, an array of
        length equal to dimension (i.e. 2), with fix/known scaling factors to be applied to semi axis, ordered in
        descending length of semi-axis
        """
        super().__init__(m, P, v, R, 0, H, Q, time_step_length, tau)
        self.scaling_factors = np.array(scaling_factors) if scaling_factors is not None else None

    def get_scaled_shape_matrix(self, X_hat):
        alpha, a, b = self._matrix_to_params_rm(X_hat)

        # construct scale matrix
        if self.scaling_factors is None:
            # estimate scale_matrix based on the derivations
            scale_matrix = np.zeros((2, 2))

            scale_matrix[0, 0] \
                = \
                quad(lambda t: np.sqrt(b ** 2 * np.cos(t) ** 2 + a ** 2 * np.sin(t) ** 2) * np.cos(t) ** 2, 0,
                     2 * np.pi)[0]
            scale_matrix[1, 1] \
                = \
                quad(lambda t: np.sqrt(b ** 2 * np.cos(t) ** 2 + a ** 2 * np.sin(t) ** 2) * np.sin(t) ** 2, 0,
                     2 * np.pi)[0]

            scale_matrix /= quad(lambda t: np.sqrt(b ** 2 * np.cos(t) ** 2 + a ** 2 * np.sin(t) ** 2), 0, 2 * np.pi)[0]
        else:
            # construct scale_matrix using the given fixed factors
            scale_matrix = np.diag(self.scaling_factors)

        return rot(alpha) @ np.diag([a, b]) @ scale_matrix @ np.diag([a, b]) @ rot(alpha).T

    def get_downscaled_measurement_matrix(self, Z):
        if self.scaling_factors is None:
            return Z / 0.5
        else:
            # initialize using known scaling factors
            alpha, a, b = self._matrix_to_params_rm(Z)
            scale_matrix = np.diag(1 / self.scaling_factors)  # note: 1/f hence we don't use get_scaled_shape_matrix
            return rot(alpha) @ np.diag([a, b]) @ scale_matrix @ np.diag([a, b]) @ rot(alpha).T

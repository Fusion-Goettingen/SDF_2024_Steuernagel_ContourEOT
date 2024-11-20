"""
Based on code from
A Comparison of Kalman Filter-based Approaches for Elliptic Extended Object Tracking
K. Thormann, S. Yang, and M. Baum
Proceedings of the 23rd International Conference on Information Fusion (Fusion 2020), Virtual, 2020.
"""
import numpy as np
from scipy.linalg import block_diag

AX_MIN = 0.1


class RHMEKF:
    """
    Single target ellipse tracker based on RHM and polar ellipse equation. Can handle combined orientation for velocity
    and shape.

    Attributes
    ----------
    state               Current state, consisting of center, velocity, orientation, and semi-axes
    cov                 Current state covariance
    R                   Measurement noise covariance
    Q                   Process noise covariance
    time_step_length    Length of a time step used for prediction
    x1                  Index of position first dimension in state
    x2                  Index of position second dimension in state
    v1                  Index of velocity first dimension in state
    v2                  Index of velocity second dimension in state
    v                   Index of polar velocity in state if used
    al                  Index of orientation in state
    l                   Index of semi-axis length in state
    w                   Index of semi-axis width in state
    """

    def __init__(self,
                 m,
                 P,
                 R,
                 Q,
                 time_step_length=1):
        self._state = np.hstack([m.copy(), np.array([0, 1, 1])])
        self._cov = block_diag(P.copy(), np.eye(3))

        self._R = R.copy()
        self._Q = block_diag(Q.copy(), np.eye(3) * 0.01)

        self._time_step_length = time_step_length

        self._x1 = 0
        self._x2 = 1
        self._v1 = 2
        self._v2 = 3
        self._al = 4
        self._l = 5
        self._w = 6

    def get_state(self):
        return np.array(self._state)

    def predict(self):
        """
        Predict kinematic state according to NCV model and add noise to shape covariance.
        :param td:  Time difference
        """
        proc_mat = np.array([
            [1.0, 0.0, self._time_step_length, 0.0],
            [0.0, 1.0, 0.0, self._time_step_length],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
        self._state[:4] = np.dot(proc_mat, self._state[:4])
        self._cov[:4, :4] = np.dot(np.dot(proc_mat, self._cov[:4, :4]), proc_mat.T)
        self._cov += self._Q

        self._cov += self._cov.T
        self._cov *= 0.5

        return self.get_state()

    def meas_equation(self, y):
        """
        Measurement equation for ellipse RHM.
        :param y:   Current measurement
        :return:    The global assumed measurement source, the measurement angle to the center, and the result of the
                    radial function
        """
        # vector from target center to measurement
        yhat_vec = y - self._state[[self._x1, self._x2]]

        # angle of expected measurement source (greedy)
        ang = (np.arctan2(yhat_vec[self._x2], yhat_vec[self._x1]) + np.pi) % (2 * np.pi) - np.pi
        ell_ang = ang - self._state[self._al]  # angle in local coordinates

        # radial function
        l = (self._state[self._l] * self._state[self._w]) / np.sqrt(((self._state[self._w] * np.cos(ell_ang)) ** 2) +
                                                                    ((self._state[self._l] * np.sin(ell_ang)) ** 2))
        # vector from center to expected surface point, given scaling factor s and global angle ang
        yhat = l * np.array([np.cos(ang), np.sin(ang)])

        return yhat + self._state[[self._x1, self._x2]], ang, l

    def update(self, Z):
        """
        Switch between different correction modes and prepare estimate depending on for of state vector.
        :param Z:        The measurement batch
        """
        self.correct(Z)

        return self.get_state()

    def correct(self, meas):
        """
        Kalman filter correction step using the implicit measurement equation. If velocity orientation is separate from
        ellipse shape orientation, its derivatives in the Jacobian would be 0
        :param meas:        The measurement batch, processed sequentially
        """
        nz = len(meas)

        # go through measurements
        for i in range(0, nz):
            yhat, ang, l = self.meas_equation(meas[i])

            # calculate Jacobians
            mu_h = l ** 2 - np.linalg.norm(meas[i] - self._state[[self._x1, self._x2]]) ** 2 + np.trace(self._R)

            jac_1 = np.zeros(len(self._state))
            dh_dalph = (self._state[self._l] ** 2 * self._state[self._w] ** 4
                        - self._state[self._l] ** 4 * self._state[self._w] ** 2) \
                       * np.sin(2 * (ang - self._state[self._al])) \
                       / ((self._state[self._w] * np.cos(ang - self._state[self._al])) ** 2
                          + (self._state[self._l] * np.sin(ang - self._state[self._al])) ** 2) ** 2
            dalph_dxc = np.array([meas[i, self._x2] - self._state[self._x2],
                                  -(meas[i, self._x1] - self._state[self._x1])]) \
                        / ((meas[i, self._x1] - self._state[self._x1]) ** 2
                           + (meas[i, self._x2] - self._state[self._x2]) ** 2)
            jac_1[[self._x1, self._x2]] = dh_dalph * dalph_dxc
            jac_1[self._al] = -dh_dalph
            jac_1[self._l] = (2 * self._state[self._l] * self._state[self._w] ** 2
                              / ((self._state[self._w] * np.cos(ang - self._state[self._al])) ** 2
                                 + (self._state[self._l] * np.sin(ang - self._state[self._al])) ** 2)
                              - 2 * self._state[self._l] ** 3 * self._state[self._w] ** 2
                              * np.sin(ang - self._state[self._al]) ** 2
                              / ((self._state[self._w] * np.cos(ang - self._state[self._al])) ** 2
                                 + (self._state[self._l] * np.sin(ang - self._state[self._al])) ** 2) ** 2)
            jac_1[self._w] = (2 * self._state[self._w] * self._state[self._l] ** 2
                              / ((self._state[self._w] * np.cos(ang - self._state[self._al])) ** 2
                                 + (self._state[self._l] * np.sin(ang - self._state[self._al])) ** 2)
                              - 2 * self._state[self._w] ** 3 * self._state[self._l] ** 2
                              * np.cos(ang - self._state[self._al]) ** 2
                              / ((self._state[self._w] * np.cos(ang - self._state[self._al])) ** 2
                                 + (self._state[self._l] * np.sin(ang - self._state[self._al])) ** 2) ** 2)

            jac_2 = np.array([-2.0 * (meas[i, self._x1] - self._state[self._x1]),
                              -2.0 * (meas[i, self._x2] - self._state[self._x2]),
                              0.0, 0.0, 0.0, 0.0, 0.0])[:len(self._state)]

            t = 2.0 * (yhat - self._state[[self._x1, self._x2]])
            r_h = np.trace(np.dot(t, t) * self._R) + 2.0 * np.trace(np.dot(self._R, self._R)) \
                  + np.trace(self._R) ** 2

            # Kalman update
            cov_xh = np.dot(self._cov, (jac_1 - jac_2).T)
            cov_h = np.dot(np.dot(jac_1 - jac_2, self._cov), jac_1 - jac_2) + r_h
            self._state = self._state + cov_xh * -mu_h / cov_h
            self._cov = self._cov - np.einsum('a, b -> ab', cov_xh / cov_h, cov_xh)
            self._cov = (self._cov + self._cov.T) * 0.5

            self._state[self._al] = (self._state[self._al] + np.pi) % (2 * np.pi) - np.pi
            # lower threshold for shape parameters
            self._state[self._l] = np.max([AX_MIN, self._state[self._l]])
            self._state[self._w] = np.max([AX_MIN, self._state[self._w]])

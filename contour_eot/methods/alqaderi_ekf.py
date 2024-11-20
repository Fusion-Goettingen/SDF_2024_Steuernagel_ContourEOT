"""
H. Alqaderi, F. Govaers, und R. Schulz,
„Spacial Elliptical Model for Extended Target Tracking Using Laser Measurements“,
in 2019 Sensor Data Fusion: Trends, Solutions, Applications (SDF), Okt. 2019, p. 1–6. doi: 10.1109/SDF.2019.8916634.

"""
import numpy as np
from numpy import cos, sin
from numpy.linalg import inv
from scipy.linalg import block_diag
from scipy.stats import multivariate_normal as mvn

from utility.utils import rot

IX_LOC_X = 0
IX_LOC_Y = 1
IX_VEL_X = 2
IX_VEL_Y = 3
IX_L1 = 4
IX_L2 = 5


class AlqaderiEKF:

    def __init__(self,
                 m_init,
                 P_init,
                 R,
                 Q,
                 F=None,  # None -> CV model
                 n_gmm_components=64,
                 mode="permutation"):
        self.P = block_diag(P_init, np.eye(2) * 5)
        self.R = R
        self.F = F if F is not None else np.array([
            [1, 0, 1, 0, 0, 0],
            [0, 1, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
        ])
        self.Q = block_diag(Q, np.eye(2) * 1e-3)
        self.x = np.array([*m_init, 1, 1])
        self._n_gmm_components = n_gmm_components
        self.mode = mode

    def update(self, Z: np.ndarray):

        R = self.R

        if self.mode == "sequential":
            for z in Z:
                # run an individual measurement update
                self.x, self.P = self.single_meas_update(x_minus=self.x,
                                                         P_minus=self.P,
                                                         z=z,
                                                         R=R
                                                         )
        elif self.mode == "permutation":
            for z in Z[np.random.permutation(range(len(Z))), :]:
                # run an individual measurement update
                self.x, self.P = self.single_meas_update(x_minus=self.x,
                                                         P_minus=self.P,
                                                         z=z,
                                                         R=R
                                                         )
        else:
            raise NotImplementedError(f"{self.__class__.__name__} does not implement mode '{self.mode}'!")
        return self.get_state()

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def get_theta(self, state):
        return np.pi / 8
        # return np.arctan2(state[IX_VEL_Y], state[IX_VEL_X])

    def get_state(self):
        state = np.zeros((7,))
        state[:4] = self.x[:4]
        state[4] = self.get_theta(self.x)
        state[5] = self.x[IX_L1]
        state[6] = self.x[IX_L2]
        return state

    def set_R(self, R):
        self.R = R

    def _g(self, x, theta):
        """Implements (5)"""
        alpha = self.get_theta(x)
        C = np.array(x[[IX_LOC_X, IX_LOC_Y]])
        y = rot(alpha) @ np.diag(x[[IX_L1, IX_L2]]) @ np.array([cos(theta), sin(theta)]).reshape((2, 1))
        y = C + y.reshape((2,))
        return y

    def single_meas_update(self, x_minus, P_minus, z, R):
        """
        Run the update for a given point measurement z

        :param x_minus: Prior state
        :param P_minus: Prior covariance
        :param z: Measurement
        :param R: Measurement noise covariance
        :return: (x, P) the updated state and covariance
        """
        # prepare variables
        n = self._n_gmm_components
        theta = np.linspace(0, 2 * np.pi, num=n)
        v, W, S, x, P, weights = [], [], [], [], [], []

        # match heading to velocity
        alpha = self.get_theta(x_minus)
        for j in range(n):
            # compute Jacobian (13)
            J = np.array([
                [1, 0, 0, 0, cos(alpha) * cos(theta[j]), -sin(alpha) * sin(theta[j])],
                [0, 1, 0, 0, sin(alpha) * cos(theta[j]), cos(alpha) * sin(theta[j])],
            ])

            # prepare update helper variables (following (20))
            g = self._g(x_minus, theta[j])
            v_j = z - g
            S_j = J @ P_minus @ J.T + R
            try:
                W_j = P_minus @ J.T @ inv(S_j)
            except np.linalg.LinAlgError:
                # numerical instability, can't inv(S_j)
                continue

            # update component (19) and (20)
            x_j = x_minus + W_j @ v_j
            P_j = P_minus - W_j @ S_j @ W_j.T

            # Ensure semi-axis are non-negative  TODO not discussed in paper
            eps = 0.05
            x_j[-2:][x_j[-2:] < eps] = eps

            # save
            v.append(v_j)
            W.append(W_j)
            S.append(S_j)
            x.append(x_j)
            P.append(P_j)

            # compute and save weight
            try:
                weights.append(mvn.pdf(z, mean=g, cov=S_j))
            except ValueError:
                # use small identity instead, numerical instability caused problems
                weights.append(mvn.pdf(z, mean=g, cov=np.eye(2) * 1e-3))
            except np.linalg.LinAlgError:
                # use small identity instead, numerical instability caused problems
                weights.append(mvn.pdf(z, mean=g, cov=np.eye(2) * 1e-3))

        # weights need to be normalized
        weights = np.array(weights)
        weights /= weights.sum()

        # weights sometimes become all zero? - potential indicator of filter divergence
        # will be nan after division with zero in above line - catch this here and set to uniform again
        if np.isnan(weights).any():
            weights = np.full(shape=weights.shape,
                              fill_value=1 / len(weights))
        # moment matching of Gaussian Mixture to single Gaussian (17) (18)
        x_plus = np.average(x, axis=0, weights=weights)
        P_plus = np.sum([
            weights[j] * (P[j] + (x[j] - x_plus).reshape((-1, 1)) @ (x[j] - x_plus).reshape((-1, 1)).T)
            for j in range(len(weights))
        ], axis=0)
        return x_plus, P_plus

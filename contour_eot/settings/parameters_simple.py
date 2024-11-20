import numpy as np
from contour_eot.utility.utils import rot

params = {
    'IX_5D': [0, 1, 4, 5, 6],  # remove the velocity from a 7D state, reducing it to just position and shape information

    'init_state': np.array([0, 0, 30, 0, np.pi / 8, 10, 4]),
    'n_steps_in_scenario': 10,
    'qualitative_scenario_step_ixs': [5, 6, 7, 8, 9],
    'R': rot(np.pi / 4) @ np.diag([3, 1]) @ rot(np.pi / 4).T,
    'P_init': np.diag([2, 2, 2, 2]),
    'v_init': 10,
    'Q_kinematic': np.diag([0.1, 0.1, 1, 0.1]),
    'n_meas_per_step': 10,
    'metric_range': [0, 7.15]
}

TRACKER_RM_ADAPTIVE_SCALING_FROM_PRIOR = -1
TRACKER_RM_NUMERICAL_FACTORS_FOR_10x4_OBJECT = -104

TRACKER_RHMEKF = -2
TRACKER_ALQADERI_EKF = -3


def get_from_id(num_id, detailed=False) -> str:
    if num_id == TRACKER_RM_ADAPTIVE_SCALING_FROM_PRIOR:
        return "RM (adaptive)"
    elif num_id == TRACKER_RM_NUMERICAL_FACTORS_FOR_10x4_OBJECT:
        return "RM (2.5:1)"
    elif num_id == TRACKER_RHMEKF:
        return "RHM-EKF"
    elif num_id == TRACKER_ALQADERI_EKF:
        if detailed:
            return "SpacialEKF\n(known orientation)"
        else:
            return "SpacialEKF"
    else:
        return f"RM (factor={num_id})"

import numpy as np
import matplotlib.pyplot as plt

from contour_eot.utility.contour_sampling import state_to_ellipse_contour_pts


def plot_elliptic_state(s,
                        fill=False,
                        **kwargs
                        ):
    """
    Plot a single state of an elliptical object, based on a polygon approximation
    :param s: 5D state
    :param fill: boolean indicating whether to fill the rectangle (True) or just draw the outline (False)
    :param kwargs: additional keyword args passed on to plt.plot/plt.fill
    """
    if fill:
        plt.fill(*state_to_ellipse_contour_pts(s).T,
                 **kwargs)
    else:
        plt.plot(*state_to_ellipse_contour_pts(s).T,
                 **kwargs)


def prepare_styling():
    """
    Utility function that should be called once before any experiments are carried out, setting up visualization and
    print settings.
    """
    # put the path to your dir here
    style_dir = "../../data/"

    # select which style should be used
    plt.style.use(f"{style_dir}paper.mplstyle")

    # set up printing
    np.set_printoptions(suppress=True, linewidth=100000)

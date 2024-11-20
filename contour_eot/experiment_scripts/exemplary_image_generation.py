import numpy as np
import matplotlib.pyplot as plt

from contour_eot.utility.visuals import plot_elliptic_state, prepare_styling
from contour_eot.utility.contour_sampling import get_deterministic_ellipse_contour_measurements
from contour_eot.utility.utils import scatter_matrix, matrix_to_params


def visualize_different_sampling_methods(target_file):
    def save(f):
        plt.axis('equal')
        plt.axis('off')
        plt.tight_layout()
        if f is None:
            plt.show()
        else:
            plt.savefig(f, bbox_inches='tight')
            plt.close()
            print(f"Saved to file f{f}")

    kwargs = dict(
        marker='o',
        c='black'
    )
    n_pts = 50
    plt.rcParams["figure.figsize"] = 8, 3

    l1, l2 = 10, 2

    s1 = np.array([0, 0, 0, l1, l2])
    plot_elliptic_state(s1, fill=True, c='grey', alpha=.3)
    z1 = get_deterministic_ellipse_contour_measurements(state=s1, number_of_points=n_pts, equidistant_on_contour=False)
    plt.scatter(*z1.T, **kwargs)
    plot_elliptic_state(np.asarray([*np.mean(z1, axis=0), *matrix_to_params(scatter_matrix(z1))]), c='C0')

    save(target_file + "_angles")

    # EQUIDISTANT
    s2 = np.array([0, 0, 0, l1, l2])
    plot_elliptic_state(s2, fill=True, c='grey', alpha=.3)
    z2 = get_deterministic_ellipse_contour_measurements(state=s2, number_of_points=n_pts, equidistant_on_contour=True)
    plt.scatter(*z2.T, **kwargs)
    plot_elliptic_state(np.asarray([*np.mean(z2, axis=0), *matrix_to_params(scatter_matrix(z2))]), c='C0')

    save(target_file + "_equidistant")


if __name__ == '__main__':
    prepare_styling()
    target_dir = "../../output/"
    visualize_different_sampling_methods(target_file=f"{target_dir}sampling_examples")

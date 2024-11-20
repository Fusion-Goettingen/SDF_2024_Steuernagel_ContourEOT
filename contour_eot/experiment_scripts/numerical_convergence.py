import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import glob
import json

from contour_eot.utility.utils import get_numerical_mm_factors_rotationfree
from contour_eot.utility.contour_sampling import get_ellipse_contour_measurements
from contour_eot.utility.visuals import prepare_styling



class NpEncoder(json.JSONEncoder):
    """https://stackoverflow.com/a/57915246"""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


def generate_data(l1: float,
                  l2: float,
                  n_meas_array: np.ndarray,
                  equidistant_on_contour: bool,
                  rng_seed: int,
                  R=None,
                  n_mc_runs: int = 100,
                  suptitle=None,
                  target_file=None):
    rng = np.random.default_rng(seed=rng_seed)
    n_meas_array = np.array(n_meas_array).astype(int)
    state = np.array([0, 0, 0, l1, l2])

    means_f1 = []
    stds_f1 = []
    means_f2 = []
    stds_f2 = []

    pbar = tqdm(total=len(n_meas_array) * n_mc_runs)
    for n in n_meas_array:
        temp_vals_f1 = []
        temp_vals_f2 = []

        for i in range(n_mc_runs):
            meas = get_ellipse_contour_measurements(state,
                                                    number_of_points=n,
                                                    equidistant_on_contour=equidistant_on_contour,
                                                    rng=rng,
                                                    R=R)

            # importantly:
            #   if measurement noise is present, the resulting distribution approximates X+R
            #   with shape matrix X and measurement noise covariance R
            #   we are interested in scaling factor for X, not X+R, and hence subtract the measurement noise R
            #   before computing the scaling factors!
            fs = get_numerical_mm_factors_rotationfree(s=state,
                                                       Z=meas,
                                                       sub=R)
            temp_vals_f1.append(fs[0])
            temp_vals_f2.append(fs[1])
            pbar.update(1)
        means_f1.append(np.mean(temp_vals_f1))
        means_f2.append(np.mean(temp_vals_f2))
        stds_f1.append(np.std(temp_vals_f1))
        stds_f2.append(np.std(temp_vals_f2))
    pbar.close()
    print("")

    means_f1 = np.array(means_f1)
    means_f2 = np.array(means_f2)
    stds_f1 = np.array(stds_f1)
    stds_f2 = np.array(stds_f2)

    data = {
        "l1": l1,
        "l2": l2,
        "equidistant_on_contour": equidistant_on_contour,
        "rng_seed": rng_seed,
        "n_mc_runs": n_mc_runs,
        "suptitle": suptitle,
        "R": R,
        "n_meas_array": n_meas_array,
        "means_f1": means_f1,
        "means_f2": means_f2,
        "stds_f1": stds_f1,
        "stds_f2": stds_f2,
    }
    with open(target_file + ".json", "w") as f:
        json.dump(data, f, cls=NpEncoder, indent=4)


def visualize_results(l1: float,
                      l2: float,
                      n_meas_array,
                      equidistant_on_contour: bool,
                      rng_seed: int,
                      R,
                      n_mc_runs,
                      suptitle,
                      target_file,
                      means_f1,
                      means_f2,
                      stds_f1,
                      stds_f2,
                      max_n_meas=None):
    means_f1 = np.array(means_f1)
    means_f2 = np.array(means_f2)
    stds_f1 = np.array(stds_f1)
    stds_f2 = np.array(stds_f2)
    n_meas_array = np.array(n_meas_array)
    if max_n_meas is not None:
        idxs = n_meas_array < max_n_meas
        n_meas_array = n_meas_array[idxs]
        means_f1 = means_f1[idxs]
        means_f2 = means_f2[idxs]
        stds_f1 = stds_f1[idxs]
        stds_f2 = stds_f2[idxs]

    print(f"Converged to f1={means_f1[-1]:.5f} and f2={means_f2[-1]:.5f}")

    fig, axs = plt.subplots(1, 2, sharey=True)
    plt.sca(axs[0])
    plt.plot(n_meas_array, means_f1)
    kwargs_fill = dict(
        alpha=.3
    )
    plt.fill_between(n_meas_array, means_f1 - stds_f1, means_f1 + stds_f1, **kwargs_fill)
    plt.title("First semi-axis", fontdict={'fontsize': plt.rcParams["font.size"]})

    plt.ylabel("Scaling factor")
    xlabel = "Number of Measurements"  # "N"
    plt.xlabel(xlabel)

    plt.sca(axs[1])
    plt.plot(n_meas_array, means_f2)
    plt.fill_between(n_meas_array, means_f2 - stds_f2, means_f2 + stds_f2, **kwargs_fill)
    plt.title("Second semi-axis", fontdict={'fontsize': plt.rcParams["font.size"]})
    # plt.ylabel("Estimated scaling factor")
    plt.xlabel(xlabel)

    if suptitle is not None:
        plt.suptitle(suptitle)
    plt.tight_layout()

    if target_file is None:
        plt.show()
    else:
        plt.savefig(target_file, bbox_inches='tight')
        plt.close()
        print(f"Saved to file f{target_file}")


def create_data(target_dir,
                with_title=True,
                n_runs=100):
    size_circle = 10, 10
    size_ellipse = 10, 4
    n_meas_array = np.arange(10, 1000 + 10, 10)
    R_where_applicable = np.eye(2)

    print("Generating data 0")
    suptitle = "10x10 object with measurements uniform in angle and no meas. noise"
    generate_data(
        l1=size_circle[0],
        l2=size_circle[1],
        n_meas_array=n_meas_array,
        equidistant_on_contour=False,
        R=None,
        rng_seed=1,
        suptitle=suptitle if with_title else None,
        target_file=target_dir + f"numerical_var_0",
        n_mc_runs=n_runs
    )

    print("Generating data 1")
    suptitle = "10x10 object with measurements uniform in angle and $\mathbf{R}=\mathbf{I}_{2\\times2}$"
    generate_data(
        l1=size_circle[0],
        l2=size_circle[1],
        n_meas_array=n_meas_array,
        equidistant_on_contour=False,
        R=R_where_applicable,
        rng_seed=2,
        suptitle=suptitle if with_title else None,
        target_file=target_dir + f"numerical_var_1",
        n_mc_runs=n_runs
    )

    print("Generating data 2")
    suptitle = "10x4 object with measurements uniform in angle and $\mathbf{R}=\mathbf{I}_{2\\times2}$"
    generate_data(
        l1=size_ellipse[0],
        l2=size_ellipse[1],
        n_meas_array=n_meas_array,
        equidistant_on_contour=False,
        R=R_where_applicable,
        rng_seed=3,
        suptitle=suptitle if with_title else None,
        target_file=target_dir + f"numerical_var_2",
        n_mc_runs=n_runs
    )

    print("Generating data 3")
    suptitle = "10x4 object with measurements equidistant on contour and $\mathbf{R}=\mathbf{I}_{2\\times2}$"
    generate_data(
        l1=size_ellipse[0],
        l2=size_ellipse[1],
        n_meas_array=n_meas_array,
        equidistant_on_contour=True,
        R=R_where_applicable,
        rng_seed=4,
        suptitle=suptitle if with_title else None,
        target_file=target_dir + f"numerical_var_3",
        n_mc_runs=n_runs
    )


def create_plots(target_dir, max_n_meas=None):
    files = glob.glob(f"{target_dir}numerical*.json")
    print(f"\n\n\nFound {files}\n")
    plt.rcParams["figure.figsize"] = 12, 5
    plt.rcParams["font.size"] = 22

    for f in files:
        with open(f) as fp:
            data = json.load(fp)
        visualize_results(target_file=f.replace(".json", ""), max_n_meas=max_n_meas, **data)


if __name__ == '__main__':
    target_dir = "../../output/"
    prepare_styling()

    n_monte_carlo_runs_per_setting = 500
    use_title = False if plt.rcParams["savefig.format"] == 'pdf' else True  # no title for paper pdf plots
    create_data(n_runs=n_monte_carlo_runs_per_setting, with_title=use_title, target_dir=target_dir)

    create_plots(target_dir, max_n_meas=500)

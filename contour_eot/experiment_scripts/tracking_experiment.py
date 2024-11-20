import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cycler

from contour_eot.settings.method_names import *
from contour_eot.settings.parameters_simple import params
from contour_eot.utility.tracking import perform_single_run, errors_over_monte_carlo_runs, runtime_print
from contour_eot.utility.visuals import prepare_styling

if __name__ == '__main__':
    prepare_styling()

    # optionally, we flip the first two colors of the default color cycler for the visualizations
    # this is just to increase contrast/visibility for the qualitative plots, no functional change!
    default_cycler = plt.rcParams["axes.prop_cycle"]
    flipped_cycler = cycler('color', ['#2ca02c', '#ff7f0e', '#1f77b4',
                                      '#d62728', '#9467bd', '#8c564b',
                                      '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'])

    target_dir = "../../output/"
    perform_qualitative = True
    perform_quantitative = True
    compare_quantitative = True
    evaluate_runtime = True
    n_runs = 500

    if perform_qualitative:
        plt.rcParams["axes.prop_cycle"] = flipped_cycler
        # Uniform in angle (1/2)
        perform_single_run([0.25, 0.5],
                           equidistant_on_contour=False,
                           target_file=f"{target_dir}angles_qualitative",
                           rng=np.random.default_rng(1),
                           disable_legend=True,
                           **params)

        # Equidistant on contour with estimating scaling factors in an online fashion
        perform_single_run([0.25, 0.5, TRACKER_RM_ADAPTIVE_SCALING_FROM_PRIOR],
                           equidistant_on_contour=True,
                           target_file=f"{target_dir}equidistant_qualitative",
                           linestyles=[None, (0, (2, 2)), None],  # plot the 0.5 ellipse as (customized) dashed
                           zorders=[1, 3, 1],  # plot the 0.5 ellipse on top of everything else
                           rng=np.random.default_rng(10),
                           disable_legend=True,
                           **params)

    if perform_quantitative:
        plt.rcParams["axes.prop_cycle"] = flipped_cycler
        # Uniform in angle (1/2)
        errors_over_monte_carlo_runs(n_runs=n_runs,
                                     factors=[0.25, 0.5],
                                     equidistant_on_contour=False,
                                     squared_gwd=False,
                                     target_file=f"{target_dir}angles_quantitative",
                                     disable_legend=False,
                                     **params)

        # Equidistant on contour with estimating scaling factors in an online fashion
        errors_over_monte_carlo_runs(n_runs=n_runs,
                                     factors=[0.25, 0.5, TRACKER_RM_ADAPTIVE_SCALING_FROM_PRIOR],
                                     equidistant_on_contour=True,
                                     squared_gwd=False,
                                     target_file=f"{target_dir}equidistant_quantitative",
                                     disable_legend=False,
                                     **params)

        # Equidistant on contour with scaling factors fixed for an ellipse of size 10x4
        errors_over_monte_carlo_runs(n_runs=n_runs,
                                     factors=[0.25, 0.5, TRACKER_RM_ADAPTIVE_SCALING_FROM_PRIOR,
                                              TRACKER_RM_NUMERICAL_FACTORS_FOR_10x4_OBJECT],
                                     equidistant_on_contour=True,
                                     squared_gwd=False,
                                     target_file=f"{target_dir}knownFactor_equidistant_quantitative",
                                     disable_legend=False,
                                     **params)

    if compare_quantitative:
        plt.rcParams["axes.prop_cycle"] = default_cycler
        # Uniform in angle (1/2)
        errors_over_monte_carlo_runs(n_runs=n_runs,
                                     factors=[0.5,
                                              TRACKER_RHMEKF,
                                              TRACKER_ALQADERI_EKF],
                                     equidistant_on_contour=False,
                                     squared_gwd=False,
                                     target_file=f"{target_dir}angles_comparison",
                                     disable_legend=False,
                                     **params)

        # Equidistant on contour with estimating scaling factors in an online fashion
        errors_over_monte_carlo_runs(n_runs=n_runs,
                                     factors=[TRACKER_RM_ADAPTIVE_SCALING_FROM_PRIOR,
                                              TRACKER_RHMEKF,
                                              TRACKER_ALQADERI_EKF
                                              ],
                                     equidistant_on_contour=True,
                                     squared_gwd=False,
                                     target_file=f"{target_dir}equidistant_comparison",
                                     disable_legend=False,
                                     **params)

    if evaluate_runtime:
        runtime_print(
            multiply_length_with=50,
            factors=[
                0.5,
                TRACKER_RM_ADAPTIVE_SCALING_FROM_PRIOR,
                TRACKER_RHMEKF,
                TRACKER_ALQADERI_EKF],
            equidistant_on_contour=True,
            **params)

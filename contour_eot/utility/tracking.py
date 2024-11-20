import time

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from contour_eot.settings.method_names import *
from contour_eot.methods.random_matrix import TrackerRM
from contour_eot.methods.random_matrix_equidistant_contour import TrackerRMEC
from contour_eot.methods.rhm_ekf import RHMEKF
from contour_eot.methods.alqaderi_ekf import AlqaderiEKF
from contour_eot.utility.contour_sampling import get_ellipse_contour_measurements
from contour_eot.utility.metrics import gwd, iou
from contour_eot.utility.visuals import plot_elliptic_state


def getTracker(m_init,
               P_init,
               v_init,
               R,
               Q_kinematic,
               f
               ):
    """
    Create random matrix tracker with given parameters.
    Parameter f is either the scaling factor or negative constant to select special method.
    """
    if f == TRACKER_RM_ADAPTIVE_SCALING_FROM_PRIOR:
        return TrackerRMEC(m=m_init,
                           P=P_init,
                           v=v_init,
                           R=R,
                           Q=Q_kinematic)
    elif f == TRACKER_RM_NUMERICAL_FACTORS_FOR_10x4_OBJECT:
        return TrackerRMEC(m=m_init,
                           P=P_init,
                           v=v_init,
                           R=R,
                           Q=Q_kinematic,
                           scaling_factors=[0.4, 0.6])
    elif f == TRACKER_RHMEKF:
        return RHMEKF(m=m_init,
                      P=P_init,
                      R=R,
                      Q=Q_kinematic)
    elif f == TRACKER_ALQADERI_EKF:
        return AlqaderiEKF(m_init=m_init,
                           P_init=P_init,
                           R=R,
                           F=None,
                           Q=Q_kinematic,
                           n_gmm_components=64,
                           mode="permutation")
    else:  # use the given scalar scaling factor for a standard RM tracker
        return TrackerRM(m=m_init,
                         P=P_init,
                         v=v_init,
                         R=R,
                         Q=Q_kinematic,
                         scaling_factor=f)


def perform_tracking(scaling_factor_list,
                     equidistant_on_contour,
                     n_meas_per_step,
                     init_state,
                     n_steps_in_scenario,
                     R,
                     P_init,
                     v_init,
                     Q_kinematic,
                     IX_5D,
                     rng=None
                     ):
    rng = np.random.default_rng(1) if rng is None else rng
    gt_state = np.array(init_state)
    F = np.eye(7)
    F[0, 2] = 1
    F[1, 3] = 1

    n_steps = n_steps_in_scenario

    m_init = rng.multivariate_normal(gt_state[:4], P_init)

    trackers = {f: getTracker(m_init,
                              P_init,
                              v_init,
                              R,
                              Q_kinematic,
                              f)
                for f in scaling_factor_list
                }
    measurements = []
    gt_list = []
    est_list = {f: [] for f in scaling_factor_list}
    gwds = {f: [] for f in scaling_factor_list}
    ious = {f: [] for f in scaling_factor_list}
    runtimes = {f: [] for f in scaling_factor_list}
    for step_ix in range(n_steps):
        gt_list.append(gt_state)
        next_meas = get_ellipse_contour_measurements(gt_state[IX_5D],
                                                     number_of_points=n_meas_per_step,
                                                     equidistant_on_contour=equidistant_on_contour,
                                                     R=R,
                                                     rng=rng)
        measurements.append(next_meas)
        for f in scaling_factor_list:
            t0 = time.time()
            trackers[f].update(Z=next_meas)
            update_time = time.time() - t0
            runtimes[f].append(update_time)
            est_list[f].append(trackers[f].get_state())
            gwds[f].append(gwd(gt_state[IX_5D], trackers[f].get_state()[IX_5D]))
            ious[f].append(iou(gt_state[IX_5D], trackers[f].get_state()[IX_5D]))
            trackers[f].predict()

        proc_noise = [*rng.multivariate_normal(np.zeros(4), Q_kinematic), 0, 0, 0]
        gt_state = F @ gt_state + proc_noise

    results_dict = dict(
        metadata=dict(
            scaling_factor_list=scaling_factor_list,
            equidistant_on_contour=equidistant_on_contour,
            n_meas_per_step=n_meas_per_step,
            n_steps=n_steps,
            R=R,
            P_init=P_init,
            v_init=v_init,
            _init_state=init_state
        ),
        measurements=np.array(measurements),
        gt_list=np.array(gt_list),
        est_list={f: np.array(est_list[f]) for f in scaling_factor_list},
        gwd_list={f: np.array(gwds[f]) for f in scaling_factor_list},
        iou_list={f: np.array(ious[f]) for f in scaling_factor_list},
        runtime_list={f: np.array(runtimes[f]) for f in scaling_factor_list}
    )
    return results_dict


def perform_single_run(factors,
                       equidistant_on_contour,
                       IX_5D,
                       init_state,
                       n_steps_in_scenario,
                       qualitative_scenario_step_ixs,
                       R,
                       P_init,
                       v_init,
                       Q_kinematic,
                       n_meas_per_step,
                       disable_legend=False,
                       target_file=None,
                       linestyles=None,
                       zorders=None,
                       rng=None,
                       **kwargs):
    data = perform_tracking(scaling_factor_list=factors,
                            equidistant_on_contour=equidistant_on_contour,
                            n_meas_per_step=n_meas_per_step,
                            init_state=init_state,
                            n_steps_in_scenario=n_steps_in_scenario,
                            R=R,
                            P_init=P_init,
                            v_init=v_init,
                            Q_kinematic=Q_kinematic,
                            IX_5D=IX_5D,
                            rng=rng)

    for f in factors:
        print(f"Method {get_from_id(f)}:")
        print("\t", data["gwd_list"][f].round(2))

    plt.rcParams["font.size"] = 21
    fig, axs = plt.subplots(1, 1, figsize=(22, 5))
    linewidth = 6

    for step_ix in np.arange(data["metadata"]["n_steps"])[qualitative_scenario_step_ixs]:
        label_flag = step_ix == np.arange(data["metadata"]["n_steps"])[qualitative_scenario_step_ixs][0]
        gt = data["gt_list"][step_ix]
        Z = data["measurements"][step_ix]
        plot_elliptic_state(gt[IX_5D], fill=True, c='grey',
                            alpha=0.5,
                            label='Ground Truth' if label_flag else None,
                            linewidth=linewidth)
        plt.scatter(*Z.T, c='k', label='Measurements' if label_flag else None, zorder=4, marker='o', s=70)

        for i, f in enumerate(factors):
            est = data["est_list"][f][step_ix]
            plot_elliptic_state(est[IX_5D],
                                fill=False,
                                c=f'C{i}',
                                label=f"{get_from_id(f)}" if label_flag else None,
                                linewidth=linewidth,
                                linestyle=linestyles[i] if linestyles is not None else None,
                                zorder=zorders[i] if zorders is not None else None)
    plt.xlabel("$m_1~/~m$")
    plt.ylabel("$m_2~/~m$")
    plt.axis('equal')
    if not disable_legend:
        plt.legend()
    plt.tight_layout()
    if target_file is not None:
        plt.savefig(target_file, bbox_inches='tight')
        plt.close()
        print(f"Saved image to {target_file}")
    else:
        plt.show()


def runtime_print(
        multiply_length_with,
        factors,
        equidistant_on_contour,
        IX_5D, init_state, n_steps_in_scenario, R, P_init, v_init, Q_kinematic,
        n_meas_per_step,
        **kwargs):
    gwd_stack_dict = {f: [] for f in factors}
    iou_stack_dict = {f: [] for f in factors}
    rng = np.random.default_rng(42)
    data = perform_tracking(scaling_factor_list=factors,
                            equidistant_on_contour=equidistant_on_contour,
                            n_meas_per_step=n_meas_per_step,
                            init_state=init_state,
                            n_steps_in_scenario=n_steps_in_scenario * multiply_length_with,
                            R=R,
                            P_init=P_init,
                            v_init=v_init,
                            Q_kinematic=Q_kinematic,
                            IX_5D=IX_5D,
                            rng=rng)
    print(f"Average runtime per method in ms over {n_steps_in_scenario * multiply_length_with} steps:")
    for f in factors:
        gwd_stack_dict[f].append(data["gwd_list"][f])
        iou_stack_dict[f].append(data["iou_list"][f])
        print(f"\t{get_from_id(f)}: {np.mean(data['runtime_list'][f]) * 1000:.3f}ms")


def errors_over_monte_carlo_runs(n_runs,
                                 factors,
                                 equidistant_on_contour,
                                 IX_5D, init_state, n_steps_in_scenario, R, P_init, v_init, Q_kinematic,
                                 n_meas_per_step,
                                 disable_legend=False,
                                 squared_gwd=True,
                                 target_file=None,
                                 metric='gwd',
                                 metric_range=None,
                                 **kwargs):
    gwd_stack_dict = {f: [] for f in factors}
    iou_stack_dict = {f: [] for f in factors}
    rng = np.random.default_rng(42)
    for i in tqdm(range(n_runs)):
        data = perform_tracking(scaling_factor_list=factors,
                                equidistant_on_contour=equidistant_on_contour,
                                n_meas_per_step=n_meas_per_step,
                                init_state=init_state,
                                n_steps_in_scenario=n_steps_in_scenario,
                                R=R,
                                P_init=P_init,
                                v_init=v_init,
                                Q_kinematic=Q_kinematic,
                                IX_5D=IX_5D,
                                rng=rng)
        for f in factors:
            gwd_stack_dict[f].append(data["gwd_list"][f])
            iou_stack_dict[f].append(data["iou_list"][f])

    if metric == 'gwd':
        plot(factors,
             error_stack_dict=gwd_stack_dict,
             ylabel="Squared GWD / $m^2$" if squared_gwd else "GWD / $m$",
             apply_sqrt=not squared_gwd,
             disable_legend=disable_legend,
             target_file=target_file,
             ylims=metric_range)
    elif metric == 'iou':
        plot(factors,
             error_stack_dict=iou_stack_dict,
             ylabel="IoU",
             apply_sqrt=False,
             disable_legend=disable_legend,
             target_file=target_file,
             ylims=metric_range)
    else:
        raise ValueError(f"Unknown metric {metric}")


def plot(factors,
         error_stack_dict,
         ylabel=None,
         apply_sqrt=False,
         disable_legend=False,
         target_file=None,
         ylims=None):
    plt.rcParams["font.size"] = 22
    fig, axs = plt.subplots(1, 1, figsize=(12, 8))
    for ix, f in enumerate(factors):
        error_stack = np.array(error_stack_dict[f])
        xs = np.arange(1, (error_stack.shape[1] + 1))
        if apply_sqrt:
            error_stack = np.sqrt(error_stack)
        error_mean = np.mean(error_stack, axis=0)
        error_std = np.std(error_stack, axis=0)
        plt.plot(xs, error_mean, c=f'C{ix}', label=f"{get_from_id(f, detailed=True)}", linewidth=5)
        plt.fill_between(xs, error_mean - error_std, error_mean + error_std, color=f'C{ix}', alpha=0.2)
        print(f"Method {get_from_id(f)} converged to final error mean {error_mean[-1]:.3f}")

    plt.xlabel("Step")
    plt.ylabel(ylabel)
    if ylims is None:
        plt.ylim(0, plt.ylim()[1])
    else:
        plt.ylim(ylims)
    if not disable_legend:
        plt.legend()
    plt.tight_layout()
    if target_file is None:
        plt.show()
    else:
        plt.savefig(target_file, bbox_inches='tight')
        plt.close()
        print(f"Saved to file {target_file}")

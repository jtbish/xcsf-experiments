import glob
import math
import pickle
import sys
from collections import OrderedDict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import parse as ps

_EXPECTED_NUM_EXP_DIRS = 30
_EXPECTED_HISTORY_LEN_DETRM = 200
_EXPECTED_HISTORY_LEN_STOCA = 400
_MONITOR_FREQ_EPISODES = 250
_EPSILON_NOUGHT = 0.01
_MAX_PERF = 1.0
_BOUND_LINESTYLE = "dashed"
_BOUND_LINEWIDTH = 1.0
_EMP_OPT_PERFS_PKL_FILE = \
    "FrozenLake_emp_opt_perfs_gamma_0.95_iodsb_frozen.pkl"
_STD_ALPHA = 0.2
_VAR_STD_DDOF = 1  # use Bessel's correction
_ROUND_DEC_PLACES = 3

_ERR_COLOR = '#d62728'  # red
_ERR_YMIN = 0.0
_ERR_YMAX = 0.5

_PERF_COLOR = '#1f77b4'  # blue
_PERF_YMIN = 0.0
_PERF_YMAX = 1.2


def main():
#    import matplotlib as mpl
#    print(mpl.rcParams["axes.prop_cycle"])
    base_dir = sys.argv[1]
    base_dir_tail = Path(base_dir).name
    if "detrm" in base_dir:
        gs = int(ps.parse("gs_{}", base_dir_tail)[0])
        sp = 0
    elif "stoca" in base_dir:
        gs = int(ps.parse("gs_{}_sp_{}", base_dir_tail)[0])
        sp = float(ps.parse("gs_{}_sp_{}", base_dir_tail)[1])
    else:
        assert False
    print(f"Grid size {gs}")
    print(f"Slip prob {sp}")

    # get optimal perf
    with open(f"./{_EMP_OPT_PERFS_PKL_FILE}", "rb") as fp:
        empirical_opt_perfs = pickle.load(fp)
    emp_opt_perf = empirical_opt_perfs[(gs, sp)].perf

    exp_dirs = glob.glob(f"{base_dir}/6*")
    assert len(exp_dirs) == _EXPECTED_NUM_EXP_DIRS
    exps_are_stochastic = ("stoca" in base_dir)
    if exps_are_stochastic:
        expected_history_len = _EXPECTED_HISTORY_LEN_STOCA
    else:
        expected_history_len = _EXPECTED_HISTORY_LEN_DETRM

    monitor_episode_ticks = [
        i * _MONITOR_FREQ_EPISODES for i in range(1, expected_history_len + 1)
    ]
    # init aggregate histories
    aggr_q_hat_mae_history = OrderedDict(
        {num_episodes: []
         for num_episodes in monitor_episode_ticks})
    aggr_perf_history = OrderedDict(
        {num_episodes: []
         for num_episodes in monitor_episode_ticks})

    # fill aggregate histories
    for exp_dir in exp_dirs:
        with open(f"{exp_dir}/q_hat_mae_history.pkl", "rb") as fp:
            q_hat_mae_history = pickle.load(fp)
        with open(f"{exp_dir}/perf_history.pkl", "rb") as fp:
            perf_history = pickle.load(fp)
        assert len(q_hat_mae_history) == expected_history_len
        assert len(perf_history) == expected_history_len

        # for histories, keys are num episodes done
        for (k, v) in q_hat_mae_history.items():
            assert k in aggr_q_hat_mae_history
            if v is None:
                print("Null pred")
            aggr_q_hat_mae_history[k].append(v)
        for (k, v) in perf_history.items():
            assert k in aggr_perf_history
            # v is perf assessment response
            perf = v.perf
            if perf > emp_opt_perf:
                print(f"Better than opt. perf. : {perf} > {emp_opt_perf}")
            perf_pcnt = (perf / emp_opt_perf)
            aggr_perf_history[k].append(perf_pcnt)

    # check aggregate histories are valid
    for (k, v) in aggr_q_hat_mae_history.items():
        assert len(v) == _EXPECTED_NUM_EXP_DIRS
    for (k, v) in aggr_perf_history.items():
        assert len(v) == _EXPECTED_NUM_EXP_DIRS

    mean_q_hat_maes = {}
    std_q_hat_maes = {}
    for (k, v) in aggr_q_hat_mae_history.items():
        # don't include mean or std if missing values for given num eps.
        mean = _mean_detect_null(v)
        if mean is not None:
            mean_q_hat_maes[k] = mean
        std = _std_detect_null(v)
        if std is not None:
            std_q_hat_maes[k] = std
    assert len(mean_q_hat_maes) == len(std_q_hat_maes)
    last_key = max(mean_q_hat_maes.keys())
    last_mean_round = _round_half_up(mean_q_hat_maes[last_key],
                                     _ROUND_DEC_PLACES)
    last_std_round = _round_half_up(std_q_hat_maes[last_key],
                                    _ROUND_DEC_PLACES)
    print(
        f"Qhat MAE @ {last_key} eps: {last_mean_round:.3f} +- {last_std_round:.3f}"
    )

    mean_perfs = {k: np.mean(v) for (k, v) in aggr_perf_history.items()}
    std_perfs = {
        k: np.std(v, ddof=_VAR_STD_DDOF)
        for (k, v) in aggr_perf_history.items()
    }
    last_key = max(mean_perfs.keys())
    last_mean_round = _round_half_up(mean_perfs[last_key], _ROUND_DEC_PLACES)
    last_std_round = _round_half_up(std_perfs[last_key], _ROUND_DEC_PLACES)
    print(
        f"Perf @ {last_key} eps: {last_mean_round:.3f} +- {last_std_round:.3f}"
    )

    plt.figure()
    xs = np.array(list(mean_q_hat_maes.keys()))
    ys = np.array(list(mean_q_hat_maes.values()))
    err = np.array(list(std_q_hat_maes.values()))
    plt.plot(xs, ys, color=_ERR_COLOR)
    plt.fill_between(xs,
                     ys - err,
                     ys + err,
                     alpha=_STD_ALPHA,
                     color=_ERR_COLOR)
    plt.axhline(y=_EPSILON_NOUGHT,
                linestyle=_BOUND_LINESTYLE,
                linewidth=_BOUND_LINEWIDTH,
                color=_ERR_COLOR)
    plt.xlim(right=max(xs))
    plt.ylim(_ERR_YMIN, _ERR_YMAX)
    xticks_step = 10000
    xticks = np.arange(0, max(xs)+xticks_step, step=xticks_step)
    plt.xticks(xticks)
    plt.gca().set_xticklabels([str(x//1000) for x in xticks])
    plt.yticks([i*0.05 for i in range(0, 10+1)])
    plt.xlabel("Num. training episodes (thousands)")
    plt.ylabel("$\hat{Q}$ MAE")
    #plt.show()
    plt.savefig(f"{base_dir}/{base_dir_tail}_q_hat_mae_plot.png",
                bbox_inches="tight")
    plt.savefig(f"{base_dir}/{base_dir_tail}_q_hat_mae_plot.pdf",
                bbox_inches="tight")

    plt.figure()
    xs = np.array(list(mean_perfs.keys()))
    ys = np.array(list(mean_perfs.values()))
    err = np.array(list(std_perfs.values()))
    plt.plot(xs, ys, color=_PERF_COLOR)
    plt.fill_between(xs,
                     ys - err,
                     ys + err,
                     alpha=_STD_ALPHA,
                     color=_PERF_COLOR)
    plt.axhline(y=_MAX_PERF,
                linestyle=_BOUND_LINESTYLE,
                linewidth=_BOUND_LINEWIDTH,
                color=_PERF_COLOR)
    plt.xlim(right=max(xs))
    plt.ylim(_PERF_YMIN, _PERF_YMAX)
    plt.xlabel("Num. training episodes")
    plt.ylabel("Testing performance")
    #plt.show()
    #plt.savefig(f"{base_dir}/{base_dir_tail}_perf_plot.png")


def _mean_detect_null(ls):
    if None in ls:
        return None
    else:
        return np.mean(ls)


def _std_detect_null(ls):
    if None in ls:
        return None
    else:
        return np.std(ls, ddof=_VAR_STD_DDOF)


def _round_half_up(n, decimals=0):
    multiplier = 10**decimals
    return math.floor(n * multiplier + 0.5) / multiplier


if __name__ == "__main__":
    main()

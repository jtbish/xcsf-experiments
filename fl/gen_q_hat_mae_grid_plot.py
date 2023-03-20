import glob
import itertools
import math
import pickle
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np

_EXPECTED_NUM_EXP_DIRS = 30
_EXPECTED_HISTORY_LEN = 250
_MONITOR_FREQ_GA_CALLSS = {4: 56, 8: 168, 12: 336}

_EPSILON_NOUGHT = 0.01
_BOUND_LINESTYLE = "dashed"
_BOUND_LINEWIDTH = 1.0
_BOUND_COLOR = "black"
_STD_ALPHA = 0.125
_VAR_STD_DDOF = 1  # use Bessel's correction for sample stdev
_ROUND_DEC_PLACES = 3

_ERR_COLOR = '#d62728'  # red
_ERR_YMIN = 0.0
_ERR_YMAX = 0.45

_GRID_SIZES = (4, 8, 12)
_SLIP_PROBS = (0, 0.1, 0.3, 0.5)
_DETRM_BASE_DIR = "./frozen/detrm"
_STOCA_BASE_DIR = "./frozen/stoca"
_FIG_DPI = 500


def main():
    # init global figure
    fig, axs = plt.subplots(nrows=len(_GRID_SIZES),
                            ncols=len(_SLIP_PROBS),
                            sharey=True)
#    fig_size = fig.get_size_inches()
#    scale_factor = 2
#    fig.set_size_inches(scale_factor * fig_size)
    fig.set_size_inches(22, 11)

    for (gs, sp) in itertools.product(_GRID_SIZES, _SLIP_PROBS):
        is_detrm = (sp == 0)
        if is_detrm:
            exp_dirs = glob.glob(f"{_DETRM_BASE_DIR}/gs_{gs}/6*")
        else:
            exp_dirs = glob.glob(f"{_STOCA_BASE_DIR}/gs_{gs}_sp_{sp}/6*")
        assert len(exp_dirs) == _EXPECTED_NUM_EXP_DIRS

        monitor_freq_ga_calls = _MONITOR_FREQ_GA_CALLSS[gs]
        monitor_ticks = [
            i * monitor_freq_ga_calls
            for i in range(1, _EXPECTED_HISTORY_LEN + 1)
        ]

        # init aggregate history
        aggr_q_hat_mae_history = OrderedDict(
            {num_ga_calls: []
             for num_ga_calls in monitor_ticks})

        # fill aggregate history
        for exp_dir in exp_dirs:
            with open(f"{exp_dir}/q_hat_mae_history.pkl", "rb") as fp:
                q_hat_mae_history = pickle.load(fp)
            assert len(q_hat_mae_history) == _EXPECTED_HISTORY_LEN

            # for history, keys are num ga calls done
            for (k, v) in q_hat_mae_history.items():
                assert k in aggr_q_hat_mae_history
                aggr_q_hat_mae_history[k].append(v)

        # check aggregate history is valid
        for (k, v) in aggr_q_hat_mae_history.items():
            assert len(v) == _EXPECTED_NUM_EXP_DIRS

        # transform keys of history to be "epochs" (starting at epoch 1)
        h = aggr_q_hat_mae_history
        aggr_q_hat_mae_history = {(idx + 1): v
                                  for (idx, v) in enumerate(list(h.values()))}
        # max epoch should be == to len of history
        assert max(aggr_q_hat_mae_history.keys()) == _EXPECTED_HISTORY_LEN

        mean_q_hat_maes = {}
        std_q_hat_maes = {}
        for (k, v) in aggr_q_hat_mae_history.items():
            # don't include mean or std if missing values for given key
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
        print(f"({gs}, {sp}): Qhat MAE @ {last_key} epochs: "
              f"{last_mean_round:.3f} +- {last_std_round:.3f}")

        subplot_row_idx = _GRID_SIZES.index(gs)
        subplot_col_idx = _SLIP_PROBS.index(sp)
        ax = axs[subplot_row_idx, subplot_col_idx]
        is_in_left_col = (subplot_col_idx == 0)
        is_in_bottom_row = (subplot_row_idx == (len(_GRID_SIZES) - 1))

        xs = np.array(list(mean_q_hat_maes.keys()))
        ys = np.array(list(mean_q_hat_maes.values()))
        err = np.array(list(std_q_hat_maes.values()))
        ax.plot(xs, ys, color=_ERR_COLOR)
        ax.fill_between(xs,
                        ys - err,
                        ys + err,
                        alpha=_STD_ALPHA,
                        color=_ERR_COLOR)
        ax.axhline(y=_EPSILON_NOUGHT,
                   linestyle=_BOUND_LINESTYLE,
                   linewidth=_BOUND_LINEWIDTH,
                   color=_BOUND_COLOR)
        ax.set_xlim(right=max(xs))
        ax.set_ylim(_ERR_YMIN, _ERR_YMAX)

        xticks_step = 25
        xticks = np.arange(0, max(xs) + xticks_step, step=xticks_step)
        ax.set_xticks(xticks)
        if is_in_bottom_row:
            # only have x axis label for bottom row
            ax.set_xlabel("Num. epochs")
        else:
            # disable all xtick labels for non bottom row
            ax.tick_params(axis="x", labelbottom=False)

        ax.set_yticks([i * 0.05 for i in range(0, 10 + 1)])
        if is_in_left_col:
            # only have y axis label for left col.
            ax.set_ylabel("$\hat{Q}$ MAE")
        ax.set_title(f"({gs}, {sp})")

    fig.tight_layout()
    #plt.show()
    plt.savefig("./q_hat_mae_grid_plot.png", bbox_inches="tight", dpi=_FIG_DPI)
    plt.savefig("./q_hat_mae_grid_plot.pdf", bbox_inches="tight", dpi=_FIG_DPI)


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

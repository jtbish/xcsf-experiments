import copy
import pickle
import sys
from collections import OrderedDict

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from rlenvs.environment import assess_perf
from rlenvs.frozen_lake import make_frozen_lake_env as make_fl

#plt.rcParams['text.usetex'] = True

XCSF_EXP_DIR = \
    "/home/Staff/uqjbish3/xcsf-experiments/fl/frozen/stoca/gs_4_sp_0.3/662578/662578"

PPLST_EXP_DIR = \
    "/home/Staff/uqjbish3/pplst-experiments/fl/frozen_redux/stoca/gs_4_sp_0.3/666017"

GOAL_STATE = (3, 3)
HOLE_STATES = [(0, 3), (1, 1), (3, 1), (3, 2)]
FROZEN_STATES = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 2), (1, 3), (2, 0),
                 (2, 1), (2, 2), (2, 3), (3, 0)]
TERMINAL_STATES = HOLE_STATES + [GOAL_STATE]
ACTIONS = (0, 1, 2, 3)
X_NOUGHT = 10
GAMMA = 0.95
GS = 4
SP = 0.3
IOD_STRAT = "frozen_repeat"
FL_SEED = 0

CMAP = "cool"
TERMINAL_COLOR = "black"
TEXT_COLOR = "black"
TEXT_FONTSIZE = "x-small"
Y_LABELPAD = 12
FIG_DPI = 200


def main():
    with open(f"{XCSF_EXP_DIR}/xcsf_ga_calls_14000.pkl", "rb") as fp:
        xcsf = pickle.load(fp)
    print(type(xcsf))

    xcsf_best_action_map = np.full((4, 4), np.nan)
    xcsf_cover_count_matrix = np.zeros((4, 4))
    xcsf_uniques = []

    for state in FROZEN_STATES:
        best_action = xcsf.select_action(state)
        xcsf_best_action_map[state] = best_action
        match_set = xcsf._gen_match_set(state)
        best_action_set = xcsf._gen_action_set(match_set, best_action)
        for clfr in best_action_set:
            if clfr not in xcsf_uniques:
                xcsf_uniques.append(clfr)
        img_idx = _transform_state(state)
        xcsf_cover_count_matrix[img_idx] = len(best_action_set)
    print(xcsf_best_action_map.T)
    print(xcsf.pop.num_macros, xcsf.pop.num_micros)
    print(len(xcsf_uniques))

    with open(f"{PPLST_EXP_DIR}/best_indiv_history.pkl", "rb") as fp:
        hist = pickle.load(fp)
    pplst_best_indiv = hist[250]
    print(type(pplst_best_indiv))

    pplst_best_action_map = np.full((4, 4), np.nan)
    pplst_cover_count_matrix = np.zeros((4, 4))
    pplst_uniques = []

    for state in FROZEN_STATES:
        best_action = pplst_best_indiv.select_action(state)
        pplst_best_action_map[state] = best_action
        match_set = [
            rule for rule in pplst_best_indiv.rules if rule.does_match(state)
        ]
        best_action_set = \
            [rule for rule in match_set if rule.action == best_action]
        for rule in best_action_set:
            if rule not in pplst_uniques:
                pplst_uniques.append(rule)
        img_idx = _transform_state(state)
        pplst_cover_count_matrix[img_idx] = len(best_action_set)

    print(pplst_best_action_map.T)
    print(len(pplst_best_indiv))
    print(len(pplst_uniques))

    class Policy:
        def __init__(self, best_action_map):
            self._bam = best_action_map

        def select_action(self, state):
            state = tuple(state)
            return self._bam[state]

    fl = make_fl(GS, SP, IOD_STRAT, FL_SEED)

    print(assess_perf(fl, Policy(xcsf_best_action_map), fl.si_size * 30,
                      GAMMA))
    print(
        assess_perf(fl, Policy(pplst_best_action_map), fl.si_size * 30, GAMMA))

    #sys.exit(1)

    transformed_frozen_states = [_transform_state(s) for s in FROZEN_STATES]
    xcsf_frozen_counts = [
        xcsf_cover_count_matrix[img_idx]
        for img_idx in transformed_frozen_states
    ]
    pplst_frozen_counts = [
        pplst_cover_count_matrix[img_idx]
        for img_idx in transformed_frozen_states
    ]
    xcsf_avg_density = np.mean(xcsf_frozen_counts)
    pplst_avg_density = np.mean(pplst_frozen_counts)
    print(xcsf_avg_density)
    print(pplst_avg_density)
    # sync the colormap ranges for both plots
    vmin = int(min(xcsf_frozen_counts + pplst_frozen_counts))
    vmax = int(max(xcsf_frozen_counts + pplst_frozen_counts))
    #print(vmin, vmax)

    # mask out terminals with black for given colormap
    cmap = copy.copy(matplotlib.cm.get_cmap(CMAP))
    cmap.set_bad(color=TERMINAL_COLOR)

    # now plot both matrices
    fig, axs = plt.subplots(nrows=1, ncols=2, sharey=True)
    xcsf_ax = axs[0]
    marr = np.ma.array(xcsf_cover_count_matrix,
                       mask=(xcsf_cover_count_matrix == 0))
    im = xcsf_ax.imshow(marr, cmap=cmap, vmin=vmin, vmax=vmax)
    # remove tick lines
    xcsf_ax.tick_params(bottom=False)
    xcsf_ax.tick_params(left=False)
    # add text annotations
    for img_idx in transformed_frozen_states:
        xcsf_ax.text(img_idx[1],
                     img_idx[0],
                     int(xcsf_cover_count_matrix[img_idx]),
                     ha="center",
                     va="center",
                     color=TEXT_COLOR,
                     fontsize=TEXT_FONTSIZE)
    xcsf_ax.set_xlabel("$x$")
    xcsf_ax.set_ylabel("$y$", rotation="horizontal", labelpad=Y_LABELPAD)
    xcsf_ax.set_title(
        "XCS\n" + r"$\vert R \vert = 203,\ \vert R_{\mathrm{BA}} \vert = 91$")

    pplst_ax = axs[1]
    marr = np.ma.array(pplst_cover_count_matrix,
                       mask=(pplst_cover_count_matrix == 0))
    im = pplst_ax.imshow(marr, cmap=cmap, vmin=vmin, vmax=vmax)
    # remove tick lines
    pplst_ax.tick_params(bottom=False)
    pplst_ax.tick_params(left=False)
    # add text annotations
    for img_idx in transformed_frozen_states:
        pplst_ax.text(img_idx[1],
                      img_idx[0],
                      int(pplst_cover_count_matrix[img_idx]),
                      ha="center",
                      va="center",
                      color=TEXT_COLOR,
                      fontsize=TEXT_FONTSIZE)
    pplst_ax.set_xlabel("$x$")
    pplst_ax.set_title(
        "PPL-ST\n" + r"$\vert R \vert = 7,\ \vert R_{\mathrm{BA}} \vert = 6$")

    # from https://stackoverflow.com/questions/13784201/how-to-have-one-colorbar-for-all-subplots
    cbar_step = 3
    cbar_ticks = list(range(vmin, vmax + 1, cbar_step))
    cbar = fig.colorbar(im,
                        ax=axs,
                        ticks=cbar_ticks,
                        shrink=0.5,
                        aspect=20 * 0.5)
    cbar.set_label("Num. rules in best $[A]$", labelpad=0.75 * Y_LABELPAD)

    #plt.show()
    plt.savefig("./xcsf_vs_pplst_rule_plot_fl4x4.png",
                bbox_inches="tight",
                dpi=FIG_DPI)
    plt.savefig("./xcsf_vs_pplst_rule_plot_fl4x4.pdf",
                bbox_inches="tight",
                dpi=FIG_DPI)


def _find_covered_states(cond_intervals):
    assert len(cond_intervals) == 2
    x_interval = cond_intervals[0]
    x_lower = x_interval.lower
    x_upper = x_interval.upper
    y_interval = cond_intervals[1]
    y_lower = y_interval.lower
    y_upper = y_interval.upper

    covered_states = []
    for x in range(x_lower, x_upper + 1):
        for y in range(y_lower, y_upper + 1):
            covered_states.append((x, y))
    return covered_states


def _transform_state(state):
    (x, y) = state
    row = y
    col = x
    return (row, col)


if __name__ == "__main__":
    main()

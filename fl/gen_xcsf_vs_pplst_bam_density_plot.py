import copy
import glob
import math
import pickle
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import AxesGrid

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

#matplotlib.use("Qt5Agg")

XCSF_BASE_DIR = \
    "/home/Staff/uqjbish3/xcsf-experiments/fl/frozen"
PPLST_BASE_DIR = \
    "/home/Staff/uqjbish3/pplst-experiments/fl/frozen_redux"
EXPECTED_NUM_EXP_DIRS = 30

GOAL_STATES = {4: (3, 3), 8: (7, 7), 12: (11, 11)}

HOLE_STATES = {
    4: [(0, 3), (1, 1), (3, 1), (3, 2)],
    8: [(1, 5), (1, 6), (2, 5), (3, 2), (3, 4), (3, 7), (4, 6), (5, 3), (6, 5),
        (6, 6)],
    12: [(0, 1), (1, 4), (1, 6), (1, 8), (2, 2), (3, 0), (3, 3), (3, 11),
         (4, 1), (4, 2), (4, 10), (5, 1), (5, 3), (5, 6), (5, 8), (5, 9),
         (5, 10), (6, 4), (8, 3), (8, 4), (8, 10), (8, 11), (9, 0), (9, 1),
         (9, 5), (9, 6), (9, 9), (9, 10), (10, 7)]
}

NUM_GENS = 250
# epoch size * num gens
XCSF_END_NUM_GA_CALLS = {
    4: (56 * NUM_GENS),
    8: (168 * NUM_GENS),
    12: (336 * NUM_GENS)
}

CMAP = "cool"
TERMINAL_COLOR = "black"
TEXT_COLOR = "black"
TEXT_FONTSIZES = {4: "medium", 8: "small", 12: "x-small"}
Y_LABELPAD = 12
GRID_LINEWIDTH = 1
FIG_DPI = 200

GLOBAL_VMIN = 1
GLOBAL_VMAX = 70


def main():
    gs = int(sys.argv[1])
    sp = float(sys.argv[2])
    use_global_cbar = bool(int(sys.argv[3]))

    # calc states for FL
    terminal_states = (HOLE_STATES[gs] + [GOAL_STATES[gs]])
    frozen_states = []
    for x in range(0, gs):
        for y in range(0, gs):
            state = (x, y)
            if state not in terminal_states:
                frozen_states.append(state)
    assert (len(terminal_states) + len(frozen_states)) == gs**2

    # get exp dirs for both algs
    is_detrm = (sp == 0.0)
    if is_detrm:
        xcsf_exp_dirs = glob.glob(f"{XCSF_BASE_DIR}/detrm/gs_{gs}/*/")
        pplst_exp_dirs = glob.glob(f"{PPLST_BASE_DIR}/detrm/gs_{gs}/*/")
    else:
        xcsf_exp_dirs = glob.glob(
            f"{XCSF_BASE_DIR}/stoca/gs_{gs}_sp_{sp:.1f}/*/")
        pplst_exp_dirs = glob.glob(
            f"{PPLST_BASE_DIR}/stoca/gs_{gs}_sp_{sp:.1f}/*/")
    assert len(xcsf_exp_dirs) == EXPECTED_NUM_EXP_DIRS
    assert len(pplst_exp_dirs) == EXPECTED_NUM_EXP_DIRS

    # get XCSF model aggr info
    xcsf_aggr_best_action_maps = []
    xcsf_aggr_cover_count_matrices = []
    xcsf_aggr_num_macross = []
    xcsf_aggr_bam_sizes = []

    end_ga_calls = XCSF_END_NUM_GA_CALLS[gs]
    for xcsf_exp_dir in xcsf_exp_dirs:
        print(xcsf_exp_dir)
        # assume pre-extracted
        with open(f"{xcsf_exp_dir}/xcsf_ga_calls_{end_ga_calls}.pkl",
                  "rb") as fp:
            xcsf = pickle.load(fp)

        xcsf_best_action_map = np.full((gs, gs), np.nan)
        xcsf_cover_count_matrix = np.zeros((gs, gs))
        xcsf_uniques = []

        for state in frozen_states:
            best_action = xcsf.select_action(state)
            xcsf_best_action_map[state] = best_action
            match_set = xcsf._gen_match_set(state)
            best_action_set = xcsf._gen_action_set(match_set, best_action)
            for clfr in best_action_set:
                if clfr not in xcsf_uniques:
                    xcsf_uniques.append(clfr)
            img_idx = _transform_state(state)
            xcsf_cover_count_matrix[img_idx] = len(best_action_set)

        #print(xcsf_best_action_map.T)
        #print(xcsf.pop.num_macros, xcsf.pop.num_micros)
        #print(len(xcsf_uniques))

        xcsf_aggr_best_action_maps.append(xcsf_best_action_map)
        xcsf_aggr_cover_count_matrices.append(xcsf_cover_count_matrix)
        xcsf_aggr_num_macross.append(xcsf.pop.num_macros)
        xcsf_aggr_bam_sizes.append(len(xcsf_uniques))

    assert len(xcsf_aggr_best_action_maps) == EXPECTED_NUM_EXP_DIRS
    assert len(xcsf_aggr_cover_count_matrices) == EXPECTED_NUM_EXP_DIRS
    assert len(xcsf_aggr_num_macross) == EXPECTED_NUM_EXP_DIRS
    assert len(xcsf_aggr_bam_sizes) == EXPECTED_NUM_EXP_DIRS

    # get PPL-ST model aggr info
    pplst_aggr_best_action_maps = []
    pplst_aggr_cover_count_matrices = []
    pplst_aggr_num_ruless = []
    pplst_aggr_bam_sizes = []

    for pplst_exp_dir in pplst_exp_dirs:
        print(pplst_exp_dir)
        with open(f"{pplst_exp_dir}/best_indiv_history.pkl", "rb") as fp:
            hist = pickle.load(fp)
        pplst_best_indiv = hist[NUM_GENS]
        #print(type(pplst_best_indiv))

        pplst_best_action_map = np.full((gs, gs), np.nan)
        pplst_cover_count_matrix = np.zeros((gs, gs))
        pplst_uniques = []

        for state in frozen_states:
            best_action = pplst_best_indiv.select_action(state)
            pplst_best_action_map[state] = best_action
            match_set = [
                rule for rule in pplst_best_indiv.rules
                if rule.does_match(state)
            ]
            best_action_set = \
                [rule for rule in match_set if rule.action == best_action]
            for rule in best_action_set:
                if rule not in pplst_uniques:
                    pplst_uniques.append(rule)
            img_idx = _transform_state(state)
            pplst_cover_count_matrix[img_idx] = len(best_action_set)

        #print(pplst_best_action_map.T)
        #print(len(pplst_best_indiv))
        #print(len(pplst_uniques))

        pplst_aggr_best_action_maps.append(pplst_best_action_map)
        pplst_aggr_cover_count_matrices.append(pplst_cover_count_matrix)
        pplst_aggr_num_ruless.append(len(pplst_best_indiv))
        pplst_aggr_bam_sizes.append(len(pplst_uniques))

    assert len(pplst_aggr_best_action_maps) == EXPECTED_NUM_EXP_DIRS
    assert len(pplst_aggr_cover_count_matrices) == EXPECTED_NUM_EXP_DIRS
    assert len(pplst_aggr_num_ruless) == EXPECTED_NUM_EXP_DIRS
    # all indiv sizes should be the same
    assert len(set(pplst_aggr_num_ruless)) == 1
    assert len(pplst_aggr_bam_sizes) == EXPECTED_NUM_EXP_DIRS

    xcsf_avg_cover_count_matrix = (sum(xcsf_aggr_cover_count_matrices) /
                                   len(xcsf_aggr_cover_count_matrices))
    #print(xcsf_avg_cover_count_matrix)
    pplst_avg_cover_count_matrix = (sum(pplst_aggr_cover_count_matrices) /
                                    len(pplst_aggr_cover_count_matrices))
    #print(pplst_avg_cover_count_matrix)

    transformed_frozen_states = [_transform_state(s) for s in frozen_states]
    if not use_global_cbar:
        xcsf_frozen_counts = [
            xcsf_avg_cover_count_matrix[img_idx]
            for img_idx in transformed_frozen_states
        ]
        pplst_frozen_counts = [
            pplst_avg_cover_count_matrix[img_idx]
            for img_idx in transformed_frozen_states
        ]
        # sync the colormap ranges for both plots
        vmin = 1
        vmax = math.ceil(max(xcsf_frozen_counts + pplst_frozen_counts))
        vrange = (vmax - vmin + 1)
        num_chunks = 7
        chunksize = int(round(vrange / num_chunks))
        cbar_ticks = [vmin]
        for i in range(1, num_chunks + 1):
            cbar_ticks.append(cbar_ticks[i - 1] + chunksize)
        cbar_ticks.append(vmax)
        if (cbar_ticks[-1] - cbar_ticks[-2]) < chunksize:
            del cbar_ticks[-2]
    else:
        # ignore calced local vmin and vmax
        vmin = GLOBAL_VMIN
        vmax = GLOBAL_VMAX
        cbar_ticks = [1] + [i * 10 for i in range(1, 7 + 1)]
        assert cbar_ticks[-1] == vmax
    #print(vmin, vmax)

    # mask out terminals with black for given colormap
    cmap = copy.copy(matplotlib.cm.get_cmap(CMAP))
    cmap.set_bad(color=TERMINAL_COLOR)

    # now plot both matrices

    fig = plt.figure(figsize=(10, 5))

    grid = AxesGrid(fig,
                    111,
                    nrows_ncols=(1, 2),
                    axes_pad=0.2,
                    cbar_mode='single',
                    cbar_location='right',
                    cbar_pad=0.2)

    xcsf_ax = grid[0]
    marr = np.ma.array(xcsf_avg_cover_count_matrix,
                       mask=(xcsf_avg_cover_count_matrix == 0))
    im = xcsf_ax.imshow(marr, cmap=cmap, vmin=vmin, vmax=vmax)

    # set axis major ticks and labels
    xcsf_ax.set_xticks(range(0, gs))
    xcsf_ax.set_yticks(range(0, gs))
    axis_labels = [str(e) for e in range(0, gs)]
    xcsf_ax.set_xticklabels(axis_labels)
    xcsf_ax.set_yticklabels(axis_labels)
    # set minor ticks for grid lines
    #https://stackoverflow.com/questions/38973868/adjusting-gridlines-and-ticks-in-matplotlib-imshow
    xcsf_ax.set_xticks(np.arange(-0.5, gs, 1), minor=True)
    xcsf_ax.set_yticks(np.arange(-0.5, gs, 1), minor=True)
    xcsf_ax.grid(which='minor',
                 color='black',
                 linestyle='-',
                 linewidth=GRID_LINEWIDTH)

    # remove tick lines
    xcsf_ax.tick_params(bottom=False, which="both")
    xcsf_ax.tick_params(left=False, which="both")
    # add text annotations (2 dec pl)
    for img_idx in transformed_frozen_states:
        val = _round_half_up(xcsf_avg_cover_count_matrix[img_idx], decimals=1)
        xcsf_ax.text(img_idx[1],
                     img_idx[0],
                     f"{val:.1f}",
                     ha="center",
                     va="center",
                     color=TEXT_COLOR,
                     fontsize=TEXT_FONTSIZES[gs])
    xcsf_ax.set_xlabel("$x$")
    xcsf_ax.set_ylabel("$y$", rotation="horizontal", labelpad=Y_LABELPAD)
    xcsf_avg_num_macros = _round_half_up(np.mean(xcsf_aggr_num_macross),
                                         decimals=1)
    xcsf_avg_bam_size = _round_half_up(np.mean(xcsf_aggr_bam_sizes),
                                       decimals=1)
    print(f"XCS avg num macros: {xcsf_avg_num_macros:.1f}")
    print(f"XCS avg bam size: {xcsf_avg_bam_size:.1f}")

    #xcsf_ax_title = "XCS"
    xcsf_ax_title = "XCS\n" + r"$\overline{\vert R \vert} = $" + \
        f"{xcsf_avg_num_macros:.1f}, " + \
        r"$\overline{\vert R_{\mathrm{BA}} \vert} = $" + \
        f"{xcsf_avg_bam_size:.1f}"
    xcsf_ax.set_title(xcsf_ax_title)

    pplst_ax = grid[1]
    marr = np.ma.array(pplst_avg_cover_count_matrix,
                       mask=(pplst_avg_cover_count_matrix == 0))
    im = pplst_ax.imshow(marr, cmap=cmap, vmin=vmin, vmax=vmax)

    # set axis major ticks and labels
    pplst_ax.set_xticks(range(0, gs))
    pplst_ax.set_yticks(range(0, gs))
    axis_labels = [str(e) for e in range(0, gs)]
    pplst_ax.set_xticklabels(axis_labels)
    pplst_ax.set_yticklabels(axis_labels)
    # set minor ticks for grid lines
    pplst_ax.set_xticks(np.arange(-0.5, gs, 1), minor=True)
    pplst_ax.set_yticks(np.arange(-0.5, gs, 1), minor=True)
    pplst_ax.grid(which='minor',
                  color='black',
                  linestyle='-',
                  linewidth=GRID_LINEWIDTH)

    # remove tick lines
    pplst_ax.tick_params(bottom=False, which="both")
    pplst_ax.tick_params(left=False, which="both")
    # add text annotations
    for img_idx in transformed_frozen_states:
        val = _round_half_up(pplst_avg_cover_count_matrix[img_idx], decimals=1)
        pplst_ax.text(img_idx[1],
                      img_idx[0],
                      f"{val:.1f}",
                      ha="center",
                      va="center",
                      color=TEXT_COLOR,
                      fontsize=TEXT_FONTSIZES[gs])
    pplst_ax.set_xlabel("$x$")
    pplst_num_rules = list(set(pplst_aggr_num_ruless))[0]
    pplst_avg_bam_size = _round_half_up(np.mean(pplst_aggr_bam_sizes),
                                        decimals=1)
    print(f"PPL-ST avg bam size: {pplst_avg_bam_size:.1f}")
    #pplst_ax_title = "PPL-ST"
    pplst_ax_title = "PPL-ST\n" + r"$\vert R \vert = $" + \
        f"{pplst_num_rules}, " + \
        r"$\overline{\vert R_{\mathrm{BA}} \vert} = $" + \
        f"{pplst_avg_bam_size:.1f}"
    pplst_ax.set_title(pplst_ax_title)

    # thanks: https://jdhao.github.io/2017/06/11/mpl_multiplot_one_colorbar/
    cax = grid.cbar_axes[0]
    cbar = fig.colorbar(im, cax=cax, ticks=cbar_ticks)
    cbar.set_label("Mean num. rules in best $[A]$", labelpad=0.75 * Y_LABELPAD)

    #plt.show()
    if use_global_cbar:
        ext = "global"
    else:
        ext = "local"
    plt.savefig(f"./xcs_vs_pplst_bam_density_{gs}_{sp}_{ext}.png",
                bbox_inches="tight",
                dpi=FIG_DPI)
    plt.savefig(f"./xcs_vs_pplst_bam_density_{gs}_{sp}_{ext}.pdf",
                bbox_inches="tight",
                dpi=FIG_DPI)


def _transform_state(state):
    (x, y) = state
    row = y
    col = x
    return (row, col)


def _round_half_up(n, decimals=0):
    multiplier = 10**decimals
    rounded = math.floor(n * multiplier + 0.5) / multiplier
    return rounded


if __name__ == "__main__":
    main()

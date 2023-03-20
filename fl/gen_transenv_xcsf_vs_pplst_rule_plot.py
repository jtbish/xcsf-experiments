import copy
import glob
import itertools
import math
import pickle
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Qt5Agg")

XCSF_BASE_DIR = \
    "/home/Staff/uqjbish3/xcsf-experiments/fl/frozen"
PPLST_BASE_DIR = \
    "/home/Staff/uqjbish3/pplst-experiments/fl/frozen_redux"
EXPECTED_NUM_EXP_DIRS = 30

GOAL_STATES_ALL_GSS = {4: (3, 3), 8: (7, 7), 12: (11, 11)}

HOLE_STATES_ALL_GSS = {
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
TEXT_FONTSIZE = "x-small"
Y_LABELPAD = 12
GRID_LINEWIDTH = 1
FIG_DPI = 500

GRID_SIZES = (4, 8, 12)
SLIP_PROBS = (0, 0.1, 0.3, 0.5)


def main():
    # init global figure
    # 2 subplots for each col since two heatmaps
    fig, axs = plt.subplots(nrows=len(GRID_SIZES), ncols=(2 * len(SLIP_PROBS)))
    fig.set_size_inches(22, 11)

    avg_cover_count_matrices = {
        (gs, sp): {
            "xcsf": None,
            "pplst": None
        }
        for (gs, sp) in itertools.product(GRID_SIZES, SLIP_PROBS)
    }

    frozen_states_all_gss = {}
    for gs in GRID_SIZES:
        terminal_states = (HOLE_STATES_ALL_GSS[gs] + [GOAL_STATES_ALL_GSS[gs]])
        frozen_states = []
        for x in range(0, gs):
            for y in range(0, gs):
                state = (x, y)
                if state not in terminal_states:
                    frozen_states.append(state)
        assert (len(terminal_states) + len(frozen_states)) == gs**2
        frozen_states_all_gss[gs] = frozen_states

    for (gs, sp) in itertools.product(GRID_SIZES, SLIP_PROBS):
        print(gs, sp)

        frozen_states = frozen_states_all_gss[gs]

        # get exp dirs for both algs for this (gs, sp) combo
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
        pplst_avg_cover_count_matrix = (sum(pplst_aggr_cover_count_matrices) /
                                        len(pplst_aggr_cover_count_matrices))

        avg_cover_count_matrices[(gs, sp)]["xcsf"] = \
            xcsf_avg_cover_count_matrix
        avg_cover_count_matrices[(gs, sp)]["pplst"] = \
            pplst_avg_cover_count_matrix

    # now, plot each one
    # but first... calc the global colorbar range
    global_vmin = None
    global_vmax = None
    for (gs, sp) in itertools.product(GRID_SIZES, SLIP_PROBS):
        print(gs, sp)
        frozen_states = frozen_states_all_gss[gs]
        transformed_frozen_states = [
            _transform_state(s) for s in frozen_states
        ]
        xcsf_avg_cover_count_matrix = avg_cover_count_matrices[(gs,
                                                                sp)]["xcsf"]
        pplst_avg_cover_count_matrix = avg_cover_count_matrices[(gs,
                                                                 sp)]["pplst"]
        xcsf_frozen_counts = [
            xcsf_avg_cover_count_matrix[img_idx]
            for img_idx in transformed_frozen_states
        ]
        pplst_frozen_counts = [
            pplst_avg_cover_count_matrix[img_idx]
            for img_idx in transformed_frozen_states
        ]
        local_vmin = math.floor(min(xcsf_frozen_counts + pplst_frozen_counts))
        local_vmax = math.ceil(max(xcsf_frozen_counts + pplst_frozen_counts))

        if (global_vmin is None) or (local_vmin < global_vmin):
            global_vmin = local_vmin
        if (global_vmax is None) or (local_vmax > global_vmax):
            global_vmax = local_vmax

    # now, plot each one!
    # mask out terminals with black for given colormap
    cmap = copy.copy(matplotlib.cm.get_cmap(CMAP))
    cmap.set_bad(color=TERMINAL_COLOR)

    for (gs, sp) in itertools.product(GRID_SIZES, SLIP_PROBS):
        frozen_states = frozen_states_all_gss[gs]
        transformed_frozen_states = [
            _transform_state(s) for s in frozen_states
        ]

        # setup shared stuff for each subplot
        subplot_row_idx = GRID_SIZES.index(gs)
        subplot_base_col_idx = (SLIP_PROBS.index(sp) * 2)
        xcsf_col_idx = subplot_base_col_idx
        pplst_col_idx = (xcsf_col_idx + 1)
        xcsf_ax = axs[subplot_row_idx][xcsf_col_idx]
        pplst_ax = axs[subplot_row_idx][pplst_col_idx]

        # now plot both matrices
        # xcsf
        xcsf_avg_cover_count_matrix = avg_cover_count_matrices[(gs,
                                                                sp)]["xcsf"]
        marr = np.ma.array(xcsf_avg_cover_count_matrix,
                           mask=(xcsf_avg_cover_count_matrix == 0))
        im = xcsf_ax.imshow(marr,
                            cmap=cmap,
                            vmin=global_vmin,
                            vmax=global_vmax)

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
            val = _round_half_up(xcsf_avg_cover_count_matrix[img_idx],
                                 decimals=2)
            xcsf_ax.text(img_idx[1],
                         img_idx[0],
                         f"{val:.2f}",
                         ha="center",
                         va="center",
                         color=TEXT_COLOR,
                         fontsize=TEXT_FONTSIZE)
        xcsf_ax.set_xlabel("$x$")
        xcsf_ax.set_ylabel("$y$", rotation="horizontal", labelpad=Y_LABELPAD)
        #xcsf_avg_num_macros = _round_half_up(np.mean(xcsf_aggr_num_macross),
        #                                     decimals=2)
        #xcsf_avg_bam_size = _round_half_up(np.mean(xcsf_aggr_bam_sizes),
        #                                   decimals=2)
        #xcsf_ax_title = "XCS\n" + r"$\overline{\vert R \vert} = $" + \
        #    f"{xcsf_avg_num_macros:.2f}, " + \
        #    r"$\overline{\vert R_{\mathrm{BA}} \vert} = $" + \
        #    f"{xcsf_avg_bam_size:.2f}"
        #xcsf_ax.set_title(xcsf_ax_title)

        # pplst
        pplst_avg_cover_count_matrix = avg_cover_count_matrices[(gs,
                                                                 sp)]["pplst"]
        marr = np.ma.array(pplst_avg_cover_count_matrix,
                           mask=(pplst_avg_cover_count_matrix == 0))
        im = pplst_ax.imshow(marr,
                             cmap=cmap,
                             vmin=global_vmin,
                             vmax=global_vmax)

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
            val = _round_half_up(pplst_avg_cover_count_matrix[img_idx],
                                 decimals=2)
            pplst_ax.text(img_idx[1],
                          img_idx[0],
                          f"{val:.2f}",
                          ha="center",
                          va="center",
                          color=TEXT_COLOR,
                          fontsize=TEXT_FONTSIZE)
        pplst_ax.set_xlabel("$x$")
        #pplst_num_rules = list(set(pplst_aggr_num_ruless))[0]
        #pplst_avg_bam_size = _round_half_up(np.mean(pplst_aggr_bam_sizes),
        #                                    decimals=2)
        #pplst_ax_title = "PPL-ST\n" + r"$\vert R \vert = $" + \
        #    f"{pplst_num_rules}, " + \
        #    r"$\overline{\vert R_{\mathrm{BA}} \vert} = $" + \
        #    f"{pplst_avg_bam_size:.2f}"
        #pplst_ax.set_title(pplst_ax_title)

    #sys.exit(1)

    plt.show()
    #plt.savefig("./xcsf_vs_pplst_rule_plot_fl4x4.png",
    #            bbox_inches="tight",
    #            dpi=FIG_DPI)
    #plt.savefig("./xcsf_vs_pplst_rule_plot_fl4x4.pdf",
    #            bbox_inches="tight",
    #            dpi=FIG_DPI)


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

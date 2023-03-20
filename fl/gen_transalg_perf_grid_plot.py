import glob
import itertools
import math
import pickle
from collections import OrderedDict

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scikit_posthocs
import scipy
from statsmodels.stats.multicomp import pairwise_tukeyhsd

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# for hypothesis testing
_ALPHA = 0.05

_EXPECTED_NUM_EXP_DIRS = 30
_NUM_EPOCHS = 250
_XCSF_EXPECTED_HISTORY_LEN = _NUM_EPOCHS
_XCSF_MONITOR_FREQ_GA_CALLSS = {4: 56, 8: 168, 12: 336}
_PPL_NUM_GENS = _NUM_EPOCHS
_PPL_EXPECTED_HISTORY_LEN = (_PPL_NUM_GENS + 1)
_EXPECTED_HISTORY_LENS = {
    "xcsf": _XCSF_EXPECTED_HISTORY_LEN,
    "ppldl": _PPL_EXPECTED_HISTORY_LEN,
    "pplst": _PPL_EXPECTED_HISTORY_LEN
}

_MAX_PERF = 1.0
_BOUND_LINESTYLE = "dashed"
_BOUND_LINEWIDTH = 1.0
_BOUND_COLOR = "black"
_STD_ALPHA = 0.125
_VAR_STD_DDOF = 1  # use Bessel's correction
_ROUND_DEC_PLACES = 3

_EPOCH_MIN = 0
_EPOCH_MAX = _NUM_EPOCHS
_EPOCH_TICK_STEP = 25
_EPOCH_TICKS = np.arange(_EPOCH_MIN, (_EPOCH_MAX + _EPOCH_TICK_STEP),
                         step=_EPOCH_TICK_STEP)

_PERF_MIN = 0.0
_PERF_MAX = 1.2
_PERF_TICK_STEP = 0.2
_PERF_TICKS = np.arange(_PERF_MIN, (_PERF_MAX + _PERF_TICK_STEP),
                        step=_PERF_TICK_STEP)

_GRID_SIZES = (4, 8, 12)
_SLIP_PROBS = (0, 0.1, 0.3, 0.5)
_ALGS = ("xcsf", "ppldl", "pplst")

_HOMEDIR = "/home/Staff/uqjbish3"
_BASE_DIRS = {
    "xcsf": f"{_HOMEDIR}/xcsf-experiments/fl/frozen",
    "ppldl": f"{_HOMEDIR}/ppl-experiments/fl/frozen",
    "pplst": f"{_HOMEDIR}/pplst-experiments/fl/frozen_redux"
}
_PERF_HISTORY_FILENAMES = {
    "xcsf": "perf_history.pkl",
    "ppldl": "best_indiv_test_perf_history.pkl"
}
_PERF_HISTORY_FILENAMES["pplst"] = _PERF_HISTORY_FILENAMES["ppldl"]

_PLOT_COLORS = {"xcsf": '#1f77b4', "ppldl": '#ff7f0e', "pplst": '#9467bd'}

_EMP_OPT_PERFS_PKL_FILE = \
    "./FrozenLake_emp_opt_perfs_gamma_0.95_iodsb_frozen.pkl"

_FIG_DPI = 500


def main():
    # init global figure
    fig, axs = plt.subplots(nrows=len(_GRID_SIZES),
                            ncols=len(_SLIP_PROBS),
                            sharey=True)
    # make fig 2x normal size
    #    fig_size = fig.get_size_inches()
    #    scale_factor = 2
    #    fig.set_size_inches(scale_factor * fig_size)
    fig.set_size_inches(22, 11)

    with open(f"{_EMP_OPT_PERFS_PKL_FILE}", "rb") as fp:
        empirical_opt_perfs = pickle.load(fp)

    for (gs, sp) in itertools.product(_GRID_SIZES, _SLIP_PROBS):
        # setup shared stuff for each subplot
        subplot_row_idx = _GRID_SIZES.index(gs)
        subplot_col_idx = _SLIP_PROBS.index(sp)

        ax = axs[subplot_row_idx, subplot_col_idx]
        ax.set_title(f"({gs}, {sp})")
        ax.set_xlim(_EPOCH_MIN, _EPOCH_MAX)
        ax.set_xticks(_EPOCH_TICKS)
        ax.axhline(y=_MAX_PERF,
                   linestyle=_BOUND_LINESTYLE,
                   linewidth=_BOUND_LINEWIDTH,
                   color=_BOUND_COLOR)
        ax.set_ylim(_PERF_MIN, _PERF_MAX)
        ax.set_yticks(_PERF_TICKS)

        is_in_bottom_row = (subplot_row_idx == (len(_GRID_SIZES) - 1))
        is_in_left_col = (subplot_col_idx == 0)

        if is_in_bottom_row:
            # only have x axis label for bottom row
            ax.set_xlabel("Num. epochs")
        else:
            # disable all xtick labels for non bottom row
            ax.tick_params(axis="x", labelbottom=False)

        if is_in_left_col:
            # only have y axis label for left col.
            ax.set_ylabel("Testing performance")

        transalg_end_data = []
        for alg in _ALGS:
            base_dir = _BASE_DIRS[alg]
            perf_history_filename = _PERF_HISTORY_FILENAMES[alg]

            is_detrm = (sp == 0)
            if is_detrm:
                exp_dirs = glob.glob(f"{base_dir}/detrm/gs_{gs}/6*")
            else:
                exp_dirs = glob.glob(f"{base_dir}/stoca/gs_{gs}_sp_{sp}/6*")
            assert len(exp_dirs) == _EXPECTED_NUM_EXP_DIRS

            expected_history_len = _EXPECTED_HISTORY_LENS[alg]
            exp_ticks = _get_exp_ticks(alg, expected_history_len, gs)
            # init aggregate history in "native tick units"
            aggr_perf_history = OrderedDict({tick: [] for tick in exp_ticks})

            # get optimal perf
            emp_opt_perf = empirical_opt_perfs[(gs, sp)].perf

            # fill aggregate history
            for exp_dir in exp_dirs:
                with open(f"{exp_dir}/{perf_history_filename}", "rb") as fp:
                    perf_history = pickle.load(fp)
                assert len(perf_history) == expected_history_len

                # for history, keys are exp ticks
                for (k, v) in perf_history.items():
                    assert k in aggr_perf_history
                    # v is perf assessment response
                    perf = v.perf
                    perf_pcnt = (perf / emp_opt_perf)
                    aggr_perf_history[k].append(perf_pcnt)

            # check aggregate history is valid
            for (k, v) in aggr_perf_history.items():
                assert len(v) == _EXPECTED_NUM_EXP_DIRS

            # transform aggregate history into keys in units of epochs
            aggr_perf_history = _transform_aggr_perf_history(
                aggr_perf_history, alg)
            assert len(aggr_perf_history) == _NUM_EPOCHS

            mean_perfs = {
                k: np.mean(v)
                for (k, v) in aggr_perf_history.items()
            }
            std_perfs = {
                k: np.std(v, ddof=_VAR_STD_DDOF)
                for (k, v) in aggr_perf_history.items()
            }
            last_key = max(mean_perfs.keys())
            last_mean_round = _round_half_up(mean_perfs[last_key],
                                             _ROUND_DEC_PLACES)
            last_std_round = _round_half_up(std_perfs[last_key],
                                            _ROUND_DEC_PLACES)
            print(f"{alg}, ({gs}, {sp}): Perf @ {last_key}: "
                  f"{last_mean_round:.3f} +- {last_std_round:.3f}")

            alg_end_data = aggr_perf_history[last_key]
            assert len(alg_end_data) == _EXPECTED_NUM_EXP_DIRS
            #normal_test_res = scipy.stats.normaltest(alg_end_data)
            normal_test_res = scipy.stats.shapiro(alg_end_data)
            print(f"({gs}, {sp}, {alg}) normal test: {normal_test_res}")

            transalg_end_data.append(alg_end_data)

            color = _PLOT_COLORS[alg]
            xs = np.array(list(mean_perfs.keys()))
            ys = np.array(list(mean_perfs.values()))
            err = np.array(list(std_perfs.values()))
            ax.plot(xs, ys, color=color)
            ax.fill_between(xs,
                            ys - err,
                            ys + err,
                            alpha=_STD_ALPHA,
                            color=color)

        # do test on transalg end data
        # idxs: xcsf == 0, ppldl == 1, pplst == 2
        # need to compare idx 2 with 0 and 1, ie do two tests
#        print(f"Doing Welch's t test for ({gs}, {sp})")
#        corrected_alpha = _ALPHA / 2
#        res1 = scipy.stats.ttest_ind(transalg_end_data[2],
#                                     transalg_end_data[0],
#                                     equal_var=False)
#        print(f"pplst vs. xcsf: p={res1.pvalue}")
#        if res1.pvalue < corrected_alpha:
#            print("pplst > xcsf")
#        else:
#            print("pplst == xcsf")
#        res2 = scipy.stats.ttest_ind(transalg_end_data[2],
#                                     transalg_end_data[1],
#                                     equal_var=False)
#        print(f"pplst vs. ppldl: p={res2.pvalue}")
#        if res2.pvalue < corrected_alpha:
#            print("pplst > ppldl")
#        else:
#            print("pplst == ppldl")
#        print("\n")

        print(f"Doing Mann-Whitney U test for ({gs}, {sp})")
        corrected_alpha = _ALPHA / 2
        res1 = scipy.stats.mannwhitneyu(transalg_end_data[2],
                                        transalg_end_data[0])
        print(f"pplst vs. xcsf: p={res1.pvalue}")
        if res1.pvalue < corrected_alpha:
            print("pplst != xcsf")
        else:
            print("pplst == xcsf")
        res2 = scipy.stats.mannwhitneyu(transalg_end_data[2],
                                        transalg_end_data[1])
        print(f"pplst vs. ppldl: p={res2.pvalue}")
        if res2.pvalue < corrected_alpha:
            print("pplst != ppldl")
        else:
            print("pplst == ppldl")
        print("\n")


#        # do ANOVA on transalg end data
#        res = scipy.stats.f_oneway(*transalg_end_data)
#        print(f"({gs}, {sp}) ANOVA: {res}")
#        if res.pvalue < _ALPHA:
#            print("ANOVA res is significant, doing pairwise tukey")
#            # make pandas df to hold data for statsmodels
#            # from e.g. at https://www.statology.org/tukey-test-python/
#            # basically, linearise all scores and say chunks of 30 belong to
#            # diff algs
#            scores = []
#            for alg_end_data in transalg_end_data:
#                scores.extend(alg_end_data)
#            df = pd.DataFrame({
#                "score":
#                scores,
#                "group":
#                np.repeat(list(_ALGS), repeats=_EXPECTED_NUM_EXP_DIRS)
#            })
#            print(
#                pairwise_tukeyhsd(endog=df["score"],
#                                  groups=df["group"],
#                                  alpha=_ALPHA))
#
#        # do non-parametric ANOVA too
#        res = scipy.stats.kruskal(*transalg_end_data)
#        print(f"({gs}, {sp}) Kruskal: {res}")
#        if res.pvalue < _ALPHA:
#            print("Kruskal res is significant, doing Dunn test")
#            print(
#                scikit_posthocs.posthoc_dunn(transalg_end_data,
#                                             p_adjust="bonferroni"))
#
#        print("\n\n")

    fig.tight_layout()
    #plt.show()
    plt.savefig("./transalg_perf_grid_plot.png",
                bbox_inches="tight",
                dpi=_FIG_DPI)
    plt.savefig("./transalg_perf_grid_plot.pdf",
                bbox_inches="tight",
                dpi=_FIG_DPI)


def _get_exp_ticks(alg, expected_history_len, gs):
    assert alg in _ALGS
    if alg == "xcsf":
        monitor_freq_ga_calls = _XCSF_MONITOR_FREQ_GA_CALLSS[gs]
        monitor_ga_calls_ticks = [
            i * monitor_freq_ga_calls
            for i in range(1, expected_history_len + 1)
        ]
        return monitor_ga_calls_ticks
    elif alg == "ppldl" or alg == "pplst":
        # expected history len inclusive of last gen
        gen_ticks = list(range(0, expected_history_len))
        return gen_ticks
    else:
        assert False


def _transform_aggr_perf_history(aggr_perf_history, alg):
    assert alg in _ALGS
    if alg == "xcsf":
        # for XCSF, idx n is epoch n+1
        return {(idx + 1): v
                for (idx, v) in enumerate(list(aggr_perf_history.values()))}
    elif alg == "ppldl" or alg == "pplst":
        # for PPL, discard first entry (gen 0), then idx n is epoch n+1
        transformed = OrderedDict(
            {k: v
             for (k, v) in aggr_perf_history.items() if k != 0})
        return {(idx + 1): v
                for (idx, v) in enumerate(list(transformed.values()))}
    else:
        assert False


def _round_half_up(n, decimals=0):
    multiplier = 10**decimals
    return math.floor(n * multiplier + 0.5) / multiplier


if __name__ == "__main__":
    main()

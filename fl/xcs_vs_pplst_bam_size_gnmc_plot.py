import glob
import pickle
import sys
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pplst.inference import infer_action_and_action_set
from rlenvs.frozen_lake import make_frozen_lake_env as make_fl

_GRID_SIZE = 12
_SLIP_PROB = 0.1
_EXPECTED_NUM_EXP_DIRS = 30
_GNMC_RHOS = [0.33, 0.66, 0.99]

_XCS_FINAL_NUM_GA_CALLS = 84000
_PPLST_FINAL_NUM_GENS = 250

_XCS_BASE_DIR = f"./frozen/stoca/gs_12_sp_{_SLIP_PROB}"
_PPLST_BASE_DIR = \
    f"/home/Staff/uqjbish3/pplst-experiments/fl/frozen_redux/stoca/gs_12_sp_{_SLIP_PROB}"

_XCS_X_NOUGHT = 10

_Y_MIN = 0
_YTICK_MAXS = {0.1: 650, 0.3: 800}

#_FIGSIZE = (5, 5)

_AXS_BREAKPOINTS = {0.1: (250, 450), 0.3: (375, 600)}


class GNMCCompactedPolicy:
    def __init__(self, compacted_pop, action_space):
        self._pop = compacted_pop
        self._action_space = action_space

    def select_action(self, obs):
        aug_obs = np.concatenate(([_XCS_X_NOUGHT], obs))
        assert len(aug_obs) == 3

        match_set = [clfr for clfr in self._pop if clfr.does_match(obs)]
        prediction_arr = OrderedDict({a: None for a in self._action_space})

        for a in self._action_space:

            action_set = [clfr for clfr in match_set if clfr.action == a]

            if len(action_set) > 0:
                numer = sum([(clfr.fitness * clfr.prediction(aug_obs))
                             for clfr in action_set])
                denom = sum([clfr.fitness for clfr in action_set])
                prediction_arr[a] = (numer / denom)

        prediction_arr = OrderedDict(
            {k: v
             for (k, v) in prediction_arr.items() if v is not None})
        assert len(prediction_arr) > 0
        return max(prediction_arr, key=prediction_arr.get)


def main():

    xcs_exp_dirs = glob.glob(f"{_XCS_BASE_DIR}/6*")
    pplst_exp_dirs = glob.glob(f"{_PPLST_BASE_DIR}/6*")

    assert len(xcs_exp_dirs) == _EXPECTED_NUM_EXP_DIRS
    assert len(pplst_exp_dirs) == _EXPECTED_NUM_EXP_DIRS

    env = make_fl(grid_size=12, slip_prob=_SLIP_PROB, iod_strat="top_left")
    nonterm_obss = env.nonterminal_states
    print(nonterm_obss)

    # XCS first
    # rho == 0 means no compaction
    xcs_bam_sizes = {rho: [] for rho in ([0] + _GNMC_RHOS)}

    for xcs_exp_dir in xcs_exp_dirs:

        # no compaction first
        with open(f"{xcs_exp_dir}/xcsf_ga_calls_{_XCS_FINAL_NUM_GA_CALLS}.pkl",
                  "rb") as fp:
            # actual model
            xcs = pickle.load(fp)

        xcs_bam_sizes[0].append(
            _calc_bam_size_xcs(pop=xcs.pop,
                               policy=xcs,
                               nonterm_obss=nonterm_obss))

        # then compaction
        for rho in _GNMC_RHOS:
            with open(f"{xcs_exp_dir}/gnmc/pop_lambda_fit_rho_{rho}.pkl",
                      "rb") as fp:
                pop = pickle.load(fp)
            policy = GNMCCompactedPolicy(pop, env.action_space)

            xcs_bam_sizes[rho].append(
                _calc_bam_size_xcs(pop, policy, nonterm_obss))

    for v in xcs_bam_sizes.values():
        assert len(v) == _EXPECTED_NUM_EXP_DIRS

    print("XCS")
    for (k, v) in xcs_bam_sizes.items():
        print(k, v)
        print("\n")
    print(np.mean(xcs_bam_sizes[0]))

    # PPLST
    pplst_bam_sizes = []
    for pplst_exp_dir in pplst_exp_dirs:
        with open(f"{pplst_exp_dir}/best_indiv_history.pkl", "rb") as fp:
            hist = pickle.load(fp)
        best_indiv = hist[_PPLST_FINAL_NUM_GENS]
        pplst_bam_sizes.append(_calc_bam_size_pplst(best_indiv, nonterm_obss))

    assert len(pplst_bam_sizes) == _EXPECTED_NUM_EXP_DIRS

    print("PPL-ST")
    print(pplst_bam_sizes)
    print(np.mean(pplst_bam_sizes))

    df = pd.DataFrame(columns=[
        "XCS no comp.", "XCS rho 0.33", "XCS rho 0.66", "XCS rho 0.99", "PPLST"
    ])
    df["XCS no comp."] = xcs_bam_sizes[0]
    df["XCS rho 0.33"] = xcs_bam_sizes[0.33]
    df["XCS rho 0.66"] = xcs_bam_sizes[0.66]
    df["XCS rho 0.99"] = xcs_bam_sizes[0.99]
    df["PPLST"] = pplst_bam_sizes

    # now do the plot
    # following broken axis example from:
    # https://matplotlib.org/stable/gallery/subplots_axes_and_figures/broken_axis.html
    if _SLIP_PROB == 0.1:
        height_ratios = [(200 / (250 + 200)), (250 / (250 + 200))]
    elif _SLIP_PROB == 0.3:
        height_ratios = [(200 / (375 + 200)), (375 / (375 + 200))]
    else:
        assert False
    fig, (ax1,
          ax2) = plt.subplots(nrows=2,
                              ncols=1,
                              sharex=True,
                              gridspec_kw={'height_ratios': height_ratios})
    fig.subplots_adjust(hspace=0.125)

    #print(df)
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    # blue, brown, olive, red, purple
    colors = [colors[0], colors[5], colors[8], colors[3], colors[4]]

    keys = [
        "XCS no comp.", "XCS rho 0.33", "XCS rho 0.66", "XCS rho 0.99", "PPLST"
    ]
    medians = []
    for (key, color) in zip(keys, colors):
        data = df[key]
        min_ = np.min(data)
        med = np.median(data)
        medians.append(med)
        max_ = np.max(data)
        yerr = np.asarray([(med - min_), (max_ - med)]).reshape((2, 1))
        # plot data on both top and bottom axes
        for ax in (ax1, ax2):
            ax.errorbar(x=key,
                        y=med,
                        yerr=yerr,
                        color=color,
                        capsize=5,
                        marker="o",
                        elinewidth=2,
                        markersize=5)

    # set yticks for both axs
    ytick_step = 25
    yticks = list(range(0, (_YTICK_MAXS[_SLIP_PROB] + ytick_step), ytick_step))
    # filter out breakpoints
    breakpoints = _AXS_BREAKPOINTS[_SLIP_PROB]
    for breakpoint in breakpoints:
        if breakpoint in yticks:
            yticks.remove(breakpoint)

    for ax in (ax1, ax2):
        ax.set_yticks(yticks)

    # change zooms of top and bottom axes
    ax1_bottom = _AXS_BREAKPOINTS[_SLIP_PROB][1]
    ax1_top = _YTICK_MAXS[_SLIP_PROB]
    print(ax1_bottom, ax1_top)
    ax1.set_ylim(bottom=ax1_bottom, top=ax1_top)

    ax2_bottom = _Y_MIN
    ax2_top = _AXS_BREAKPOINTS[_SLIP_PROB][0]
    print(ax2_bottom, ax2_top)
    ax2.set_ylim(bottom=ax2_bottom, top=ax2_top)

    # hide the spines between ax and ax2
    ax1.spines.bottom.set_visible(False)
    ax2.spines.top.set_visible(False)
    ax1.tick_params(bottom=False)  # don't put tick labels at the top
    ax2.xaxis.tick_bottom()

    for ax in (ax1, ax2):
        ax.grid(which='major', axis='y')

    # Now, let's turn towards the cut-out slanted lines.
    # We create line objects in axes coordinates, in which (0,0), (0,1),
    # (1,0), and (1,1) are the four corners of the axes.
    # The slanted lines themselves are markers at those locations, such that the
    # lines keep their angle and position, independent of the axes size or scale
    # Finally, we need to disable clipping.

    d = .5  # proportion of vertical to horizontal extent of the slanted line
    kwargs = dict(marker=[(-1, -d), (1, d)],
                  markersize=12,
                  linestyle="none",
                  color='k',
                  mec='k',
                  mew=1,
                  clip_on=False)
    ax1.plot([0, 1], [0, 0], transform=ax1.transAxes, **kwargs)
    ax2.plot([0, 1], [1, 1], transform=ax2.transAxes, **kwargs)

    xtick_labels = [
        "XCS\nno compact", "XCS\n" + r"$\rho=0.33$", "XCS\n" + r"$\rho=0.66$",
        "XCS\n" + r"$\rho=0.99$", "PPL-ST"
    ]
    plt.gca().set_xticklabels(xtick_labels)
    plt.ylabel(r"$\vert R_{\mathrm{BA}} \vert$")
    plt.savefig(f"./xcs_vs_pplst_bam_size_gnmc_plot_gs_12_sp_{_SLIP_PROB}.pdf",
                bbox_inches="tight")


def _calc_bam_size_xcs(pop, policy, nonterm_obss):
    rules_bam = []
    for obs in nonterm_obss:
        best_action = policy.select_action(obs)
        best_action_set = [
            rule for rule in pop
            if (rule.action == best_action and rule.does_match(obs))
        ]
        for rule in best_action_set:
            if rule not in rules_bam:
                rules_bam.append(rule)
    return len(rules_bam)


def _calc_bam_size_pplst(indiv, nonterm_obss):
    rules_bam = []
    for obs in nonterm_obss:
        (best_action, best_action_set) = \
            infer_action_and_action_set(indiv, obs)
        for rule in best_action_set:
            if rule not in rules_bam:
                rules_bam.append(rule)
    return len(rules_bam)


if __name__ == "__main__":
    main()

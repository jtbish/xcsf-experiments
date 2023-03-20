import glob
import pickle

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

_SLIP_PROB = 0.3
_EXPECTED_NUM_EXP_DIRS = 30
_GNMC_RHOS = [0.33, 0.66, 0.99]

_PARAMETRIC = True

_XCS_FINAL_NUM_GA_CALLS = 84000
_PPLST_FINAL_NUM_GENS = 250

_XCS_BASE_DIR = f"./frozen/stoca/gs_12_sp_{_SLIP_PROB}"
_PPLST_BASE_DIR = \
    f"/home/Staff/uqjbish3/pplst-experiments/fl/frozen_redux/stoca/gs_12_sp_{_SLIP_PROB}"

_OPTIMAL_PERF_ASSESS_RESS_PKL = "./FrozenLake_emp_opt_perfs_gamma_0.95_iodsb_frozen.pkl"

_YTICKS = {
    0.1: np.linspace(0.4, 1.0, num=13, endpoint=True),
    0.3: np.linspace(0.5, 1.0, num=11, endpoint=True)
}


def main():

    with open(_OPTIMAL_PERF_ASSESS_RESS_PKL, "rb") as fp:
        optimal_perf_assess_ress = pickle.load(fp)

    optimal_perf = optimal_perf_assess_ress[(12, _SLIP_PROB)].perf

    xcs_exp_dirs = glob.glob(f"{_XCS_BASE_DIR}/6*")
    pplst_exp_dirs = glob.glob(f"{_PPLST_BASE_DIR}/6*")

    assert len(xcs_exp_dirs) == _EXPECTED_NUM_EXP_DIRS
    assert len(pplst_exp_dirs) == _EXPECTED_NUM_EXP_DIRS

    # XCS first
    # rho == 0 means no compaction
    xcs_ftps = {rho: [] for rho in ([0] + _GNMC_RHOS)}

    for xcs_exp_dir in xcs_exp_dirs:

        # no compaction first
        with open(f"{xcs_exp_dir}/perf_history.pkl", "rb") as fp:
            perf_hist = pickle.load(fp)

        perf_assess_res = perf_hist[_XCS_FINAL_NUM_GA_CALLS]
        perf_pcnt = (perf_assess_res.perf / optimal_perf)
        xcs_ftps[0].append(perf_pcnt)

        # then compaction
        with open(f"{xcs_exp_dir}/gnmc/perf_assess_ress.pkl", "rb") as fp:
            gnmc_perf_assess_ress = pickle.load(fp)

        for rho in _GNMC_RHOS:
            perf_assess_res = gnmc_perf_assess_ress[rho]
            perf_pcnt = (perf_assess_res.perf / optimal_perf)
            xcs_ftps[rho].append(perf_pcnt)

    for v in xcs_ftps.values():
        assert len(v) == _EXPECTED_NUM_EXP_DIRS

    print("XCS")
    for (k, v) in xcs_ftps.items():
        print(k, v)
        print("\n")

    # PPLST
    pplst_ftps = []
    for pplst_exp_dir in pplst_exp_dirs:
        with open(f"{pplst_exp_dir}/best_indiv_test_perf_history.pkl",
                  "rb") as fp:
            perf_hist = pickle.load(fp)
        perf_assess_res = perf_hist[_PPLST_FINAL_NUM_GENS]
        perf_pcnt = (perf_assess_res.perf / optimal_perf)
        pplst_ftps.append(perf_pcnt)

    assert len(pplst_ftps) == _EXPECTED_NUM_EXP_DIRS

    print("PPL-ST")
    print(pplst_ftps)

    df = pd.DataFrame(columns=[
        "XCS no comp.", "XCS rho 0.33", "XCS rho 0.66", "XCS rho 0.99", "PPLST"
    ])
    df["XCS no comp."] = xcs_ftps[0]
    df["XCS rho 0.33"] = xcs_ftps[0.33]
    df["XCS rho 0.66"] = xcs_ftps[0.66]
    df["XCS rho 0.99"] = xcs_ftps[0.99]
    df["PPLST"] = pplst_ftps

    print(df)
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    # blue, brown, olive, red, purple
    colors = [colors[0], colors[5], colors[8], colors[3], colors[4]]
    sns.boxplot(data=df, palette=colors)
    xtick_labels = [
        "XCS\nno compact", "XCS\n" + r"$\rho=0.33$", "XCS\n" + r"$\rho=0.66$",
        "XCS\n" + r"$\rho=0.99$", "PPL-ST"
    ]
    plt.gca().set_xticklabels(xtick_labels)
    plt.gca().set_yticks(_YTICKS[_SLIP_PROB])
    #plt.title(f"(12, {_SLIP_PROB})")
    plt.ylabel("Testing performance")
    plt.ylim(top=1.0)
    plt.savefig(f"./xcs_vs_pplst_perf_gnmc_plot_gs_12_sp_{_SLIP_PROB}.pdf",
                bbox_inches="tight")


if __name__ == "__main__":
    main()

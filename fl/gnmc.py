import glob
import itertools
import pickle
from collections import OrderedDict
from pathlib import Path

import numpy as np
from rlenvs.environment import assess_perf
from rlenvs.frozen_lake import make_frozen_lake_env as make_fl

_SLIP_PROB = 0.3
_FL_SEED = 0

_GAMMA = 0.95

_EXPECTED_NUM_EXP_DIRS = 30
_XCSF_PKL_FILENAME = "xcsf_ga_calls_84000.pkl"

_GNMC_RHOS = (0.33, 0.66, 0.99)

_X_NOUGHT = 10


class GNMCCompactedPolicy:
    def __init__(self, compacted_pop, action_space):
        self._pop = compacted_pop
        self._action_space = action_space

    def select_action(self, obs):
        aug_obs = np.concatenate(([_X_NOUGHT], obs))
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

    base_dir = f"./frozen/stoca/gs_12_sp_{_SLIP_PROB}"
    exp_dirs = glob.glob(f"{base_dir}/6*")
    assert len(exp_dirs) == _EXPECTED_NUM_EXP_DIRS

    assert _SLIP_PROB > 0
    env = make_fl(grid_size=12,
                  slip_prob=_SLIP_PROB,
                  iod_strat="frozen_repeat",
                  seed=_FL_SEED)
    states = env.nonterminal_states
    actions = env.action_space
    num_perf_rollouts = (len(env.nonterminal_states) * 30)

    print(states)
    print(actions)

    for exp_dir in exp_dirs:

        save_path = Path(f"{exp_dir}/gnmc")
        save_path.mkdir(exist_ok=True)

        with open(f"{exp_dir}/{_XCSF_PKL_FILENAME}", "rb") as fp:
            xcsf = pickle.load(fp)
        pop = xcsf.pop

        with open(f"{exp_dir}/perf_history.pkl", "rb") as fp:
            perf_hist = pickle.load(fp)
        xcsf_final_perf_assess_res = perf_hist[84000]
        actual_xcsf_perf_assess_res = assess_perf(
            env=env, policy=xcsf, num_rollouts=num_perf_rollouts, gamma=_GAMMA)
        assert xcsf_final_perf_assess_res == actual_xcsf_perf_assess_res
        print(f"{exp_dir}: no compaction")
        print(actual_xcsf_perf_assess_res)

        perf_assess_ress = {rho: None for rho in _GNMC_RHOS}

        for rho in _GNMC_RHOS:
            print(f"{exp_dir}: {rho}")
            compacted = _gnmc_lambda_fit(pop, states, actions, rho)
            with open(save_path / f"pop_lambda_fit_rho_{rho}.pkl", "wb") as fp:
                pickle.dump(compacted, fp)

            compacted_policy = GNMCCompactedPolicy(compacted, env.action_space)
            perf_assess_res = assess_perf(env=env,
                                          policy=compacted_policy,
                                          num_rollouts=num_perf_rollouts,
                                          gamma=_GAMMA)
            print(perf_assess_res)
            perf_assess_ress[rho] = perf_assess_res

        with open(save_path / "perf_assess_ress.pkl", "wb") as fp:
            pickle.dump(perf_assess_ress, fp)

        print("\n")


def _gnmc_lambda_fit(pop, states, actions, rho):
    to_keep = []

    for (s, a) in itertools.product(states, actions):

        match_set = [clfr for clfr in pop if clfr.does_match(s)]
        assert len(match_set) > 0
        action_set = [clfr for clfr in match_set if clfr.action == a]
        assert len(action_set) > 0

        sorted_action_set = sorted(action_set,
                                   key=lambda clfr: clfr.fitness,
                                   reverse=True)

        total_mass = sum([clfr.fitness for clfr in sorted_action_set])
        target_mass = ((1 - rho) * total_mass)

        current_mass = 0
        idx = 0
        while current_mass < target_mass:
            clfr = sorted_action_set[idx]
            if clfr not in to_keep:
                to_keep.append(clfr)
            current_mass += clfr.fitness
            idx += 1

    return to_keep


if __name__ == "__main__":
    main()

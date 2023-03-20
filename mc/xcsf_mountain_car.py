#!/usr/bin/python3
import argparse
import copy
import glob
import logging
import os
import pickle
import shutil
import subprocess
import time
from pathlib import Path

import __main__
import numpy as np
from rlenvs.environment import assess_perf
from rlenvs.mountain_car import COVER_GRID_SIZE, NUM_BOTTOM_ZERO_VEL_SAMPLES
from rlenvs.mountain_car import make_mountain_car_env as make_mc
from xcsfrl.action_selection import (FixedEpsilonGreedy,
                                     LinearDecayEpsilonGreedy)
from xcsfrl.encoding import RealUnorderedBoundEncoding
from xcsfrl.prediction import (NormalisedLeastMeanSquaresPrediction,
                               RecursiveLeastSquaresPrediction)
from xcsfrl.xcsf import XCSF

_USE_OBS_NORMALISATION = True
_MC_SEED = 0
_NUM_CPUS = int(os.environ['SLURM_JOB_CPUS_PER_NODE'])
_Q_NPY_PATH = ""


class NullPredictionError(Exception):
    pass


class XCSFOnTheFlyCachedPolicy:
    """Cached policy for XCSF that is updated on the fly as observations come
    in."""
    def __init__(self, xcsf):
        self._xcsf = xcsf
        self._policy_cache = {}

        self._num_hits = 0
        self._num_misses = 0

    @property
    def num_hits(self):
        return self._num_hits

    @property
    def num_misses(self):
        return self._num_misses

    def select_action(self, obs):
        obs_hashable = tuple(obs)
        try:
            action = self._policy_cache[obs_hashable]
        except KeyError:
            action = self._xcsf.select_action(obs)
            self._policy_cache[obs_hashable] = action
            self._num_misses += 1
        else:
            self._num_hits += 1
        return action


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--mc-iod-strat-base", required=True)
    parser.add_argument("--xcsf-pred-strat", required=True)
    parser.add_argument("--xcsf-poly-order", type=int, required=True)
    parser.add_argument("--xcsf-seed", type=int, required=True)
    parser.add_argument("--xcsf-pop-size", type=int, required=True)
    parser.add_argument("--xcsf-beta-epsilon", type=float, required=True)
    parser.add_argument("--xcsf-beta", type=float, required=True)
    parser.add_argument("--xcsf-alpha", type=float, required=True)
    parser.add_argument("--xcsf-epsilon-nought", type=float, required=True)
    parser.add_argument("--xcsf-nu", type=float, required=True)
    parser.add_argument("--xcsf-gamma", type=float, required=True)
    parser.add_argument("--xcsf-theta-ga", type=int, required=True)
    parser.add_argument("--xcsf-tau", type=float, required=True)
    parser.add_argument("--xcsf-chi", type=float, required=True)
    parser.add_argument("--xcsf-upsilon", type=float, required=True)
    parser.add_argument("--xcsf-mu", type=float, required=True)
    parser.add_argument("--xcsf-theta-del", type=int, required=True)
    parser.add_argument("--xcsf-delta", type=float, required=True)
    parser.add_argument("--xcsf-theta-sub", type=int, required=True)
    parser.add_argument("--xcsf-r-nought", type=float, required=True)
    parser.add_argument("--xcsf-weight-i-min", type=float, required=True)
    parser.add_argument("--xcsf-weight-i-max", type=float, required=True)
    parser.add_argument("--xcsf-mu-i", type=float, required=True)
    parser.add_argument("--xcsf-epsilon-i", type=float, required=True)
    parser.add_argument("--xcsf-fitness-i", type=float, required=True)
    parser.add_argument("--xcsf-m-nought", type=float, required=True)
    parser.add_argument("--xcsf-x-nought", type=float, required=True)
    parser.add_argument("--xcsf-do-ga-subsumption", action="store_true")
    parser.add_argument("--xcsf-do-as-subsumption", action="store_true")
    parser.add_argument("--xcsf-delta-rls", type=float, default=None)
    parser.add_argument("--xcsf-tau-rls", type=int, default=None)
    parser.add_argument("--xcsf-lambda-rls", type=float, default=None)
    parser.add_argument("--xcsf-eta", type=float, default=None)
    parser.add_argument("--xcsf-p-explr", type=float, required=True)
    parser.add_argument("--monitor-freq-ga-calls", type=int, required=True)
    parser.add_argument("--monitor-num-ticks", type=int, required=True)
    return parser.parse_args()


def main(args):
    save_path = _setup_save_path(args.experiment_name)
    _setup_logging(save_path)
    logging.info(str(args))

    iod_strat_base = args.mc_iod_strat_base
    assert iod_strat_base in ("cover_grid", "bottom_zero_vel")

    iod_strat_train = iod_strat_base + "_uniform_rand"
    iod_strat_test = iod_strat_base + "_no_repeat"
    iod_strat_q_hat_mae = iod_strat_test

    train_env = _make_env(iod_strat_train)
    test_env = _make_env(iod_strat_test)
    # q_hat_mae_env = _make_env(iod_strat_q_hat_mae)

    if iod_strat_base == "cover_grid":
        num_test_rollouts = (COVER_GRID_SIZE**2)
    elif iod_strat_base == "bottom_zero_vel":
        num_test_rollouts = NUM_BOTTOM_ZERO_VEL_SAMPLES
    else:
        assert False

    logging.info(f"Training on iod strat: {iod_strat_train}")
    logging.info(f"Testing on iod strat: {iod_strat_test}")
    logging.info(f"Measuring Qhat MAE on iod strat: {iod_strat_q_hat_mae}")
    logging.info(f"Num test rollouts: {num_test_rollouts}")

    xcsf_hyperparams = {
        "seed": args.xcsf_seed,
        "N": args.xcsf_pop_size,
        "beta_epsilon": args.xcsf_beta_epsilon,
        "beta": args.xcsf_beta,
        "alpha": args.xcsf_alpha,
        "epsilon_nought": args.xcsf_epsilon_nought,
        "nu": args.xcsf_nu,
        "gamma": args.xcsf_gamma,
        "theta_ga": args.xcsf_theta_ga,
        "tau": args.xcsf_tau,
        "chi": args.xcsf_chi,
        "upsilon": args.xcsf_upsilon,
        "mu": args.xcsf_mu,
        "theta_del": args.xcsf_theta_del,
        "delta": args.xcsf_delta,
        "theta_sub": args.xcsf_theta_sub,
        "r_nought": args.xcsf_r_nought,
        "mu_I": args.xcsf_mu_i,
        "weight_I_min": args.xcsf_weight_i_min,
        "weight_I_max": args.xcsf_weight_i_max,
        "epsilon_I": args.xcsf_epsilon_i,
        "fitness_I": args.xcsf_fitness_i,
        "m_nought": args.xcsf_m_nought,
        "x_nought": args.xcsf_x_nought,
        "do_ga_subsumption": args.xcsf_do_ga_subsumption,
        "do_as_subsumption": args.xcsf_do_as_subsumption,
        "delta_rls": args.xcsf_delta_rls,
        "tau_rls": args.xcsf_tau_rls,
        "lambda_rls": args.xcsf_lambda_rls,
        "eta": args.xcsf_eta,
        "p_explr": args.xcsf_p_explr
    }
    logging.info(xcsf_hyperparams)
    encoding = RealUnorderedBoundEncoding(train_env.obs_space)

    if iod_strat_base == "cover_grid":
        action_selection_strat = FixedEpsilonGreedy(train_env.action_space)
    elif iod_strat_base == "bottom_zero_vel":
        total_num_ga_calls = (args.monitor_freq_ga_calls *
                              args.monitor_num_ticks)
        action_selection_strat = \
            LinearDecayEpsilonGreedy(train_env.action_space,
                                     total_num_ga_calls)
    else:
        assert False

    if args.xcsf_pred_strat == "rls":
        pred_strat = RecursiveLeastSquaresPrediction(args.xcsf_poly_order)
    elif args.xcsf_pred_strat == "nlms":
        pred_strat = NormalisedLeastMeanSquaresPrediction(args.xcsf_poly_order)
    else:
        assert False
    xcsf = XCSF(train_env, encoding, action_selection_strat, pred_strat,
                xcsf_hyperparams)

    #    q = _load_q_npy()
    q_hat_mae_history = {}
    perf_history = {}

    assert args.monitor_freq_ga_calls >= 1
    train_batch_size = args.monitor_freq_ga_calls
    assert args.monitor_num_ticks >= 1
    logging.info(f"Training for {train_batch_size} * "
                 f"{args.monitor_num_ticks} = "
                 f"{train_batch_size*args.monitor_num_ticks} num GA calls")
    ga_calls_done = 0
    for _ in range(args.monitor_num_ticks):
        xcsf.train_for_ga_calls(num_ga_calls=train_batch_size)
        ga_calls_done += train_batch_size
        _save_xcsf(save_path, copy.deepcopy(xcsf), ga_calls_done)

        pop = xcsf.pop
        logging.info(f"\nAfter {ga_calls_done} GA calls")
        logging.info(f"Num macros: {pop.num_macros}")
        logging.info(f"Num micros: {pop.num_micros}")
        ratio = pop.num_micros / pop.num_macros
        logging.info(f"Micro:macro ratio: {ratio:.4f}")

        errors = [clfr.error for clfr in pop]
        min_error = min(errors)
        avg_error = sum([clfr.error * clfr.numerosity
                         for clfr in pop]) / pop.num_micros
        median_error = np.median(errors)
        max_error = max(errors)
        logging.info(f"Min error: {min_error}")
        logging.info(f"Mean error: {avg_error}")
        logging.info(f"Median error: {median_error}")
        logging.info(f"Max error: {max_error}")

        numerosities = [clfr.numerosity for clfr in pop]
        logging.info(f"Min numerosity: {min(numerosities)}")
        logging.info(f"Mean numerosity: {np.mean(numerosities)}")
        logging.info(f"Median numerosity: {np.median(numerosities)}")
        logging.info(f"Max numerosity: {max(numerosities)}")

        generalities = [clfr.condition.generality for clfr in pop]
        logging.info(f"Min generality: {min(generalities)}")
        logging.info(f"Mean generality: {np.mean(generalities)}")
        logging.info(f"Median generality: {np.median(generalities)}")
        logging.info(f"Max generality: {max(generalities)}")

        logging.info(f"Pop ops history: {pop.ops_history}")

        #        try:
        #            q_hat_mae = _calc_q_hat_mae(q, xcsf, q_hat_mae_env)
        #            logging.info(f"q_hat mae = {q_hat_mae}")
        #        except NullPredictionError:
        #            q_hat_mae = None
        #            logging.info(f"Holes in q_hat")
        #        q_hat_mae_history[ga_calls_done] = q_hat_mae

        perf = _assess_perf(test_env, xcsf, num_test_rollouts, args.xcsf_gamma)
        perf_history[ga_calls_done] = perf
        logging.info(f"Perf = {perf}")

    assert ga_calls_done == \
        (args.monitor_freq_ga_calls * args.monitor_num_ticks)

    _save_histories(save_path, q_hat_mae_history, perf_history)
    _save_main_py_script(save_path)


#    _compress_xcsf_pkl_files(save_path, args.monitor_num_ticks)
#    _delete_uncompressed_xcsf_pkl_files(save_path)


def _setup_save_path(experiment_name):
    save_path = Path(args.experiment_name)
    save_path.mkdir(exist_ok=False)
    return save_path


def _setup_logging(save_path):
    logging.basicConfig(filename=save_path / "experiment.log",
                        format="%(levelname)s: %(message)s",
                        level=logging.DEBUG)


def _make_env(iod_strat):
    return make_mc(iod_strat=iod_strat,
                   normalise=_USE_OBS_NORMALISATION,
                   seed=_MC_SEED)


def _load_q_npy():
    return np.load(_Q_NPY_PATH)


def _calc_q_hat_mae(q, xcsf, env):
    """q is the optimal Q function for given (grid size, slip prob) combo.
    env is the environment (non-repeat) used for learning that states to
    compare are determined from"""
    q_shape = (env.grid_size, env.grid_size, len(env.action_space))
    assert q.shape == q_shape

    # iterate copy of env to get all starting states out of it
    env = copy.deepcopy(env)
    states_to_compare = []
    while True:
        try:
            states_to_compare.append(env.reset())
        except StopIteration:
            break
    assert len(states_to_compare) == env.si_size

    # frozen lake state is given by [x, y] pair, q arrs store values in (y, x)
    # fashion, i.e. y component is first dim (row), x component is second dim
    # (column); this is to make the array visually look like the grid.
    q_hat = np.full(q_shape, np.nan)
    for state in states_to_compare:
        [x, y] = state
        prediction_arr = xcsf.gen_prediction_arr(state)
        for a in env.action_space:
            prediction = prediction_arr[a]
            if prediction is not None:
                a_idx = list(env.action_space).index(a)
                q_idx = tuple([y, x] + [a_idx])
                q_hat[q_idx] = prediction

    # make a flat list of q_vals for all states to compare for both q and
    # q_hat, in order to check if q_hat has any null predictions for
    # states to compare
    q_compare_states_flat = []
    for state in states_to_compare:
        [x, y] = state
        idx = (y, x)
        q_compare_states_flat.extend(list(q[idx]))
    q_compare_states_flat = np.array(q_compare_states_flat)
    assert len(q_compare_states_flat) == \
        (len(states_to_compare) * len(env.action_space))

    q_hat_compare_states_flat = []
    for state in states_to_compare:
        [x, y] = state
        idx = (y, x)
        q_hat_compare_states_flat.extend(list(q_hat[idx]))
    q_hat_compare_states_flat = np.array(q_hat_compare_states_flat)
    assert len(q_hat_compare_states_flat) == \
        (len(states_to_compare) * len(env.action_space))

    contains_null_predictions = np.isnan(q_hat_compare_states_flat).any()
    if not contains_null_predictions:
        return np.mean(
            np.abs(q_compare_states_flat - q_hat_compare_states_flat))
    else:
        raise NullPredictionError


def _assess_perf(test_env, xcsf, num_test_rollouts, gamma):
    #    policy = XCSFOnTheFlyCachedPolicy(xcsf)
    #    perf_assess_res = assess_perf(env=test_env,
    #                                  policy=policy,
    #                                  num_rollouts=num_test_rollouts,
    #                                  gamma=gamma)
    #
    #    cache_num_queries = (policy.num_hits + policy.num_misses)
    #    cache_hit_rate_pcnt = (policy.num_hits / cache_num_queries * 100)
    #    logging.info(f"XCSF policy cache hit rate = {policy.num_hits} /"
    #                 f" {cache_num_queries} = {cache_hit_rate_pcnt:.2f} %")
    perf_assess_res = assess_perf(env=test_env,
                                  policy=xcsf,
                                  num_rollouts=num_test_rollouts,
                                  gamma=gamma)
    return perf_assess_res


def _save_xcsf(save_path, xcsf, ga_calls_done):
    with open(save_path / f"xcsf_ga_calls_{ga_calls_done}.pkl", "wb") as fp:
        pickle.dump(xcsf, fp)


def _save_histories(save_path, q_hat_mae_history, perf_history):
    with open(save_path / "q_hat_mae_history.pkl", "wb") as fp:
        pickle.dump(q_hat_mae_history, fp)
    with open(save_path / "perf_history.pkl", "wb") as fp:
        pickle.dump(perf_history, fp)


def _save_main_py_script(save_path):
    main_file_path = Path(__main__.__file__)
    shutil.copy(main_file_path, save_path)


def _compress_xcsf_pkl_files(save_path, monitor_num_ticks):
    xcsf_pkl_files = glob.glob(f"{save_path}/xcsf*.pkl")
    assert len(xcsf_pkl_files) == monitor_num_ticks
    os.environ["XZ_OPT"] = "-9e"
    subprocess.run(["tar", "-cJf", f"{save_path}/xcsfs.tar.xz"] +
                   xcsf_pkl_files,
                   check=True)


def _delete_uncompressed_xcsf_pkl_files(save_path):
    xcsf_pkl_files = glob.glob(f"{save_path}/xcsf*.pkl")
    for file_ in xcsf_pkl_files:
        os.remove(file_)


if __name__ == "__main__":
    start_time = time.time()
    args = parse_args()
    main(args)
    end_time = time.time()
    elpased = end_time - start_time
    logging.info(f"Runtime: {elpased:.3f}s with {_NUM_CPUS} cpus")

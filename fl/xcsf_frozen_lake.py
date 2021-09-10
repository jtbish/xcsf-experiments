#!/usr/bin/python3
import argparse
import logging
import os
import pickle
import shutil
import subprocess
import copy
import time
from pathlib import Path

import numpy as np
from xcsfrl.encoding import IntegerUnorderedBoundEncoding
from xcsfrl.xcsf import XCSF
from xcsfrl.action_selection import FixedEpsilonGreedy
from xcsfrl.prediction import (RecursiveLeastSquaresPrediction,
                               NormalisedLeastMeanSquaresPrediction)
from rlenvs.frozen_lake import make_frozen_lake_env as make_fl
from rlenvs.environment import assess_perf
import __main__

_FL_SEED = 0
_NUM_CPUS = int(os.environ['SLURM_JOB_CPUS_PER_NODE'])
_IOD_STRAT_TRAIN = "uniform_rand"
_IOD_STRAT_TEST_DETERMINISTIC = "frozen_no_repeat"
_IOD_STRAT_TEST_STOCHASTIC = "frozen_repeat"
_ROLLS_PER_SI_TEST_DETERMINISTIC = 1
_ROLLS_PER_SI_TEST_STOCHASTIC = 30
_SI_SIZES = {4: 11, 8: 53, 12: 114, 16: 203}


class NullPredictionError(Exception):
    pass


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--fl-grid-size", type=int, required=True)
    parser.add_argument("--fl-slip-prob", type=float, required=True)
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
    parser.add_argument("--xcsf-r-nought", type=int, required=True)
    parser.add_argument("--xcsf-weight-i-min", type=float, required=True)
    parser.add_argument("--xcsf-weight-i-max", type=float, required=True)
    parser.add_argument("--xcsf-mu-i", type=float, required=True)
    parser.add_argument("--xcsf-epsilon-i", type=float, required=True)
    parser.add_argument("--xcsf-fitness-i", type=float, required=True)
    parser.add_argument("--xcsf-m-nought", type=int, required=True)
    parser.add_argument("--xcsf-x-nought", type=float, required=True)
    parser.add_argument("--xcsf-do-ga-subsumption", action="store_true")
    parser.add_argument("--xcsf-do-as-subsumption", action="store_true")
    parser.add_argument("--xcsf-delta-rls", type=float, default=None)
    parser.add_argument("--xcsf-tau-rls", type=int, default=None)
    parser.add_argument("--xcsf-lambda-rls", type=float, default=None)
    parser.add_argument("--xcsf-eta", type=float, default=None)
    parser.add_argument("--xcsf-p-explr", type=float, required=True)
    parser.add_argument("--monitor-freq-episodes", type=int, required=True)
    parser.add_argument("--monitor-num-ticks", type=int, required=True)
    return parser.parse_args()


def main(args):
    save_path = _setup_save_path(args.experiment_name)
    _setup_logging(save_path)
    logging.info(str(args))

    train_env = _make_train_env(args)
    test_env = _make_test_env(args)
    si_size = _SI_SIZES[args.fl_grid_size]
    if args.fl_slip_prob == 0:
        num_test_rollouts = (si_size * _ROLLS_PER_SI_TEST_DETERMINISTIC)
        logging.info(f"Num test rollouts = {si_size} * "
                     f"{_ROLLS_PER_SI_TEST_DETERMINISTIC} = "
                     f"{num_test_rollouts}")
    elif 0 < args.fl_slip_prob < 1:
        num_test_rollouts = (si_size * _ROLLS_PER_SI_TEST_STOCHASTIC)
        logging.info(f"Num test rollouts = {si_size} * "
                     f"{_ROLLS_PER_SI_TEST_STOCHASTIC} = {num_test_rollouts}")
    else:
        assert False

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
    encoding = IntegerUnorderedBoundEncoding(train_env.obs_space)
    action_selection_strat = FixedEpsilonGreedy(train_env.action_space)
    if args.xcsf_pred_strat == "rls":
        pred_strat = RecursiveLeastSquaresPrediction(args.xcsf_poly_order)
    elif args.xcsf_pred_strat == "nlms":
        pred_strat = NormalisedLeastMeanSquaresPrediction(args.xcsf_poly_order)
    else:
        assert False
    xcsf = XCSF(train_env, encoding, action_selection_strat, pred_strat,
                xcsf_hyperparams)

    q = _load_q_npy(args)
    q_hat_mae_history = {}
    perf_history = {}

    assert args.monitor_freq_episodes >= 1
    episode_batch_size = args.monitor_freq_episodes
    assert args.monitor_num_ticks >= 1
    episodes_done = 0
    for _ in range(args.monitor_num_ticks):
        xcsf.train_for_episodes(num_episodes=episode_batch_size)
        episodes_done += episode_batch_size
        _save_xcsf(save_path, copy.deepcopy(xcsf), episodes_done)

        pop = xcsf.pop
        logging.info(f"\nAfter {episodes_done} episodes")
        logging.info(f"Num macros: {pop.num_macros}")
        logging.info(f"Num micros: {pop.num_micros}")
        ratio = pop.num_micros / pop.num_macros
        logging.info(f"Micro:macro ratio: {ratio:.4f}")

        errors = [clfr.error for clfr in pop]
        min_error = min(errors)
        avg_error = sum([clfr.error * clfr.numerosity for clfr in
                        pop]) / pop.num_micros
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

        try:
            q_hat_mae = _calc_q_hat_mae(q, test_env, xcsf)
            logging.info(f"q_hat mae = {q_hat_mae}")
        except NullPredictionError:
            q_hat_mae = None
            logging.info(f"Holes in q_hat")
        q_hat_mae_history[episodes_done] = q_hat_mae

        perf = _assess_perf(test_env, xcsf, num_test_rollouts, args.xcsf_gamma)
        perf_history[episodes_done] = perf
        logging.info(f"Perf = {perf}")

    assert episodes_done == \
        (args.monitor_freq_episodes * args.monitor_num_ticks)

    _save_histories(save_path, q_hat_mae_history, perf_history)
    _save_python_env_info(save_path)
    _save_main_py_script(save_path)


def _setup_save_path(experiment_name):
    save_path = Path(args.experiment_name)
    save_path.mkdir(exist_ok=False)
    return save_path


def _setup_logging(save_path):
    logging.basicConfig(filename=save_path / "experiment.log",
                        format="%(levelname)s: %(message)s",
                        level=logging.DEBUG)


def _make_train_env(args):
    return make_fl(grid_size=args.fl_grid_size,
                   slip_prob=args.fl_slip_prob,
                   iod_strat=_IOD_STRAT_TRAIN,
                   seed=_FL_SEED)


def _make_test_env(args):
    slip_prob = args.fl_slip_prob
    if slip_prob == 0:
        iod_strat = _IOD_STRAT_TEST_DETERMINISTIC
    elif 0 < slip_prob < 1:
        iod_strat = _IOD_STRAT_TEST_STOCHASTIC
    else:
        assert False
    return make_fl(grid_size=args.fl_grid_size,
                   slip_prob=args.fl_slip_prob,
                   iod_strat=iod_strat,
                   seed=_FL_SEED)


def _load_q_npy(args):
    gs = args.fl_grid_size
    gm = args.xcsf_gamma
    sp = args.fl_slip_prob
    q_npy_path = \
        f"q_npy/FrozenLake{gs}x{gs}-v0_gamma_{gm}_slip_prob_{sp:.2f}_Qstar.npy"
    return np.load(q_npy_path)


def _calc_q_hat_mae(q, env, xcsf):
    shape = (env.grid_size, env.grid_size, len(env.action_space))
    assert q.shape == shape

    # frozen lake state is given by [x, y] pair, q arrs store values in (y, x)
    # fashion, i.e. y component is first dim (row), x component is second dim
    # (column); this is to make the array visually look like the grid.
    q_hat = np.full(shape, np.nan)
    for state in env.nonterminal_states:
        [x, y] = state
        prediction_arr = xcsf.gen_prediction_arr(state)
        for a in env.action_space:
            prediction = prediction_arr[a]
            if prediction is not None:
                a_idx = list(env.action_space).index(a)
                q_idx = tuple([y, x] + [a_idx])
                q_hat[q_idx] = prediction

    # make a flat list of q_vals for all nonterminal states for both q and
    # q_hat, in order to check if q_hat has any null predictions for
    # nonterminal states
    q_nonterm_states_flat = []
    for state in env.nonterminal_states:
        [x, y] = state
        idx = (y, x)
        q_nonterm_states_flat.extend(list(q[idx]))
    q_nonterm_states_flat = np.array(q_nonterm_states_flat)

    q_hat_nonterm_states_flat = []
    for state in env.nonterminal_states:
        [x, y] = state
        idx = (y, x)
        q_hat_nonterm_states_flat.extend(list(q_hat[idx]))
    q_hat_nonterm_states_flat = np.array(q_hat_nonterm_states_flat)

    contains_null_predictions = np.isnan(q_hat_nonterm_states_flat).any()
    if not contains_null_predictions:
        return np.mean(np.abs(q_nonterm_states_flat -
                       q_hat_nonterm_states_flat))
    else:
        raise NullPredictionError


def _assess_perf(test_env, xcsf, num_test_rollouts, gamma):
    return assess_perf(env=test_env,
                       policy=xcsf,
                       num_rollouts=num_test_rollouts,
                       gamma=gamma)


def _save_xcsf(save_path, xcsf, episodes_done):
    with open(save_path / f"xcsf_eps_{episodes_done}.pkl", "wb") as fp:
        pickle.dump(xcsf, fp)


def _save_histories(save_path, q_hat_mae_history, perf_history):
    with open(save_path / "q_hat_mae_history.pkl", "wb") as fp:
        pickle.dump(q_hat_mae_history, fp)
    with open(save_path / "perf_history.pkl", "wb") as fp:
        pickle.dump(perf_history, fp)


def _save_python_env_info(save_path):
    result = subprocess.run(["pip3", "freeze"],
                            stdout=subprocess.PIPE,
                            stderr=subprocess.DEVNULL)
    return_val = result.stdout.decode("utf-8")
    with open(save_path / "python_env_info.txt", "w") as fp:
        fp.write(str(return_val))


def _save_main_py_script(save_path):
    main_file_path = Path(__main__.__file__)
    shutil.copy(main_file_path, save_path)


if __name__ == "__main__":
    start_time = time.time()
    args = parse_args()
    main(args)
    end_time = time.time()
    elpased = end_time - start_time
    logging.info(f"Runtime: {elpased:.3f}s with {_NUM_CPUS} cpus")

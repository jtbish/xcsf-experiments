#!/usr/bin/python3
import argparse
import copy
import logging
import os
import pickle
import shutil
import subprocess
import time
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
from xcsfrl.encoding import RealUnorderedBoundEncoding
from xcsfrl.xcsf import XCSF
from xcsfrl.action_selection import FixedEpsilonGreedy
from xcsfrl.prediction import LinearPrediction, QuadraticPrediction
from rlenvs.mountain_car import make_mountain_car_env as make_mc
from rlenvs.environment import assess_perf
from util import compute_dims_ref_points, calc_discrete_states

_MC_SEED = 0
_NORMALISE_ENV = True

_NUM_CPUS = int(os.environ['SLURM_JOB_CPUS_PER_NODE'])
# no noise in transition function so zero out beta_epsilon and mu_I
_XCSF_BETA_EPSILON = 0.0
_XCSF_MU_I = 0.0

_NUM_SOLN_BINS_PER_DIM = 100


class NullPredictionError(Exception):
    pass


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--xcsf-seed", type=int, required=True)
    parser.add_argument("--xcsf-pred-strat", required=True)
    parser.add_argument("--xcsf-pop-size", type=int, required=True)
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
    parser.add_argument("--xcsf-epsilon-i", type=float, required=True)
    parser.add_argument("--xcsf-fitness-i", type=float, required=True)
    parser.add_argument("--xcsf-m-nought", type=float, required=True)
    parser.add_argument("--xcsf-x-nought", type=float, required=True)
    parser.add_argument("--xcsf-do-ga-subsumption", action="store_true")
    parser.add_argument("--xcsf-do-as-subsumption", action="store_true")
    parser.add_argument("--xcsf-delta-rls", type=float, required=True)
    parser.add_argument("--xcsf-tau-rls", type=int, required=True)
    parser.add_argument("--xcsf-p-explr", type=float, required=True)
    parser.add_argument("--num-train-steps", type=int, required=True)
    parser.add_argument("--num-test-rollouts", type=int, required=True)
    parser.add_argument("--monitor-freq", type=int, required=True)
    return parser.parse_args()


def main(args):
    save_path = _setup_save_path(args.experiment_name)
    _setup_logging(save_path)
    logging.info(str(args))

    train_env = _make_train_env(args)
    test_env = _make_test_env(args)

    xcsf_hyperparams = {
        "seed": args.xcsf_seed,
        "N": args.xcsf_pop_size,
        "beta_epsilon": _XCSF_BETA_EPSILON,
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
        "mu_I": _XCSF_MU_I,
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
        "p_explr": args.xcsf_p_explr
    }
    logging.info(xcsf_hyperparams)
    encoding = RealUnorderedBoundEncoding(train_env.obs_space)
    action_selection_strat = FixedEpsilonGreedy(train_env.action_space)
    if args.xcsf_pred_strat == "linear":
        pred_strat = LinearPrediction()
    elif args.xcsf_pred_strat == "quadratic":
        pred_strat = QuadraticPrediction()
    else:
        assert False
    xcsf = XCSF(train_env, encoding, action_selection_strat, pred_strat,
                xcsf_hyperparams)

    v = np.load("mc_v_num_bins_100.npy")
    q = np.load("mc_q_num_bins_100.npy")

    assert args.num_train_steps % args.monitor_freq == 0
    num_batches = (args.num_train_steps // args.monitor_freq)
    batch_size = args.monitor_freq
    xcsf_history = {}
    steps_done = 0
    for _ in range(num_batches):
        xcsf.train(num_steps=batch_size)
        steps_done += batch_size
        xcsf_history[steps_done] = copy.deepcopy(xcsf)

        pop = xcsf.pop
        pop_size_macros = len(pop)
        pop_size_micros = sum([clfr.numerosity for clfr in pop])
        avg_error = sum([clfr.error * clfr.numerosity for clfr in
                        pop])/pop_size_micros
        logging.info(f"\nAfter {steps_done} time steps")
        logging.info(f"Num macros: {pop_size_macros}")
        logging.info(f"Num micros: {pop_size_micros}")
        logging.info(f"Avg error: {avg_error}")
        logging.info(f"Pop ops history: {xcsf.pop_ops_history}")
        try:
            q_hat_mae = _calc_q_hat_mae(v, q, test_env, xcsf, steps_done,
                                        save_path)
            logging.info(f"q_hat mae = {q_hat_mae}")
        except NullPredictionError:
            logging.info(f"Holes in q_hat")
        perf = _assess_perf(test_env, xcsf, args)
        logging.info(f"Perf = {perf}")
    assert steps_done == args.num_train_steps

    _save_data(save_path, xcsf_history)
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
    return make_mc(iod_strat="uniform",
                   normalise=_NORMALISE_ENV,
                   seed=_MC_SEED)


def _make_test_env(args):
    return make_mc(iod_strat="bottom_zero_vel",
                   normalise=_NORMALISE_ENV,
                   seed=_MC_SEED)


def _assess_perf(test_env, xcsf, args):
    expected_return = assess_perf(env=test_env,
                                  policy=xcsf,
                                  num_rollouts=args.num_test_rollouts,
                                  gamma=args.xcsf_gamma)
    if expected_return is not None:
        return expected_return
    else:
        logging.info("Failed perf eval")
        return test_env.perf_lower_bound


def _calc_q_hat_mae(v, q, env, xcsf, steps_done, save_path):
    v_shape = (_NUM_SOLN_BINS_PER_DIM, _NUM_SOLN_BINS_PER_DIM)
    q_shape = (_NUM_SOLN_BINS_PER_DIM, _NUM_SOLN_BINS_PER_DIM,
               len(env.action_space))
    assert v.shape == v_shape
    assert q.shape == q_shape
    num_ref_points_per_dim = _NUM_SOLN_BINS_PER_DIM + 1
    dims_ref_points = compute_dims_ref_points(env.obs_space,
                                              num_ref_points_per_dim)
    num_dims = len(env.obs_space)
    discrete_states = calc_discrete_states(_NUM_SOLN_BINS_PER_DIM, num_dims,
                                           dims_ref_points)

    v_hat = np.full(v_shape, np.nan)
    q_hat = np.full(q_shape, np.nan)
    for (idx_combo, repr_real_state) in discrete_states.items():
        prediction_arr = xcsf.gen_prediction_arr(repr_real_state)
        # fill q_hat
        for a in env.action_space:
            prediction = prediction_arr[a]
            if prediction is not None:
                a_idx = list(env.action_space).index(a)
                q_idx = tuple(list(idx_combo) + [a_idx])
                q_hat[q_idx] = prediction
        # fill v_hat
        non_null_predictions = \
            [p for p in prediction_arr.values() if p is not None]
        if len(non_null_predictions) > 0:
            v_hat[idx_combo] = max(non_null_predictions)

    contains_null_predictions = np.isnan(q_hat).any()
    if not contains_null_predictions:
        assert not (np.isnan(v_hat).any())
        v_diff = (v_hat - v)
        plt.figure()
        min_ = np.min(v_diff)
        max_ = np.max(v_diff)
        largest_abs = max(abs(min_), abs(max_))
        # plot v_diff transpose to make pos cols, vel rows
        # make colours symmetrical around 0
        plt.imshow(v_diff.T, cmap="bwr", vmin=-largest_abs, vmax=largest_abs)
        plt.colorbar()
        plt.savefig(f"{save_path}/v_diff_{steps_done}_steps.png")
        mean_abs_error = np.mean(np.abs(q - q_hat))
        return mean_abs_error
    else:
        raise NullPredictionError


def _save_data(save_path, xcsf_history):
    with open(save_path / "xcsf_history.pkl", "wb") as fp:
        pickle.dump(xcsf_history, fp)


def _save_python_env_info(save_path):
    result = subprocess.run(["pip3", "freeze"],
                            stdout=subprocess.PIPE,
                            stderr=subprocess.DEVNULL)
    return_val = result.stdout.decode("utf-8")
    with open(save_path / "python_env_info.txt", "w") as fp:
        fp.write(str(return_val))


def _save_main_py_script(save_path):
    this_file_path = Path(__file__).absolute()
    shutil.copy(this_file_path, save_path)


if __name__ == "__main__":
    start_time = time.time()
    args = parse_args()
    main(args)
    end_time = time.time()
    elpased = end_time - start_time
    logging.info(f"Runtime: {elpased:.3f}s with {_NUM_CPUS} cpus")

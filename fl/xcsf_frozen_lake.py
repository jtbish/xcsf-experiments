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
from xcsfrl.prediction import LinearPrediction, QuadraticPrediction
from rlenvs.frozen_lake import make_frozen_lake_env as make_fl
from rlenvs.environment import assess_perf
import __main__

_FL_SEED = 0
_NUM_CPUS = int(os.environ['SLURM_JOB_CPUS_PER_NODE'])


class NullPredictionError(Exception):
    pass


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--fl-grid-size", type=int, required=True)
    parser.add_argument("--fl-slip-prob", type=float, required=True)
    parser.add_argument("--fl-tl-mult", type=int, required=True)
    parser.add_argument("--xcsf-pred-strat", required=True)
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
    parser.add_argument("--xcsf-delta-rls", type=float, required=True)
    parser.add_argument("--xcsf-tau-rls", type=int, required=True)
    parser.add_argument("--xcsf-p-explr", type=float, required=True)
    parser.add_argument("--num-test-rollouts", type=int, required=True)
    parser.add_argument("--monitor-steps", required=True)
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
        "p_explr": args.xcsf_p_explr
    }
    logging.info(xcsf_hyperparams)
    encoding = IntegerUnorderedBoundEncoding(train_env.obs_space)
    action_selection_strat = FixedEpsilonGreedy(train_env.action_space)
    if args.xcsf_pred_strat == "linear":
        pred_strat = LinearPrediction()
    elif args.xcsf_pred_strat == "quadratic":
        pred_strat = QuadraticPrediction()
    else:
        assert False
    xcsf = XCSF(train_env, encoding, action_selection_strat, pred_strat,
                xcsf_hyperparams)

    q = _load_q_npy(args)
    monitor_steps = _parse_monitor_steps_str(args.monitor_steps)
    training_step_sizes = _calc_training_step_sizes(monitor_steps)
    logging.info(f"Monitor steps: {monitor_steps}")
    logging.info(f"Training step sizes: {training_step_sizes}")

    xcsf_history = {}
    q_hat_mae_history = {}
    perf_history = {}
    steps_done = 0
    for training_step_size in training_step_sizes:
        xcsf.train(num_steps=training_step_size)
        steps_done += training_step_size
        xcsf_history[steps_done] = copy.deepcopy(xcsf)

        pop = xcsf.pop
        avg_error = sum([clfr.error * clfr.numerosity for clfr in
                        pop]) / pop.num_micros
        logging.info(f"\nAfter {steps_done} time steps")
        logging.info(f"Num macros: {pop.num_macros}")
        logging.info(f"Num micros: {pop.num_micros}")
        actual_micros = sum([clfr.numerosity for clfr in pop])
        logging.info(f"Actual num micros via numer loop: {actual_micros}")
        logging.info(f"Avg error: {avg_error}")
        logging.info(f"Pop ops history: {pop.ops_history}")

        try:
            q_hat_mae = _calc_q_hat_mae(q, test_env, xcsf)
            logging.info(f"q_hat mae = {q_hat_mae}")
        except NullPredictionError:
            q_hat_mae = None
            logging.info(f"Holes in q_hat")
        q_hat_mae_history[steps_done] = q_hat_mae

        perf = _assess_perf(test_env, xcsf, args)
        perf_history[steps_done] = perf
        logging.info(f"Perf = {perf}")

    last_monitor_step = monitor_steps[-1]
    assert steps_done == last_monitor_step

    _save_data(save_path, xcsf_history, q_hat_mae_history, perf_history)
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
    # training always use uniform rand initial obss
    return make_fl(grid_size=args.fl_grid_size,
                   slip_prob=args.fl_slip_prob,
                   iod_strat="uniform_rand",
                   time_limit_mult=args.fl_tl_mult,
                   seed=_FL_SEED)


def _make_test_env(args):
    # testing uses non-repeated/repeated frozen obss dependent on slip prob
    slip_prob = args.fl_slip_prob
    if slip_prob == 0:
        iod_strat = "frozen_no_repeat"
    elif 0 < slip_prob < 1:
        iod_strat = "frozen_repeat"
    else:
        assert False
    return make_fl(grid_size=args.fl_grid_size,
                   slip_prob=args.fl_slip_prob,
                   iod_strat=iod_strat,
                   time_limit_mult=args.fl_tl_mult,
                   seed=_FL_SEED)


def _load_q_npy(args):
    gs = args.fl_grid_size
    gm = args.xcsf_gamma
    sp = args.fl_slip_prob
    q_npy_path = \
        f"q_npy/FrozenLake{gs}x{gs}-v0_gamma_{gm}_slip_prob_{sp:.2f}_Qstar.npy"
    return np.load(q_npy_path)


def _parse_monitor_steps_str(monitor_steps_str):
    """Monitor steps are comma sepd list of ints that specify time steps at
    which to measure error & perf of model."""
    monitor_steps = [int(elem) for elem in monitor_steps_str.split(",")]
    # check all steps positive and increasing
    for idx in range(0, len(monitor_steps)):
        assert monitor_steps[idx] > 0
        if idx != 0:
            assert monitor_steps[idx] > monitor_steps[idx-1]
    return monitor_steps


def _calc_training_step_sizes(monitor_steps):
    # monitor steps are *absolute* training step checkpoints, so need to calc
    # the *relative* chunk sizes between them, first one taken literally since
    # relative to zero.
    training_step_sizes = [monitor_steps[0]]
    for idx in range(1, len(monitor_steps)):
        delta = (monitor_steps[idx] - monitor_steps[idx-1])
        assert delta > 0
        training_step_sizes.append(delta)
    assert len(training_step_sizes) == len(monitor_steps)
    return training_step_sizes


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


def _assess_perf(test_env, xcsf, args):
    return assess_perf(env=test_env,
                       policy=xcsf,
                       num_rollouts=args.num_test_rollouts,
                       gamma=args.xcsf_gamma)


def _save_data(save_path, xcsf_history, q_hat_mae_history, perf_history):
    with open(save_path / "xcsf_history.pkl", "wb") as fp:
        pickle.dump(xcsf_history, fp)
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

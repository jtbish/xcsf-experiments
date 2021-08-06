#!/bin/bash
fl_grid_size=8
fl_slip_prob=0
xcsf_poly_order=1
xcsf_pop_size=5000
xcsf_beta_epsilon=0.0
xcsf_beta=0.1
xcsf_alpha=0.1
xcsf_epsilon_nought=0.01
xcsf_nu=5
xcsf_gamma=0.95
xcsf_theta_ga=50
xcsf_tau=0.5
xcsf_chi=0.8
xcsf_upsilon=0.5
xcsf_mu=0.05
xcsf_theta_del=50
xcsf_delta=0.1
xcsf_theta_sub=50
xcsf_r_nought=$(( $fl_grid_size / 2 ))
xcsf_weight_i_min=0
xcsf_weight_i_max=0
xcsf_mu_i=0.001
xcsf_epsilon_i=0.001
xcsf_fitness_i=0.001
xcsf_m_nought=$(( $fl_grid_size / 4 ))
xcsf_x_nought=10
xcsf_p_explr=0.5

xcsf_pred_strat="nlms"
xcsf_delta_rls=10
xcsf_tau_rls=0
xcsf_lambda_rls=0.99
xcsf_eta=0.1

# si sizes:
# 4x4 num frozen = 11
# 8x8 num frozen = 53
# 12x12 num frozen = 114
# 16x16 num frozen = 203
si_size=53
tests_per_si=1
num_test_rollouts=$(( $si_size * $tests_per_si ))
monitor_steps=$(python3 -c 'print(",".join([str(i*10000) for i in range(1,25+1)]))')

for xcsf_seed in {0..0}; do
   echo sbatch xcsf_frozen_lake.sh \
        "$fl_grid_size" \
        "$fl_slip_prob" \
        "$xcsf_pred_strat" \
        "$xcsf_poly_order" \
        "$xcsf_seed" \
        "$xcsf_pop_size" \
        "$xcsf_beta_epsilon" \
        "$xcsf_beta" \
        "$xcsf_alpha" \
        "$xcsf_epsilon_nought" \
        "$xcsf_nu" \
        "$xcsf_gamma" \
        "$xcsf_theta_ga" \
        "$xcsf_tau" \
        "$xcsf_chi" \
        "$xcsf_upsilon" \
        "$xcsf_mu" \
        "$xcsf_theta_del" \
        "$xcsf_delta" \
        "$xcsf_theta_sub" \
        "$xcsf_r_nought" \
        "$xcsf_weight_i_min" \
        "$xcsf_weight_i_max" \
        "$xcsf_mu_i" \
        "$xcsf_epsilon_i" \
        "$xcsf_fitness_i" \
        "$xcsf_m_nought" \
        "$xcsf_x_nought" \
        "$xcsf_delta_rls" \
        "$xcsf_tau_rls" \
        "$xcsf_lambda_rls" \
        "$xcsf_eta" \
        "$xcsf_p_explr" \
        "$num_test_rollouts" \
        "$monitor_steps"
done

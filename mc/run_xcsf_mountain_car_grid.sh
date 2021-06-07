#!/bin/bash
#xcsf_pred_strat="quadratic"
xcsf_pop_size=7500
xcsf_beta=0.1
xcsf_alpha=0.1
#xcsf_epsilon_nought=0.25
xcsf_nu=5
xcsf_gamma=1.0
xcsf_theta_ga=50
xcsf_tau=0.4
xcsf_chi=0.8
xcsf_upsilon=0.5
xcsf_mu=0.05
xcsf_theta_del=50
xcsf_delta=0.1
xcsf_theta_sub=50
xcsf_r_nought=0.2
xcsf_weight_i_min=0
xcsf_weight_i_max=0
xcsf_epsilon_i=0.001
xcsf_fitness_i=0.001
xcsf_m_nought=0.1
xcsf_x_nought=1.0
#xcsf_delta_rls=1000
#xcsf_tau_rls=0
xcsf_p_explr=0.1
num_train_steps=200000
num_test_rollouts=10
monitor_freq=5000

xcsf_pred_strats=( "linear" "quadratic" )
xcsf_epsilon_noughts=( 0.1 0.25 )
#xcsf_delta_rlss=( 1 1 10 10 10 100 100 100 1000 1000 )
#xcsf_tau_rlss=( 0 100 0 100 1000 0 1000 10000 0 10000 )
xcsf_delta_rlss=( 10 )
xcsf_tau_rlss=( 10000 )

for xcsf_pred_strat in "${xcsf_pred_strats[@]}"; do
    for xcsf_epsilon_nought in "${xcsf_epsilon_noughts[@]}"; do
        for (( i=0; i<"${#xcsf_delta_rlss[@]}"; i++ )); do
            xcsf_delta_rls="${xcsf_delta_rlss[$i]}"
            xcsf_tau_rls="${xcsf_tau_rlss[$i]}"
            for xcsf_seed in {0..4}; do
               echo sbatch xcsf_mountain_car.sh \
                    "$xcsf_seed" \
                    "$xcsf_pred_strat" \
                    "$xcsf_pop_size" \
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
                    "$xcsf_epsilon_i" \
                    "$xcsf_fitness_i" \
                    "$xcsf_m_nought" \
                    "$xcsf_x_nought" \
                    "$xcsf_delta_rls" \
                    "$xcsf_tau_rls" \
                    "$xcsf_p_explr" \
                    "$num_train_steps" \
                    "$num_test_rollouts" \
                    "$monitor_freq"
            done
        done
    done
done

#!/bin/bash
mc_iod_strat_base="bottom_zero_vel"

xcsf_poly_order=1
declare -A xcsf_pop_sizes=( ["bottom_zero_vel"]=2000 ["cover_grid"]=3000 )
xcsf_pop_size="${xcsf_pop_sizes[$mc_iod_strat_base]}"
xcsf_beta_epsilon=0
xcsf_beta=0.1
xcsf_alpha=0.1
xcsf_epsilon_nought=0.1
xcsf_nu=5
xcsf_gamma=1
xcsf_theta_ga=50
xcsf_tau=0.5
xcsf_chi=0.8
xcsf_upsilon=0.5
xcsf_mu=0.04
xcsf_theta_del=50
xcsf_delta=0.1
xcsf_theta_sub=50
xcsf_r_nought=0.25
xcsf_weight_i_min=0
xcsf_weight_i_max=0
xcsf_mu_i=0.001
xcsf_epsilon_i=0.001
xcsf_fitness_i=0.001
xcsf_m_nought=0.1
xcsf_x_nought=1
xcsf_p_explr=0.5

xcsf_pred_strat="rls"
xcsf_delta_rls=10
# if tau_rls > 0, lambda_rls should == 1
# if lambda_rls < 1, tau_rls should == 0
# i.e. use one or the other
xcsf_tau_rls=0
#xcsf_rls_mem_len=$(( $xcsf_theta_ga * 10 ))
#xcsf_lambda_rls=$(bc -l <<< "1 - (1 / $xcsf_rls_mem_len)")
xcsf_lambda_rls=0.999
# leftover for nlms pred strat: ignore
xcsf_eta=0.1

declare -A monitor_freq_ga_callss=( ["bottom_zero_vel"]=150 ["cover_grid"]=225 )
monitor_freq_ga_calls="${monitor_freq_ga_callss[$mc_iod_strat_base]}"
monitor_num_ticks=250

for xcsf_seed in {0..29}; do
   echo sbatch xcsf_mountain_car.sh \
        "$mc_iod_strat_base" \
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
        "$monitor_freq_ga_calls" \
        "$monitor_num_ticks"
done

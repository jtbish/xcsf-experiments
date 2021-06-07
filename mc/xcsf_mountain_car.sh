#!/bin/bash
#SBATCH --partition=batch
#SBATCH --cpus-per-task=1

source ~/virtualenvs/xcsf/bin/activate
python3 xcsf_mountain_car.py \
    --experiment-name="$SLURM_JOB_ID" \
    --xcsf-seed="$1" \
    --xcsf-pred-strat="$2" \
    --xcsf-pop-size="$3" \
    --xcsf-beta="$4" \
    --xcsf-alpha="$5" \
    --xcsf-epsilon-nought="$6" \
    --xcsf-nu="$7" \
    --xcsf-gamma="$8" \
    --xcsf-theta-ga="$9" \
    --xcsf-tau="${10}" \
    --xcsf-chi="${11}" \
    --xcsf-upsilon="${12}" \
    --xcsf-mu="${13}" \
    --xcsf-theta-del="${14}" \
    --xcsf-delta="${15}" \
    --xcsf-theta-sub="${16}" \
    --xcsf-r-nought="${17}" \
    --xcsf-weight-i-min="${18}" \
    --xcsf-weight-i-max="${19}" \
    --xcsf-epsilon-i="${20}" \
    --xcsf-fitness-i="${21}" \
    --xcsf-m-nought="${22}" \
    --xcsf-x-nought="${23}" \
    --xcsf-do-ga-subsumption \
    --xcsf-delta-rls="${24}" \
    --xcsf-tau-rls="${25}" \
    --xcsf-p-explr="${26}" \
    --num-train-steps="${27}" \
    --num-test-rollouts="${28}" \
    --monitor-freq="${29}"
mv "slurm-${SLURM_JOB_ID}.out" "${SLURM_JOB_ID}/"

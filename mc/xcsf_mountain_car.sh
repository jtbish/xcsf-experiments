#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1

source ~/virtualenvs/lcs/bin/activate
python3 xcsf_mountain_car.py \
    --experiment-name="$SLURM_JOB_ID" \
    --mc-iod-strat-base="$1" \
    --xcsf-pred-strat="$2" \
    --xcsf-poly-order="$3" \
    --xcsf-seed="$4" \
    --xcsf-pop-size="$5" \
    --xcsf-beta-epsilon="$6" \
    --xcsf-beta="$7" \
    --xcsf-alpha="$8" \
    --xcsf-epsilon-nought="$9" \
    --xcsf-nu="${10}" \
    --xcsf-gamma="${11}" \
    --xcsf-theta-ga="${12}" \
    --xcsf-tau="${13}" \
    --xcsf-chi="${14}" \
    --xcsf-upsilon="${15}" \
    --xcsf-mu="${16}" \
    --xcsf-theta-del="${17}" \
    --xcsf-delta="${18}" \
    --xcsf-theta-sub="${19}" \
    --xcsf-r-nought="${20}" \
    --xcsf-weight-i-min="${21}" \
    --xcsf-weight-i-max="${22}" \
    --xcsf-mu-i="${23}" \
    --xcsf-epsilon-i="${24}" \
    --xcsf-fitness-i="${25}" \
    --xcsf-m-nought="${26}" \
    --xcsf-x-nought="${27}" \
    --xcsf-do-ga-subsumption \
    --xcsf-delta-rls="${28}" \
    --xcsf-tau-rls="${29}" \
    --xcsf-lambda-rls="${30}" \
    --xcsf-eta="${31}" \
    --xcsf-p-explr="${32}"\
    --monitor-freq-ga-calls="${33}" \
    --monitor-num-ticks="${34}"
mv "slurm-${SLURM_JOB_ID}.out" "${SLURM_JOB_ID}/"
#mv "${SLURM_JOB_ID}.prof" "${SLURM_JOB_ID}/"

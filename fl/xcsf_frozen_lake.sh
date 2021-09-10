#!/bin/bash
#SBATCH --partition=batch
#SBATCH --cpus-per-task=1

source ~/virtualenvs/xcsf/bin/activate
python3 xcsf_frozen_lake.py \
    --experiment-name="$SLURM_JOB_ID" \
    --fl-grid-size="$1" \
    --fl-slip-prob="$2" \
    --xcsf-pred-strat="$3" \
    --xcsf-poly-order="$4" \
    --xcsf-seed="$5" \
    --xcsf-pop-size="$6" \
    --xcsf-beta-epsilon="$7" \
    --xcsf-beta="$8" \
    --xcsf-alpha="$9" \
    --xcsf-epsilon-nought="${10}" \
    --xcsf-nu="${11}" \
    --xcsf-gamma="${12}" \
    --xcsf-theta-ga="${13}" \
    --xcsf-tau="${14}" \
    --xcsf-chi="${15}" \
    --xcsf-upsilon="${16}" \
    --xcsf-mu="${17}" \
    --xcsf-theta-del="${18}" \
    --xcsf-delta="${19}" \
    --xcsf-theta-sub="${20}" \
    --xcsf-r-nought="${21}" \
    --xcsf-weight-i-min="${22}" \
    --xcsf-weight-i-max="${23}" \
    --xcsf-mu-i="${24}" \
    --xcsf-epsilon-i="${25}" \
    --xcsf-fitness-i="${26}" \
    --xcsf-m-nought="${27}" \
    --xcsf-x-nought="${28}" \
    --xcsf-do-ga-subsumption \
    --xcsf-delta-rls="${29}" \
    --xcsf-tau-rls="${30}" \
    --xcsf-lambda-rls="${31}" \
    --xcsf-eta="${32}" \
    --xcsf-p-explr="${33}"\
    --monitor-freq-episodes="${34}" \
    --monitor-num-ticks="${35}"
mv "slurm-${SLURM_JOB_ID}.out" "${SLURM_JOB_ID}/"

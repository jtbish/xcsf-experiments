#!/bin/bash
#SBATCH --partition=batch
#SBATCH --cpus-per-task=1

source ~/virtualenvs/xcsf/bin/activate
python3 xcsf_frozen_lake.py \
    --experiment-name="$SLURM_JOB_ID" \
    --fl-grid-size="$1" \
    --fl-slip-prob="$2" \
    --fl-iod-strat-base-train="$3" \
    --fl-iod-strat-base-test="$4" \
    --xcsf-pred-strat="$5" \
    --xcsf-poly-order="$6" \
    --xcsf-seed="$7" \
    --xcsf-pop-size="$8" \
    --xcsf-beta-epsilon="$9" \
    --xcsf-beta="${10}" \
    --xcsf-alpha="${11}" \
    --xcsf-epsilon-nought="${12}" \
    --xcsf-nu="${13}" \
    --xcsf-gamma="${14}" \
    --xcsf-theta-ga="${15}" \
    --xcsf-tau="${16}" \
    --xcsf-chi="${17}" \
    --xcsf-upsilon="${18}" \
    --xcsf-mu="${19}" \
    --xcsf-theta-del="${20}" \
    --xcsf-delta="${21}" \
    --xcsf-theta-sub="${22}" \
    --xcsf-r-nought="${23}" \
    --xcsf-weight-i-min="${24}" \
    --xcsf-weight-i-max="${25}" \
    --xcsf-mu-i="${26}" \
    --xcsf-epsilon-i="${27}" \
    --xcsf-fitness-i="${28}" \
    --xcsf-m-nought="${29}" \
    --xcsf-x-nought="${30}" \
    --xcsf-do-ga-subsumption \
    --xcsf-delta-rls="${31}" \
    --xcsf-tau-rls="${32}" \
    --xcsf-lambda-rls="${33}" \
    --xcsf-eta="${34}" \
    --xcsf-p-explr="${35}"\
    --monitor-freq-ga-calls="${36}" \
    --monitor-num-ticks="${37}"
mv "slurm-${SLURM_JOB_ID}.out" "${SLURM_JOB_ID}/"
#mv "${SLURM_JOB_ID}.prof" "${SLURM_JOB_ID}/"

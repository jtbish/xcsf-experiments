# xcsf-experiments

XCS(F) experiments for the paper "Pittsburgh learning classifier systems for explainable reinforcement learning: comparing with XCS" (https://doi.org/10.1145/3512290.3528767)

Most important file is the run script:

```fl/xcsf_frozen_lake.py```

this being the script that actually runs XCS on FrozenLake.

Incidental scripts to pass args to this .py file and run on Slurm are:
```fl/xcsf_frozen_lake.sh``` and ```fl/run_xcsf_frozen_lake.sh```

## Dependencies for run script
rlenvs: https://github.com/jtbish/rlenvs

xcsfrl: https://github.com/jtbish/xcsfrl

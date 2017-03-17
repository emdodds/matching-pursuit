#!/bin/bash
#
# Partition:
#SBATCH --partition=cortex
#
# Wall clock limit:
#SBATCH --time=48:00:00
#
# Memory:
#SBATCH --mem-per-cpu=8G
#
# Constraint:
#SBATCH --constraint=cortex_k40
#SBATCH --gres=gpu:1

cd /global/home/users/edodds/matching-pursuit
export MODULEPATH=/global/software/sl-6.x64_64/modfiles/apps:$MODULEPATH
module load ml/tensorflow/0.12.1
python scripts/fit_mp.py
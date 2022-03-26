#!/bin/bash
#
#
#SBATCH --job-name=delmodel
#SBATCH --output=logs/delmodel.out
#SBATCH --error=logs/delmodel.err
#SBATCH -p dpwall
#SBATCH -D /oak/stanford/groups/dpwall/users/kpaskov/PhasingFamilies
#SBATCH -t 6:00:00
#SBATCH --mem=128GB

module load py-numpy/1.19.2_py36

srun python3 phase/model_deletions.py 0.1 5

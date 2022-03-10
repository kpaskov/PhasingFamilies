#!/bin/bash
#
#
#SBATCH --job-name=delmodel
#SBATCH --output=logs/delmodel.out
#SBATCH --error=logs/delmodel.err
#SBATCH -p dpwall
#SBATCH -D /oak/stanford/groups/dpwall/users/kpaskov/PhasingFamilies
#SBATCH -t 3:00:00
#SBATCH --mem=8G
#SBATCH --array=0-40

module load py-numpy/1.14.3_py36
module load py-scipy/1.1.0_py36

srun python3 phase/phase/model_deletions.py 0.1 $SLURM_ARRAY_TASK_ID
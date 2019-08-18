#!/bin/bash
#
#
#SBATCH --job-name=pull
#SBATCH --output=pullvars%A_%a.out
#SBATCH --error=pullvars%A_%a.err
#SBATCH --array=0-20
#SBATCH -p dpwall
#SBATCH -D /scratch/PI/dpwall/DATA/iHART/kpaskov/PhasingFamilies
#SBATCH -t 1:00:00
#SBATCH --mem=64G

module load py-scipystack/1.0_py36

# Print this sub-job's task ID
echo "My SLURM_ARRAY_TASK_ID is " $SLURM_ARRAY_TASK_ID
srun python3 maya/pull_genotypes_for_variants.py /scratch/PI/dpwall/maya_project/autismClassifier/data/bootstrap/1e6_bootstrap/$SLURM_ARRAY_TASK_ID.pkl split_gen_miss_1kg /scratch/PI/dpwall/maya_project/autismClassifier/data/bootstrap/1e6_bootstrap/$SLURM_ARRAY_TASK_ID.1kg

#!/bin/bash
#
#
#SBATCH --job-name=ad
#SBATCH --output=ad%A_%a.out
#SBATCH --error=ad%A_%a.err
#SBATCH --array=20-22
#SBATCH -p dpwall
#SBATCH -D /scratch/PI/dpwall/DATA/iHART/kpaskov/PhasingFamilies
#SBATCH -t 10:00:00
#SBATCH --mem=64

# Print this sub-job's task ID
echo "My SLURM_ARRAY_TASK_ID is " $SLURM_ARRAY_TASK_ID

module load py-scipystack/1.0_py36

srun python3 preprocessing/pull_ad_data.py ../../vcf/v3.4/$SLURM_ARRAY_TASK_ID.reheader.vcf.gz split_ad $SLURM_ARRAY_TASK_ID

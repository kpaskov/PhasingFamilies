#!/bin/bash
#
#
#SBATCH --job-name=mask
#SBATCH --output=mask%A_%a.out
#SBATCH --error=mask%A_%a.err
#SBATCH --array=1-1
#SBATCH -p dpwall
#SBATCH -D /scratch/PI/dpwall/DATA/iHART/kpaskov/PhasingFamilies
#SBATCH -t 10:00:00
#SBATCH --mem=32G  

# Print this sub-job's task ID
echo "My SLURM_ARRAY_TASK_ID is " $SLURM_ARRAY_TASK_ID

module load py-scipystack/1.0_py36
srun python3 phase/mask_bad_fits.py $SLURM_ARRAY_TASK_ID $1 split_gen_miss_ssc phased_ssc

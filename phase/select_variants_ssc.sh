#!/bin/bash
#
#
#SBATCH --job-name=select
#SBATCH --output=select%A_%a.out
#SBATCH --error=select%A_%a.err
#SBATCH --array=5-22
#SBATCH -p dpwall
#SBATCH -D /scratch/PI/dpwall/DATA/iHART/kpaskov/PhasingFamilies

# Print this sub-job's task ID
echo "My SLURM_ARRAY_TASK_ID is " $SLURM_ARRAY_TASK_ID

module load py-scipystack/1.0_py36
srun python3 phase/select_variants_snp.py $SLURM_ARRAY_TASK_ID split_gen_ssc

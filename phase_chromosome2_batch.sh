#!/bin/bash
#
#
#SBATCH --job-name=phase
#SBATCH --output=phase%A_%a.out
#SBATCH --error=phase%A_%a.err
#SBATCH --array=1-22
#SBATCH -p dpwall
#SBATCH -D /scratch/PI/dpwall/DATA/iHART/kpaskov/PhasingFamilies
#SBATCH -t 30:00:00
#SBATCH --mem=64G

# Print this sub-job's task ID
echo "My SLURM_ARRAY_TASK_ID is " $SLURM_ARRAY_TASK_ID

module load py-scipystack/1.0_py36
srun python3 phase_chromosome2.py $SLURM_ARRAY_TASK_ID $1 ../../vcf/v3.4/v34.forCompoundHet.ped split_gen phased

#!/bin/bash
#
#
#SBATCH --job-name=split
#SBATCH --output=split%A_%a.out
#SBATCH --error=split%A_%a.err
#SBATCH --array=1-22
#SBATCH --time=01:00:00
#SBATCH -p dpwall
#SBATCH -D /scratch/PI/dpwall/DATA/iHART/kpaskov/PhasingFamilies

module load gcc/4.8.1
module load boost

# Print this sub-job's task ID
echo "My SLURM_ARRAY_TASK_ID is " $SLURM_ARRAY_TASK_ID
srun ./split_by_family ../../vcf/v3.4/v34.$SLURM_ARRAY_TASK_ID.vcf.gz ../../vcf/v3.4/vcf.noLCL.ped ../../vcf/by_family 

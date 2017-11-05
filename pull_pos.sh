#!/bin/bash
#
#
#SBATCH --job-name=pull_pos
#SBATCH --output=pull_pos%A_%a.out
#SBATCH --error=pull_pos%A_%a.err
#SBATCH --array=1-22
#SBATCH -p dpwall
#SBATCH -D /scratch/PI/dpwall/DATA/iHART/kpaskov/PhasingFamilies

module load bcftools

# Print this sub-job's task ID
echo "My SLURM_ARRAY_TASK_ID is " $SLURM_ARRAY_TASK_ID

bcftools query -f '%POS\n' ../../vcf/v3.4/v34.$SLURM_ARRAY_TASK_ID.vcf.gz > data/v34.$SLURM_ARRAY_TASK_ID.txt

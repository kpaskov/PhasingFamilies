#!/bin/bash
#
#
#SBATCH --job-name=pull
#SBATCH --output=pull%A_%a.out
#SBATCH --error=pull%A_%a.err
#SBATCH --array=1-22
#SBATCH -p dpwall
#SBATCH -D /scratch/PI/dpwall/DATA/iHART/kpaskov/PhasingFamilies
#SBATCH -t 20:00:00
#SBATCH --mem=64G

module load python/3.4.3

# Print this sub-job's task ID
echo "My SLURM_ARRAY_TASK_ID is " $SLURM_ARRAY_TASK_ID
srun python3 pull_family_data.py ../../vcf/v3.4/$SLURM_ARRAY_TASK_ID.reheader.vcf.gz ../../vcf/v3.4/v34.forCompoundHet.ped raw_data $SLURM_ARRAY_TASK_ID

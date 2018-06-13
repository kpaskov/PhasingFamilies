#!/bin/bash
#
#
#SBATCH --job-name=pull
#SBATCH --output=pull%A_%a.out
#SBATCH --error=pull%A_%a.err
#SBATCH --array=1-7
#SBATCH -p dpwall
#SBATCH -D /scratch/PI/dpwall/DATA/iHART/kpaskov/PhasingFamilies
#SBATCH -t 30:00:00
#SBATCH --mem=32G

module load py-scipystack/1.0_py36

# Print this sub-job's task ID
echo "My SLURM_ARRAY_TASK_ID is " $SLURM_ARRAY_TASK_ID
srun python3 pull_gen_data.py ../../vcf/v3.4/$SLURM_ARRAY_TASK_ID.reheader.vcf.gz ../../vcf/v3.4/v34.forCompoundHet.ped split_gen $SLURM_ARRAY_TASK_ID

#!/bin/bash
#
#
#SBATCH --job-name=dp
#SBATCH --output=dp%A_%a.out
#SBATCH --error=dp%A_%a.err
#SBATCH --array=1-22
#SBATCH -p dpwall
#SBATCH -D /scratch/PI/dpwall/DATA/iHART/kpaskov/PhasingFamilies
#SBATCH -t 10:00:00
#SBATCH --mem=32

module load py-scipystack/1.0_py36

srun python3 phase/pull_dp_data.py ../../vcf/v3.4/$1.reheader.vcf.gz ../../vcf/v3.4/v34.vcf.ped split_dp $SLURM_ARRAY_TASK_ID $1

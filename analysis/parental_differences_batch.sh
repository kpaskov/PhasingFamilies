#!/bin/bash
#
#
#SBATCH --job-name=pardiff
#SBATCH --output=pardiff%A_%a.out
#SBATCH --error=pardiff%A_%a.err
#SBATCH --array=1-22
#SBATCH -p dpwall
#SBATCH -D /scratch/PI/dpwall/DATA/iHART/kpaskov/PhasingFamilies
#SBATCH -t 10:00:00
#SBATCH --mem=64G

# Print this sub-job's task ID
echo "My SLURM_ARRAY_TASK_ID is " $SLURM_ARRAY_TASK_ID

module load py-scipystack/1.0_py36
#srun python3 analysis/parental_differences.py $SLURM_ARRAY_TASK_ID split_gen_miss data/160826.ped parental_differences_ihart
srun python3 analysis/parental_differences.py $SLURM_ARRAY_TASK_ID split_gen_miss_ssc ../../vcf/SSC/ssc.ped parental_differences_ssc
srun python3 analysis/parental_differences.py $SLURM_ARRAY_TASK_ID split_gen_miss_psp ../../vcf/SSC/ssc.ped parental_differences_psp

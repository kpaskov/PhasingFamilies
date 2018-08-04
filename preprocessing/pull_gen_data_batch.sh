#!/bin/bash
#
#
#SBATCH --job-name=pull
#SBATCH --output=pull%A_%a.out
#SBATCH --error=pull%A_%a.err
#SBATCH --array=1-22
#SBATCH -p dpwall
#SBATCH -D /scratch/PI/dpwall/DATA/iHART/kpaskov/PhasingFamilies
#SBATCH -t 30:00:00
#SBATCH --mem=32G

module load py-scipystack/1.0_py36

# Print this sub-job's task ID
echo "My SLURM_ARRAY_TASK_ID is " $SLURM_ARRAY_TASK_ID
#srun python3 preprocessing/pull_gen_data.py ../../vcf/v3.4/$SLURM_ARRAY_TASK_ID.reheader.vcf.gz split_gen_miss $SLURM_ARRAY_TASK_ID
#srun python3 preprocessing/pull_gen_data.py ../../vcf/SSC/ssc.$SLURM_ARRAY_TASK_ID.vcf.gz split_gen_miss_ssc $SLURM_ARRAY_TASK_ID 
srun python3 preprocessing/pull_gen_data.py ../../../1kg.release.20130502/ALL.chr$SLURM_ARRAY_TASK_ID.phase3_shapeit2_mvncall_integrated_v5.20130502.genotypes.vcf.gz split_gen_miss_1kg $SLURM_ARRAY_TASK_ID

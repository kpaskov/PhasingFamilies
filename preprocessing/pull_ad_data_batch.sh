#!/bin/bash
#
#
#SBATCH --job-name=pull
#SBATCH --output=logs/pullad_%a.out
#SBATCH --error=logs/pullad_%a.err
#SBATCH --array=1-22
#SBATCH -p dpwall
#SBATCH -D /oak/stanford/groups/dpwall/users/kpaskov/PhasingFamilies
#SBATCH -t 30:00:00
#SBATCH --mem=32G

module load py-scipystack/1.0_py36

# Print this sub-job's task ID
echo "My SLURM_ARRAY_TASK_ID is " $SLURM_ARRAY_TASK_ID

#srun python3 preprocessing/pull_ad_data.py ../../../ihart/vcf/v34/$SLURM_ARRAY_TASK_ID.reheader.vcf.gz data/SSSEs_ihart_pass.txt split_gen_ihart $SLURM_ARRAY_TASK_ID

#srun python3 preprocessing/pull_ad_data.py ../../../ihart/vcf/psp/psp.$SLURM_ARRAY_TASK_ID.vcf.gz ../SBSE/SSSEs_psp_pass.txt split_gen_psp $SLURM_ARRAY_TASK_ID

srun python3 preprocessing/pull_ad_data.py ../../../ihart/vcf/ssc/ssc.$SLURM_ARRAY_TASK_ID.vcf.gz ../SBSE/SSSEs_ssc_pass.txt split_gen_ssc $SLURM_ARRAY_TASK_ID

#!/bin/bash
#
#
#SBATCH --job-name=famgen
#SBATCH --output=logs/famgen%A_%a.out
#SBATCH --error=logs/famgen%A_%a.err
#SBATCH --array=1-22
#SBATCH -p dpwall
#SBATCH -D /oak/stanford/groups/dpwall/users/kpaskov/PhasingFamilies
#SBATCH --mem=32G

module load py-numpy/1.14.3_py36
module load py-scipy/1.1.0_py36

# Print this sub-job's task ID
echo "My SLURM_ARRAY_TASK_ID is " $SLURM_ARRAY_TASK_ID

#srun python3 preprocessing/pull_famgen_counts.py split_gen_ihart data/160826.ped $SLURM_ARRAY_TASK_ID

#srun python3 preprocessing/pull_famgen_counts.py split_gen_ihart_ancestry ../../vcf/v3.4/v34.vcf.ped $SLURM_ARRAY_TASK_ID

#srun python3 preprocessing/pull_famgen_counts.py split_gen_ihart_23andme_ancestry ../../vcf/v3.4/v34.vcf.ped $SLURM_ARRAY_TASK_ID

#srun python3 preprocessing/pull_famgen_counts.py split_gen_ssc data/ssc.ped $SLURM_ARRAY_TASK_ID

#srun python3 preprocessing/pull_famgen_counts.py split_gen_ssc_pilot data/ssc.ped $SLURM_ARRAY_TASK_ID

srun python3 parameter_estimation/pull_famgen_counts.py split_gen_spark /oak/stanford/groups/dpwall/simons_spark/snp/SPARK.pruned.fam $SLURM_ARRAY_TASK_ID

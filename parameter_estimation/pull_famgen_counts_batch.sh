#!/bin/bash
#
#
#SBATCH --job-name=famgen
#SBATCH --output=logs/famgen_%a.out
#SBATCH --error=logs/famgen_%a.err
#SBATCH --array=1-22%15
#SBATCH -p dpwall
#SBATCH -D /oak/stanford/groups/dpwall/users/kpaskov/PhasingFamilies
#SBATCH --mem=16G

module load py-numpy/1.14.3_py36
module load py-scipy/1.1.0_py36

# Print this sub-job's task ID
echo "My SLURM_ARRAY_TASK_ID is " $SLURM_ARRAY_TASK_ID

#srun python3 parameter_estimation/pull_famgen_counts.py split_gen_ihart data/v34.vcf.ped $SLURM_ARRAY_TASK_ID

#srun python3 parameter_estimation/pull_famgen_counts.py split_gen_ihart data/v34.vcf.ped $SLURM_ARRAY_TASK_ID --exclude data/btu356-suppl_data/btu356_LCR-hs37d5.bed/btu356_LCR-hs37d5.bed split_gen_ihart_HCR 

#srun python3 parameter_estimation/pull_famgen_counts.py split_gen_ihart data/v34.vcf.ped $SLURM_ARRAY_TASK_ID --include data/btu356-suppl_data/btu356_LCR-hs37d5.bed/btu356_LCR-hs37d5.bed split_gen_ihart_LCR

#srun python3 parameter_estimation/pull_famgen_counts.py split_gen_spark data/spark.ped $SLURM_ARRAY_TASK_ID 

#srun python3 parameter_estimation/pull_famgen_counts.py split_gen_ssc data/ssc.ped $SLURM_ARRAY_TASK_ID

#srun python3 parameter_estimation/pull_famgen_counts.py split_gen_ihart_chip data/v34.vcf.ped $SLURM_ARRAY_TASK_ID

#srun python3 parameter_estimation/pull_famgen_counts.py split_gen_ihart_famsize data/v34.vcf.ped.famsize.ped $SLURM_ARRAY_TASK_ID  

#srun python3 parameter_estimation/pull_famgen_counts.py split_gen_spark_exome_pilot /scratch/groups/dpwall/DATA/SSC/SPARK/phenotype/spark.ped $SLURM_ARRAY_TASK_ID

#srun python3 parameter_estimation/pull_famgen_counts.py split_gen_spark_exome data/spark_jae.ped $SLURM_ARRAY_TASK_ID --include data/exome_calling_regions.v1.interval_list split_gen_spark_exome_EX

#srun python3 parameter_estimation/pull_famgen_counts.py split_gen_spark_exome data/spark_jae.ped $SLURM_ARRAY_TASK_ID --exclude data/btu356-suppl_data/btu356_LCR-hs37d5.bed/btu356_LCR-hs37d5.bed split_gen_spark_exome_HCR

#srun python3 parameter_estimation/pull_famgen_counts.py split_gen_spark_exome data/spark_jae.ped $SLURM_ARRAY_TASK_ID --include data/btu356-suppl_data/btu356_LCR-hs37d5.bed/btu356_LCR-hs37d5.bed split_gen_spark_exome_LCR

#srun python3 preprocessing/pull_famgen_counts.py split_gen_ssc_pilot data/ssc.ped $SLURM_ARRAY_TASK_ID

#srun python3 parameter_estimation/pull_famgen_counts.py split_gen_spark data/spark.ped $SLURM_ARRAY_TASK_ID

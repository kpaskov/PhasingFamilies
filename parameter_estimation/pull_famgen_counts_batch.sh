#!/bin/bash
#
#
#SBATCH --job-name=famgen
#SBATCH --output=logs/famgen_%a.out
#SBATCH --error=logs/famgen_%a.err
#SBATCH --array=1-22%15
#SBATCH -p dpwall
#SBATCH -D /oak/stanford/groups/dpwall/users/kpaskov/PhasingFamilies
#SBATCH -t 30:00:00
#SBATCH --mem=8G

module load py-numpy/1.14.3_py36
module load py-scipy/1.1.0_py36

# Print this sub-job's task ID
echo "My SLURM_ARRAY_TASK_ID is " $SLURM_ARRAY_TASK_ID

# ------------------------ For parameter estimation paper -----------------------

# ihart
#srun python3 parameter_estimation/pull_famgen_counts.py split_gen_ihart data/v34.vcf.ped $SLURM_ARRAY_TASK_ID split_gen_ihart
srun python3 parameter_estimation/pull_famgen_counts_vcf.py /oak/stanford/groups/dpwall/ihart/vcf/v34/$SLURM_ARRAY_TASK_ID.reheader.vcf.gz data/v34.vcf.ped $SLURM_ARRAY_TASK_ID counts_ihart --family_sizes 3 4 5 6 7 --depth_bins 20 30

# ihart HCR
#srun python3 parameter_estimation/pull_famgen_counts.py split_gen_ihart data/v34.vcf.ped $SLURM_ARRAY_TASK_ID split_gen_ihart_HCR --exclude data/btu356-suppl_data/btu356_LCR-hs37d5.bed/btu356_LCR-hs37d5.bed

# ihart LCR
# srun python3 parameter_estimation/pull_famgen_counts.py split_gen_ihart data/v34.vcf.ped $SLURM_ARRAY_TASK_ID split_gen_ihart_LCR --include data/btu356-suppl_data/btu356_LCR-hs37d5.bed/btu356_LCR-hs37d5.bed

# ihart (GATK 3.2)
# srun python3 parameter_estimation/pull_famgen_counts.py split_gen_ihart3.2 data/v34.vcf.ped $SLURM_ARRAY_TASK_ID split_gen_ihart3.2

# ihart EX
# srun python3 parameter_estimation/pull_famgen_counts.py split_gen_ihart3.2 data/v34.vcf.ped $SLURM_ARRAY_TASK_ID split_gen_ihart_EX --include data/VCRome_2_1_hg19_primary_targets.bed

# ihart identicals
# srun python3 parameter_estimation/pull_famgen_counts.py split_gen_ihart data/ihart_identicals.ped $SLURM_ARRAY_TASK_ID split_gen_ihart_identicals

# spark exome EX
# srun python3 parameter_estimation/pull_famgen_counts.py split_gen_spark_exome data/spark.ped $SLURM_ARRAY_TASK_ID split_gen_spark_exome_EX --include data/VCRome_2_1_hg38_primary_targets_liftover_ordered.bed 
#srun python3 parameter_estimation/pull_famgen_counts_vcf.py /oak/stanford/groups/dpwall/ihart/vcf/v34/$SLURM_ARRAY_TASK_ID.reheader.vcf.gz data/v34.vcf.ped $SLURM_ARRAY_TASK_ID counts_ihart --family_sizes 3 4 5 6 7 --depth_bins 20 30

# spark exome EX25
# srun python3 parameter_estimation/pull_famgen_counts.py split_gen_spark_exome data/spark.ped $SLURM_ARRAY_TASK_ID split_gen_spark_exome_EX25 --include data/VCRome_2_1_hg38_primary_targets_liftover_0_25.bed

# spark exome EX50
# srun python3 parameter_estimation/pull_famgen_counts.py split_gen_spark_exome data/spark.ped $SLURM_ARRAY_TASK_ID split_gen_spark_exome_EX50 --include data/VCRome_2_1_hg38_primary_targets_liftover_25_50.bed

# spark exome EX75
#srun python3 parameter_estimation/pull_famgen_counts.py split_gen_spark_exome data/spark.ped $SLURM_ARRAY_TASK_ID split_gen_spark_exome_EX75 --include data/VCRome_2_1_hg38_primary_targets_liftover_50_75.bed

# spark exome EX1000
#srun python3 parameter_estimation/pull_famgen_counts.py split_gen_spark_exome data/spark.ped $SLURM_ARRAY_TASK_ID split_gen_spark_exome_EX1000 --include data/VCRome_2_1_hg38_primary_targets_liftover_75_1000.bed

# spark exome identicals
#srun python3 parameter_estimation/pull_famgen_counts.py split_gen_spark_exome data/spark_exome_identicals.ped $SLURM_ARRAY_TASK_ID split_gen_spark_exome_EX_identicals --include data/VCRome_2_1_hg38_primary_targets_liftover_ordered.bed

# spark
#srun python3 parameter_estimation/pull_famgen_counts.py split_gen_spark data/spark.ped $SLURM_ARRAY_TASK_ID split_gen_spark

# spark identicals
#srun python3 parameter_estimation/pull_famgen_counts.py split_gen_spark data/spark_identicals.ped $SLURM_ARRAY_TASK_ID split_gen_spark_identicals

# ------------------------ For deletions paper -----------------------    

# platinum
#srun python3 parameter_estimation/pull_famgen_counts.py split_gen_platinum data/platinum.ped $SLURM_ARRAY_TASK_ID split_gen_platinum

# SSC
#srun python3 parameter_estimation/pull_famgen_counts.py split_gen_ssc data/ssc.ped $SLURM_ARRAY_TASK_ID split_gen_ssc

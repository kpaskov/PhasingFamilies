#!/bin/bash
#
#
#SBATCH --job-name=ihart
#SBATCH --output=logs/ihart.out
#SBATCH --error=logs/ihart.err
#SBATCH -p dpwall
#SBATCH -D /oak/stanford/groups/dpwall/users/kpaskov/PhasingFamilies
#SBATCH -t 10:00:00
#SBATCH --mem=8G
#SBATCH --array=0-702%15

module load py-numpy/1.14.3_py36
module load py-scipy/1.1.0_py36


#srun python3 phase/phase_chromosome.py ../DATA/spark/spark.ped.quads.ped ../DATA/spark/genotypes phased_spark_quads params/spark_quads_params.json 1 --batch_size 5 --batch_num $SLURM_ARRAY_TASK_ID --max_af_cost 4.679 --family_size 4

#srun python3 phase/phase_chromosome.py ../DATA/ancestry/ancestry.ped.quads.ped ../DATA/ancestry/genotypes38 phased_ancestry_quads_del params/ancestry_quads_params.json 1 --batch_size 1 --batch_num $SLURM_ARRAY_TASK_ID --family_size 4 --detect_inherited_deletions

#srun python3 phase/phase_chromosome.py ../DATA/spark/sparkfam.ped.quads.ped ../DATA/spark/genotypes/wes1_array phased_spark_wes1_array_quads_upd params/spark_wes1_array_quads_params.json 1 --batch_size 4 --batch_num $SLURM_ARRAY_TASK_ID --family_size 4 --detect_inherited_deletions --detect_upd

#srun python3 phase/phase_chromosome.py ../DATA/spark/sparkfam.ped.quads.ped ../DATA/spark/genotypes/wes3_array phased_spark_wes3_array_quads_upd params/spark_wes3_array_quads_params.json 1 --batch_size 2 --batch_num $SLURM_ARRAY_TASK_ID --family_size 4 --detect_inherited_deletions --detect_upd

#srun python3 phase/phase_chromosome.py ../DATA/spark/sparkfam.ped.quads.ped ../DATA/spark/genotypes/wgs1_b01_array38 phased_spark_wgs1_b01_array_quads_upd params/spark_wgs1_b01_array_quads38_params.json 1 --batch_size 1 --batch_num $SLURM_ARRAY_TASK_ID --family_size 4 --detect_inherited_deletions --detect_upd

#srun python3 phase/phase_chromosome.py ../DATA/spark/sparkfam.ped.quads.ped ../DATA/spark/genotypes/wgs1_b02_array38 phased_spark_wgs1_b02_array_quads_del params/spark_wgs1_b02_array_quads38_params.json 1 --batch_size 1 --batch_num $SLURM_ARRAY_TASK_ID --family_size 4 --detect_inherited_deletions

#srun python3 phase/phase_chromosome.py ../DATA/spark/sparkfam.ped.quads.ped ../DATA/spark/genotypes/wgs2_array phased_spark_wgs2_array_quads_del params/spark_wgs2_array_quads_params.json 1 --batch_size 1 --batch_num $SLURM_ARRAY_TASK_ID --family_size 4 --detect_inherited_deletions

#srun python3 phase/phase_chromosome.py ../DATA/spark/sparkfam.ped.quads.ped ../DATA/spark/genotypes/wgs3_array phased_spark_wgs3_array_quads_del params/spark_wgs3_array_quads_params.json 1 --batch_size 1 --batch_num $SLURM_ARRAY_TASK_ID --family_size 4 --detect_inherited_deletions

#srun python3 phase/phase_chromosome.py ../DATA/ihart.chip/ihart.ped.quads.ped ../DATA/ihart.chip/genotypes38 phased_ihart.chip_quads38_del params/ihart.chip_quads_params.json 1 --batch_size 2 --batch_num $SLURM_ARRAY_TASK_ID --family_size 4 --detect_inherited_deletions 

#srun python3 phase/phase_chromosome.py ../DATA/spark/sparkfam.ped.quads.ped ../DATA/spark/genotypes/wgs1_b01_array phased_spark2_quads_del params/spark_wgs1_b01_array_quads_params.json 1 --batch_size 1 --batch_num $SLURM_ARRAY_TASK_ID --family_size 4 --detect_inherited_deletions

#srun python3 phase/phase_chromosome.py ../DATA/spark/sparkfam.ped.quads.ped ../DATA/spark/genotypes/wgs1_b02_array phased_spark2_quads_del params/spark_wgs1_b02_array_quads_params.json 1 --batch_size 1 --batch_num $SLURM_ARRAY_TASK_ID --family_size 4 --detect_inherited_deletions

#srun python3 phase/phase_chromosome.py ../DATA/spark/sparkfam.ped.quads.ped ../DATA/spark/genotypes/wgs2_array phased_spark2_quads_del params/spark_wgs2_array_quads_params.json 1 --batch_size 1 --batch_num $SLURM_ARRAY_TASK_ID --family_size 4 --detect_inherited_deletions

#srun python3 phase/phase_chromosome.py ../DATA/spark/sparkfam.ped.quads.ped ../DATA/spark/genotypes/wgs3_array phased_spark2_quads_del params/spark_wgs3_array_quads_params.json 1 --batch_size 1 --batch_num $SLURM_ARRAY_TASK_ID --family_size 4 --detect_inherited_deletions

#srun python3 phase/phase_chromosome.py ../DATA/spark20190423/spark.ped.quads.ped ../DATA/spark20190423/genotypes phased_spark20190423_quads params/spark20190423_quads_params.json 1 --batch_size 5 --batch_num $SLURM_ARRAY_TASK_ID --max_af_cost 4.679 --family_size 4

#srun python3 phase/phase_chromosome.py ../DATA/ancestry/ancestry.ped.quads.ped ../DATA/ancestry/genotypes38 phased_ancestry_quads_38 params/ancestry_quads_params.json 1 --batch_size 1 --batch_num $SLURM_ARRAY_TASK_ID --max_af_cost 4.679 --family_size 4

#srun python3 phase/phase_chromosome.py ../DATA/spark.exome/spark.ped.quads.ped ../DATA/spark/genotypes phased_spark.exome_quads params/spark.exome_quads_params.json 2 --batch_size 5 --batch_num $SLURM_ARRAY_TASK_ID --max_af_cost 4.679 --family_size 4

#srun python3 phase/phase_chromosome.py ../DATA/mssng/mssng.ped.quads.ped ../DATA/mssng/genotypes phased_mssng_quads_del params/mssng_quads_params.json 2 --batch_size 1 --batch_num $SLURM_ARRAY_TASK_ID --detect_inherited_deletions --family_size 4 

#srun python3 phase/phase_chromosome.py ../DATA/ssc.hg38/ssc.ped ../DATA/ssc.hg38/genotypes/phase1-1 phased_ssc.hg38_phase1-1_del params/ssc.hg38_phase1-1_params.json 2 --batch_size 1 --batch_num $SLURM_ARRAY_TASK_ID --detect_inherited_deletions --family_size 4  

#srun python3 phase/phase_chromosome.py ../DATA/ssc.hg38/ssc.ped ../DATA/ssc.hg38/genotypes/phase1-1 phased_ssc.hg38_phase1-1_upd params/ssc.hg38_phase1-1_params.json 2 --batch_size 1 --batch_num $SLURM_ARRAY_TASK_ID --detect_inherited_deletions --family_size 4 --detect_upd

#srun python3 phase/phase_chromosome.py ../DATA/ssc.hg38/ssc.ped ../DATA/ssc.hg38/genotypes/phase1-2 phased_ssc.hg38_phase1-2_del params/ssc.hg38_phase1-2_params.json 2 --batch_size 1 --batch_num $SLURM_ARRAY_TASK_ID --detect_inherited_deletions --family_size 4

#srun python3 phase/phase_chromosome.py ../DATA/ssc.hg38/ssc.ped ../DATA/ssc.hg38/genotypes/phase1-2 phased_ssc.hg38_phase1-2_upd params/ssc.hg38_phase1-2_params.json 2 --batch_size 1 --batch_num $SLURM_ARRAY_TASK_ID --detect_inherited_deletions --family_size 4 --detect_upd   

#srun python3 phase/phase_chromosome.py ../DATA/ssc.hg38/ssc.ped ../DATA/ssc.hg38/genotypes/phase1-3 phased_ssc.hg38_phase1-3_del params/ssc.hg38_phase1-3_params.json 2 --batch_size 1 --batch_num $SLURM_ARRAY_TASK_ID --detect_inherited_deletions --family_size 4 

#srun python3 phase/phase_chromosome.py ../DATA/ssc.hg38/ssc.ped ../DATA/ssc.hg38/genotypes/phase1-3 phased_ssc.hg38_phase1-3_upd params/ssc.hg38_phase1-3_params.json 2 --batch_size 1 --batch_num $SLURM_ARRAY_TASK_ID --detect_inherited_deletions --family_size 4 --detect_upd

#srun python3 phase/phase_chromosome.py ../DATA/ssc.hg38/ssc.ped ../DATA/ssc.hg38/genotypes/phase1-4 phased_ssc.hg38_phase1-4_del params/ssc.hg38_phase1-4_params.json 2 --batch_size 1 --batch_num $SLURM_ARRAY_TASK_ID --detect_inherited_deletions --family_size 4

#srun python3 phase/phase_chromosome.py ../DATA/ssc.hg38/ssc.ped ../DATA/ssc.hg38/genotypes/phase1-4 phased_ssc.hg38_phase1-4_upd params/ssc.hg38_phase1-4_params.json 2 --batch_size 1 --batch_num $SLURM_ARRAY_TASK_ID --detect_inherited_deletions --family_size 4 --detect_upd

#srun python3 phase/phase_chromosome.py ../DATA/ssc.hg38/ssc.ped ../DATA/ssc.hg38/genotypes/phase1-5 phased_ssc.hg38_phase1-5_del params/ssc.hg38_phase1-5_params.json 2 --batch_size 1 --batch_num $SLURM_ARRAY_TASK_ID --detect_inherited_deletions --family_size 4

#srun python3 phase/phase_chromosome.py ../DATA/ssc.hg38/ssc.ped ../DATA/ssc.hg38/genotypes/phase1-5 phased_ssc.hg38_phase1-5_upd params/ssc.hg38_phase1-5_params.json 2 --batch_size 1 --batch_num $SLURM_ARRAY_TASK_ID --detect_inherited_deletions --family_size 4 --detect_upd

#srun python3 phase/phase_chromosome.py ../DATA/ssc.hg38/ssc.ped ../DATA/ssc.hg38/genotypes/phase1-7 phased_ssc.hg38_phase1-7_del params/ssc.hg38_phase1-7_params.json 2 --batch_size 1 --batch_num $SLURM_ARRAY_TASK_ID --detect_inherited_deletions --family_size 4 

#srun python3 phase/phase_chromosome.py ../DATA/ssc.hg38/ssc.ped ../DATA/ssc.hg38/genotypes/phase1-7 phased_ssc.hg38_phase1-7_upd params/ssc.hg38_phase1-7_params.json 2 --batch_size 1 --batch_num $SLURM_ARRAY_TASK_ID --detect_inherited_deletions --family_size 4 --detect_upd

#srun python3 phase/phase_chromosome.py ../DATA/ssc.hg38/ssc.ped ../DATA/ssc.hg38/genotypes/phase2 phased_ssc.hg38_phase2_del params/ssc.hg38_phase2_params.json 2 --batch_size 1 --batch_num $SLURM_ARRAY_TASK_ID --detect_inherited_deletions --family_size 4 

#srun python3 phase/phase_chromosome.py ../DATA/ssc.hg38/ssc.ped ../DATA/ssc.hg38/genotypes/phase2 phased_ssc.hg38_phase2_upd params/ssc.hg38_phase2_params.json 2 --batch_size 1 --batch_num $SLURM_ARRAY_TASK_ID --detect_inherited_deletions --family_size 4 --detect_upd

#srun python3 phase/phase_chromosome.py ../DATA/ssc.hg38/ssc.ped ../DATA/ssc.hg38/genotypes/phase2_B01 phased_ssc.hg38_phase2_B01_del params/ssc.hg38_phase2_B01_params.json 2 --batch_size 1 --batch_num $SLURM_ARRAY_TASK_ID --detect_inherited_deletions --family_size 4 

#srun python3 phase/phase_chromosome.py ../DATA/ssc.hg38/ssc.ped ../DATA/ssc.hg38/genotypes/phase2_B01 phased_ssc.hg38_phase2_B01_upd params/ssc.hg38_phase2_B01_params.json 2 --batch_size 1 --batch_num $SLURM_ARRAY_TASK_ID --detect_inherited_deletions --family_size 4 --detect_upd

#srun python3 phase/phase_chromosome.py ../DATA/ssc.hg38/ssc.ped ../DATA/ssc.hg38/genotypes/phase2_Replacements phased_ssc.hg38_phase2_Replacements_del params/ssc.hg38_phase2_Replacements_params.json 2 --batch_size 1 --batch_num $SLURM_ARRAY_TASK_ID --detect_inherited_deletions --family_size 4 

#srun python3 phase/phase_chromosome.py ../DATA/ssc.hg38/ssc.ped ../DATA/ssc.hg38/genotypes/phase2_Replacements phased_ssc.hg38_phase2_Replacements_upd params/ssc.hg38_phase2_Replacements_params.json 2 --batch_size 1 --batch_num $SLURM_ARRAY_TASK_ID --detect_inherited_deletions --family_size 4 --detect_upd

#srun python3 phase/phase_chromosome.py ../DATA/ssc.hg38/ssc.ped ../DATA/ssc.hg38/genotypes/phase3_1 phased_ssc.hg38_phase3_1_del params/ssc.hg38_phase3_1_params.json 2 --batch_size 1 --batch_num $SLURM_ARRAY_TASK_ID --detect_inherited_deletions --family_size 4 

#srun python3 phase/phase_chromosome.py ../DATA/ssc.hg38/ssc.ped ../DATA/ssc.hg38/genotypes/phase3_1 phased_ssc.hg38_phase3_1_upd params/ssc.hg38_phase3_1_params.json 2 --batch_size 1 --batch_num $SLURM_ARRAY_TASK_ID --detect_inherited_deletions --family_size 4 --detect_upd

#srun python3 phase/phase_chromosome.py ../DATA/ssc.hg38/ssc.ped ../DATA/ssc.hg38/genotypes/phase3_1_B02 phased_ssc.hg38_phase3_1_B02_del params/ssc.hg38_phase3_1_B02_params.json 2 --batch_size 1 --batch_num $SLURM_ARRAY_TASK_ID --detect_inherited_deletions --family_size 4

#srun python3 phase/phase_chromosome.py ../DATA/ssc.hg38/ssc.ped ../DATA/ssc.hg38/genotypes/phase3_1_B02 phased_ssc.hg38_phase3_1_B02_upd params/ssc.hg38_phase3_1_B02_params.json 2 --batch_size 1 --batch_num $SLURM_ARRAY_TASK_ID --detect_inherited_deletions --family_size 4 --detect_upd

#srun python3 phase/phase_chromosome.py ../DATA/ssc.hg38/ssc.ped ../DATA/ssc.hg38/genotypes/phase3_2 phased_ssc.hg38_phase3_2_del params/ssc.hg38_phase3_2_params.json 2 --batch_size 1 --batch_num $SLURM_ARRAY_TASK_ID --detect_inherited_deletions --family_size 4

#srun python3 phase/phase_chromosome.py ../DATA/ssc.hg38/ssc.ped ../DATA/ssc.hg38/genotypes/phase3_2 phased_ssc.hg38_phase3_2_upd params/ssc.hg38_phase3_2_params.json 2 --batch_size 1 --batch_num $SLURM_ARRAY_TASK_ID --detect_inherited_deletions --family_size 4 --detect_upd

#srun python3 phase/phase_chromosome.py ../DATA/ssc.hg38/ssc.ped ../DATA/ssc.hg38/genotypes/phase4 phased_ssc.hg38_phase4_del params/ssc.hg38_phase4_params.json 2 --batch_size 1 --batch_num $SLURM_ARRAY_TASK_ID --detect_inherited_deletions --family_size 4 

#srun python3 phase/phase_chromosome.py ../DATA/ssc.hg38/ssc.ped ../DATA/ssc.hg38/genotypes/phase4 phased_ssc.hg38_phase4_upd params/ssc.hg38_phase4_params.json 2 --batch_size 1 --batch_num $SLURM_ARRAY_TASK_ID --detect_inherited_deletions --family_size 4 --detect_upd

#srun python3 phase/phase_chromosome.py ../DATA/ssc.hg38/ssc.ped ../DATA/ssc.hg38/genotypes/pilot phased_ssc.hg38_pilot_del params/ssc.hg38_pilot_params.json 2 --batch_size 1 --batch_num $SLURM_ARRAY_TASK_ID --detect_inherited_deletions --family_size 4

#srun python3 phase/phase_chromosome.py ../DATA/ssc.hg38/ssc.ped ../DATA/ssc.hg38/genotypes/pilot phased_ssc.hg38_pilot_upd params/ssc.hg38_pilot_params.json 2 --batch_size 1 --batch_num $SLURM_ARRAY_TASK_ID --detect_inherited_deletions --family_size 4 --detect_upd

#srun python3 phase/phase_chromosome.py ../DATA/ssc/ssc.ped ../DATA/ssc/genotypes phased_ssc_del params/ssc_params.json 2 --batch_size 1 --batch_num $SLURM_ARRAY_TASK_ID --detect_inherited_deletions --family_size 4

#srun python3 phase/phase_chromosome.py ../DATA/ihart.ms2/ihart.ped ../DATA/ihart.ms2/genotypes phased_ihart.ms2_del params/ihart.ms2_params.json 2 --family_size 6 --batch_size 1 --batch_num $SLURM_ARRAY_TASK_ID --detect_inherited_deletions

#srun python3 phase/phase_chromosome.py ../DATA/ihart.ms2/ihart.ped ../DATA/ihart.ms2/genotypes phased_ihart.ms2_del params/ihart.ms2_params.json 2 --detect_inherited_deletions --family $1

srun python3 phase/phase_chromosome.py ../DATA/ihart.ms2/ihart.ped.quads.ped ../DATA/ihart.ms2/genotypes phased_ihart.ms2_quads_del params/ihart.ms2_quads_params.json 2 --family_size 4 --batch_size 3 --batch_num $SLURM_ARRAY_TASK_ID --detect_inherited_deletions --chrom X

#srun python3 phase/phase_chromosome.py ../DATA/ihart.ms2/ihart.ped.quads.ped ../DATA/ihart.ms2/genotypes phased_ihart.ms2_quads_upd params/ihart.ms2_quads_params.json 2 --family_size 4 --batch_size 3 --batch_num $SLURM_ARRAY_TASK_ID --detect_inherited_deletions --detect_upd

#srun python3 phase/phase_chromosome.py ../DATA/spark/spark.ped ../DATA/spark/genotypes phased_spark_del params/spark_params.json 1 --batch_size 4 --batch_num $SLURM_ARRAY_TASK_ID --detect_inherited_deletions 

#srun python3 phase/phase_chromosome.py ../DATA/ihart.chip/ihart.ped ../DATA/ihart.chip/genotypes phased_ihart.chip_del params/ihart.chip_params.json 1 --batch_size 1 --batch_num $SLURM_ARRAY_TASK_ID --detect_inherited_deletions

#srun python3 phase/phase_chromosome.py ../DATA/ihart.v34/ihart.ped ../DATA/ihart.v34/genotypes phased_ihart.v34_del params/ihart.v34_params.json 2 --batch_size 1 --batch_num $SLURM_ARRAY_TASK_ID --family_size 6 --detect_inherited_deletions 

#srun python3 phase/phase_chromosome.py ../DATA/ihart.ms2/ihart.ped.quads.ped ../DATA/ihart.ms2/genotypes phased_ihart.ms2_quads_del params/ihart.ms2_quads_params.json 2 --batch_size 3 --batch_num $SLURM_ARRAY_TASK_ID --family_size 4 --detect_inherited_deletions

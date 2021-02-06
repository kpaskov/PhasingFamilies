#!/bin/bash
#
#
#SBATCH --job-name=phase_ihart
#SBATCH --output=logs/phase_ihart.out
#SBATCH --error=logs/phase_ihart.err
#SBATCH -p dpwall
#SBATCH -D /oak/stanford/groups/dpwall/users/kpaskov/PhasingFamilies
#SBATCH -t 3:00:00
#SBATCH --mem=8G
#SBATCH --array=0-710%15

module load py-numpy/1.14.3_py36
module load py-scipy/1.1.0_py36

#srun python3 phase/phase_chromosome.py ../DATA/platinum/platinum6.ped ../DATA/platinum/genotypes phased_platinum6_del params/platinum_params.json 2 --detect_deletions --batch_size 1 --batch_num $SLURM_ARRAY_TASK_ID

#srun python3 phase/phase_chromosome.py ../DATA/spark/spark.ped ../DATA/spark/genotypes 38 phased_spark params/spark_multiloss_params.json 1 --batch_size 4 --batch_num $SLURM_ARRAY_TASK_ID

#srun python3 phase/phase_chromosome.py ../DATA/spark/spark.ped.quads.ped ../DATA/spark/genotypes phased_spark_quads_del_X params/spark_quads_params.json 1 --batch_size 5 --batch_num $SLURM_ARRAY_TASK_ID --max_af_cost 4.679 --detect_deletions --chrom X

#srun python3 phase/phase_chromosome.py ../DATA/ancestry/ancestry.ped.quads.ped ../DATA/ancestry/genotypes 37 phased_ancestry_quads params/ancestry_quads_multiloss_params.json 1 --family $1 --batch_size 1 --batch_num $SLURM_ARRAY_TASK_ID --max_af_cost 4.679

srun python3 phase/phase_chromosome.py ../DATA/ihart.ms2/ihart.ped.quads.ped ../DATA/ihart.ms2/genotypes phased_ihart.ms2_quads_del params/ihart.ms2_quads_params.json 2 --batch_size 3 --batch_num $SLURM_ARRAY_TASK_ID --max_af_cost 4.679 --detect_deletions --chrom X

#srun python3 phase/phase_chromosome.py ../DATA/ihart/ihart.ped ../DATA/ihart/genotypes 37 phased_ihart_complex_families params/ihart_multiloss_params.json 2 --family $1

#srun python3 phase/phase_chromosome.py ../DATA/ihart/ihart.ped ../DATA/ihart/genotypes 37 phased_ihart params/ihart_multiloss_params.json 2 --batch_size 1 --batch_num $SLURM_ARRAY_TASK_ID

#srun python3 phase/phase_chromosome.py data/v34.vcf.ped.quads.ped split_gen_ihart 37 phased_ihart_quads params/ihart_multiloss_params.json 2 --batch_size 100 --batch_num $1 

#srun python3 phase/phase_chromosome.py ../DATA/ancestry/ancestryDNA.ped.quads.ped ../DATA/ancestry/genotypes 37 phased_ancestry_quads params/ancestry_multiloss_params.json 1

#srun python3 phase/phase_chromosome.py ../DATA/mssng/mssng.ped.quads.ped ../DATA/mssng/genotypes phased_mssng_quads_del params/mssng_quads_params.json 2 --batch_size 1 --batch_num $SLURM_ARRAY_TASK_ID --max_af_cost 4.679 --detect_deletions --no_overwrite

#srun python3 phase/phase_chromosome.py ../DATA/ssc.hg38/ssc.ped ../DATA/ssc.hg38/genotypes/phase1-1 phased_ssc.hg38_del params/ssc.hg38_phase1-1_params.json 2 --batch_size 1 --batch_num $SLURM_ARRAY_TASK_ID --max_af_cost 4.679 --detect_deletions

#srun python3 phase/phase_chromosome.py ../DATA/ssc.hg38/ssc.ped ../DATA/ssc.hg38/genotypes/phase1-2 phased_ssc.hg38_del params/ssc.hg38_phase1-2_params.json 2 --batch_size 1 --batch_num $SLURM_ARRAY_TASK_ID --max_af_cost 4.679 --detect_deletions

#srun python3 phase/phase_chromosome.py ../DATA/ssc.hg38/ssc.ped ../DATA/ssc.hg38/genotypes/phase1-3 phased_ssc.hg38_del params/ssc.hg38_phase1-3_params.json 2 --batch_size 1 --batch_num $SLURM_ARRAY_TASK_ID --max_af_cost 4.679 --detect_deletions

#srun python3 phase/phase_chromosome.py ../DATA/ssc.hg38/ssc.ped ../DATA/ssc.hg38/genotypes/phase1-4 phased_ssc.hg38_del params/ssc.hg38_phase1-4_params.json 2 --batch_size 1 --batch_num $SLURM_ARRAY_TASK_ID --max_af_cost 4.679 --detect_deletions

#srun python3 phase/phase_chromosome.py ../DATA/ssc.hg38/ssc.ped ../DATA/ssc.hg38/genotypes/phase1-5 phased_ssc.hg38_del params/ssc.hg38_phase1-5_params.json 2 --batch_size 1 --batch_num $SLURM_ARRAY_TASK_ID --max_af_cost 4.679 --detect_deletions

#srun python3 phase/phase_chromosome.py ../DATA/ssc.hg38/ssc.ped ../DATA/ssc.hg38/genotypes/phase1-7 phased_ssc.hg38_del params/ssc.hg38_phase1-7_params.json 2 --batch_size 1 --batch_num $SLURM_ARRAY_TASK_ID --max_af_cost 4.679 --detect_deletions

#srun python3 phase/phase_chromosome.py ../DATA/ssc.hg38/ssc.ped ../DATA/ssc.hg38/genotypes/phase2 phased_ssc.hg38_del params/ssc.hg38_phase2_params.json 2 --batch_size 1 --batch_num $SLURM_ARRAY_TASK_ID --max_af_cost 4.679 --detect_deletions

#srun python3 phase/phase_chromosome.py ../DATA/ssc.hg38/ssc.ped ../DATA/ssc.hg38/genotypes/phase2_B01 phased_ssc.hg38_del params/ssc.hg38_phase2_B01_params.json 2 --batch_size 1 --batch_num $SLURM_ARRAY_TASK_ID --max_af_cost 4.679 --detect_deletions

#srun python3 phase/phase_chromosome.py ../DATA/ssc.hg38/ssc.ped ../DATA/ssc.hg38/genotypes/phase2_Replacements phased_ssc.hg38_del params/ssc.hg38_phase2_Replacements_params.json 2 --batch_size 1 --batch_num $SLURM_ARRAY_TASK_ID --max_af_cost 4.679 --detect_deletions --chrom X --continue_writing

#srun python3 phase/phase_chromosome.py ../DATA/ssc.hg38/ssc.ped ../DATA/ssc.hg38/genotypes/phase3_1 phased_ssc.hg38_del params/ssc.hg38_phase3_1_params.json 2 --batch_size 1 --batch_num $SLURM_ARRAY_TASK_ID --max_af_cost 4.679 --detect_deletions

#srun python3 phase/phase_chromosome.py ../DATA/ssc.hg38/ssc.ped ../DATA/ssc.hg38/genotypes/phase3_1_B02 phased_ssc.hg38_del params/ssc.hg38_phase3_1_B02_params.json 2 --batch_size 1 --batch_num $SLURM_ARRAY_TASK_ID --max_af_cost 4.679 --detect_deletions

#srun python3 phase/phase_chromosome.py ../DATA/ssc.hg38/ssc.ped ../DATA/ssc.hg38/genotypes/phase3_2 phased_ssc.hg38_del params/ssc.hg38_phase3_2_params.json 2 --batch_size 1 --batch_num $SLURM_ARRAY_TASK_ID --max_af_cost 4.679 --detect_deletions

#srun python3 phase/phase_chromosome.py ../DATA/ssc.hg38/ssc.ped ../DATA/ssc.hg38/genotypes/phase4 phased_ssc.hg38_del params/ssc.hg38_phase4_params.json 2 --batch_size 1 --batch_num $SLURM_ARRAY_TASK_ID --max_af_cost 4.679 --detect_deletions

#srun python3 phase/phase_chromosome.py ../DATA/ssc.hg38/ssc.ped ../DATA/ssc.hg38/genotypes/pilot phased_ssc.hg38_del params/ssc.hg38_pilot_params.json 2 --batch_size 1 --batch_num $SLURM_ARRAY_TASK_ID --max_af_cost 4.679 --detect_deletions

#!/bin/bash
#
#
#SBATCH --job-name=phase
#SBATCH --output=logs/phase_%a.out
#SBATCH --error=logs/phase_%a.err
#SBATCH --array=1-23%3
#SBATCH -p dpwall
#SBATCH -D /oak/stanford/groups/dpwall/users/kpaskov/PhasingFamilies
#SBATCH -t 30:00:00
#SBATCH --mem=32G

# Print this sub-job's task ID
echo "My SLURM_ARRAY_TASK_ID is " $SLURM_ARRAY_TASK_ID

module load py-numpy/1.14.3_py36
module load py-scipy/1.1.0_py36

#srun python3 phase/phase_chromosome.py $SLURM_ARRAY_TASK_ID $1 data/spark.ped split_gen_spark 38 phased_spark parameter_estimation/spark_params_ext.json False

#srun python3 phase/phase_chromosome.py $SLURM_ARRAY_TASK_ID $1 data/spark.ped.quads.ped split_gen_spark 38 phased_spark_quads params/spark_quads_params_ext.json False

#srun python3 phase/phase_chromosome.py $SLURM_ARRAY_TASK_ID $1 data/ssc.ped split_gen_ssc 37 phased_ssc parameter_estimation/ssc_params_ext.json False

#srun python3 phase/phase_chromosome.py $SLURM_ARRAY_TASK_ID $1 data/v34.vcf.ped split_gen_ihart 37 phased_ihart parameter_estimation/ihart_params_ext.json FALSE

#srun python3 phase/phase_chromosome.py $SLURM_ARRAY_TASK_ID $1 data/v34.vcf.ped.quads.ped split_gen_ihart 37 phased_ihart_quads params/ihart_quads_params_ext.json False $2 $3

srun python3 phase/phase_chromosome.py $SLURM_ARRAY_TASK_ID data/v34.vcf.ped split_gen_ihart 37 phased_ihart_AU0197 params/ihart_multiloss_params.json 2 --detect_deletions --family AU0197

#srun python3 phase/phase_chromosome.py $SLURM_ARRAY_TASK_ID $1 data/platinum.ped split_gen_platinum 37 phased_platinum parameter_estimation/platinum_params.json $2 $3

#srun python3 phase/phase_chromosome.py $SLURM_ARRAY_TASK_ID $1 data/ssc.ped split_gen_ssc 37 phased_ssc parameter_estimation/ssc_ind_pass_nodel_params.json $2 $3 

#srun python3 phase/phase_chromosome.py $SLURM_ARRAY_TASK_ID $1 data/v34.vcf.ped split_gen_ihart_chip phased_ihart_chip parameter_estimation/ihart_chip_ind_pass_params.json $2 $3

#srun python3 phase/phase_chromosome.py $SLURM_ARRAY_TASK_ID $1 data/spark_jae.ped split_gen_spark phased_spark parameter_estimation/spark_ind_pass_params.json $2 $3

#srun python3 phase/phase_chromosome.py $SLURM_ARRAY_TASK_ID $1 /scratch/groups/dpwall/DATA/SSC/SPARK/phenotype/spark.ped split_gen_spark_exome_pilot phased_spark_exome_pilot parameter_estimation/spark_exome_pilot_ind_pass_params.json $2 $3

#srun python3 phase/phase_chromosome.py $SLURM_ARRAY_TASK_ID $1 data/spark2.ped split_gen_spark_exome 38 phased_spark_exome parameter_estimation/spark_exome_ind_pass_nodel_params.json $2 $3

#srun python3 phase/phase_chromosome.py $SLURM_ARRAY_TASK_ID $1 data/v34.vcf.ped split_gen_ihart phased_ihart_males_only parameter_estimation/ihart_params.json $2 $3

#srun python3 phase/phase_chromosome.py $SLURM_ARRAY_TASK_ID $1 data/160826.ped.quads.ped split_gen_ihart phased_ihart_quad parameter_estimation/ihart_params.json $2 $3   

#srun python3 phase/phase_chromosome.py $SLURM_ARRAY_TASK_ID $1 ../../vcf/v3.4/v34.vcf.ped split_gen_ihart_23andme phased_ihart_23andme parameter_estimation/23andme_params.json $2 $3

#srun python3 phase/phase_chromosome.py $SLURM_ARRAY_TASK_ID $1 ../../vcf/v3.4/v34.vcf.ped split_gen_ihart_chip phased_ihart_chip parameter_estimation/ihart_chip_params.json $2 $3

#srun python3 phase/phase_chromosome.py $SLURM_ARRAY_TASK_ID $1 ../../vcf/v3.4/v34.vcf.ped split_gen_ihart_ancestry phased_ihart_ancestry parameter_estimation/ancestry_params.json $2 $3 

#srun python3 phase/phase_chromosome.py $SLURM_ARRAY_TASK_ID $1 ../../vcf/v3.4/v34.vcf.ped split_gen_ihart_23andme_ancestry phased_ihart_23andme_ancestry parameter_estimation/23andme_ancestry_params.json $2 $3

#srun python3 phase/phase_chromosome.py $SLURM_ARRAY_TASK_ID $1 data/spark.ped split_gen_spark phased_spark parameter_estimation/spark_params.json $2 $3

#srun python3 phase/phase_chromosome.py $SLURM_ARRAY_TASK_ID $1 data/spark.ped.quads.ped split_gen_spark phased_spark_quad parameter_estimation/spark_params.json $2 $3

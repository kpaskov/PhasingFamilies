#!/bin/bash
#
#SBATCH --job-name=phaseX
#SBATCH --output=logs/phaseX.out
#SBATCH --error=logs/phaseX.err
#SBATCH -p dpwall                                                                                
#SBATCH -D /oak/stanford/groups/dpwall/users/kpaskov/PhasingFamilies                           
#SBATCH -t 30:00:00                                                                          
#SBATCH --mem=64G

module load py-numpy/1.14.3_py36
module load py-scipy/1.1.0_py36

#srun python3 phase/phase_chromosome.py X $1 data/v34.vcf.ped split_gen_ihart phased_ihart parameter_estimation/ihart_params.json $2 $3 

#srun python3 phase/phase_chromosome.py X $1 data/ssc.ped split_gen_ssc phased_ssc parameter_estimation/ssc_params.json $2 $3

#srun python3 phase/phase_chromosome.py X $1 data/v34.vcf.ped split_gen_ihart phased_ihart_males_only parameter_estimation/ihart_params.json $2 $3

#srun python3 phase/phase_chromosome.py X $1 data/spark.ped split_gen_spark phased_spark parameter_estimation/spark_params.json $2 $3

srun python3 phase/phase_chromosome.py X $1 data/spark.ped.quads.ped split_gen_spark phased_spark_quad parameter_estimation/spark_params.json $2 $3

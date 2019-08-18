#!/bin/bash
#
#SBATCH --job-name=phaseX
#SBATCH --output=logs/phase%A_X.out
#SBATCH --error=logs/phase%A_X.err
#SBATCH -p dpwall                                                                                
#SBATCH -D /oak/stanford/groups/dpwall/users/kpaskov/PhasingFamilies                           
#SBATCH -t 30:00:00                                                                          
#SBATCH --mem=64G

module load py-numpy/1.14.3_py36
module load py-scipy/1.1.0_py36

#srun python3 phase/phase_chromosome.py X $1 data/v34.vcf.ped split_gen_ihart phased_ihart parameter_estimation/ihart_params.json $2 $3 

#srun python3 phase/phase_chromosome.py X $1 data/ssc.ped split_gen_ssc phased_ssc parameter_estimation/ssc_params.json $2 $3

srun python3 phase/phase_chromosome.py X $1 data/v34.vcf.ped split_gen_ihart p\
hased_ihart_males_only parameter_estimation/ihart_params.json $2 $3

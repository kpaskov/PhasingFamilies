#!/bin/bash                                                                     
#                                                                             
#                                                           
#SBATCH --job-name=pass                                                        
#SBATCH --output=logs/pass.out                                              
#SBATCH --error=logs/pass.err                                                
#SBATCH -p dpwall                                                             
#SBATCH -D /oak/stanford/groups/dpwall/users/kpaskov/PhasingFamilies            
#SBATCH -t 10:00:00                                                              
#SBATCH --mem=8G                                                                

module load py-numpy/1.14.3_py36
module load py-scipy/1.1.0_py36

#python3 preprocessing/pull_pass.py split_gen_spark --pass_from_gen

#python3 preprocessing/pull_pass.py split_gen_ssc

python3 preprocessing/pull_pass.py split_gen_ihart_chip --pass_from_gen

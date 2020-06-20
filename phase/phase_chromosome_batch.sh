#!/bin/bash
#
#
#SBATCH --job-name=phase
#SBATCH --output=logs/phase.out
#SBATCH --error=logs/phase.err
#SBATCH -p dpwall
#SBATCH -D /oak/stanford/groups/dpwall/users/kpaskov/PhasingFamilies
#SBATCH -t 30:00:00
#SBATCH --mem=8G

module load py-numpy/1.14.3_py36
module load py-scipy/1.1.0_py36


#srun python3 phase/phase_chromosome.py ../DATA/spark/spark.ped.quads.ped ../DATA/spark/genotypes 38 phased_spark_quads params/spark_quads_multiloss_params.json 1 --batch_size 200 --batch_num $1

#srun python3 phase/phase_chromosome.py data/v34.vcf.ped split_gen_ihart 37 phased_ihart_AU0197 params/ihart_multiloss_params.json 2 --family $1

#srun python3 phase/phase_chromosome.py data/v34.vcf.ped.quads.ped split_gen_ihart 37 phased_ihart_quads params/ihart_multiloss_params.json 2 --batch_size 100 --batch_num $1 

srun python3 phase/phase_chromosome.py ../DATA/ancestry/ancestryDNA.ped.quads.ped ../DATA/ancestry/genotypes 37 phased_ancestry_quads params/spark_ancestry_multiloss_params.json 1

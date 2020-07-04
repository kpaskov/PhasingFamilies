#!/bin/bash
#
#
#SBATCH --job-name=pull
#SBATCH --output=logs/af.out
#SBATCH --error=logs/af.err
#SBATCH -p dpwall
#SBATCH -D /oak/stanford/groups/dpwall/users/kpaskov/PhasingFamilies
#SBATCH -t 1:00:00
#SBATCH --mem=8G

module load py-scipystack/1.0_py36


srun python3 preprocessing/pull_af.py ../DATA/spark/genotypes ../DATA/spark/spark.ped ../DATA/spark/genotypes


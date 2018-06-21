#!/bin/bash
#
#
#SBATCH --job-name=phase
#SBATCH -p dpwall
#SBATCH -D /scratch/PI/dpwall/DATA/iHART/kpaskov/PhasingFamilies
#SBATCH -t 30:00:00
#SBATCH --mem=64G

module load py-scipystack/1.0_py36
srun python3 phase_chromosome.py $1 $2 ../../vcf/v3.4/v34.forCompoundHet.ped split_gen

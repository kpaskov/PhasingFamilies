#!/bin/bash
#
#
#SBATCH --job-name=phase
#SBATCH -p dpwall
#SBATCH -D /scratch/PI/dpwall/DATA/iHART/kpaskov/PhasingFamilies
#SBATCH -t 30:00:00
#SBATCH --mem=32

module load py-scipystack/1.0_py36

srun python3 phase_chromosome2.py $1 ../../vcf/v3.4/v34.forCompoundHet.ped split_gen

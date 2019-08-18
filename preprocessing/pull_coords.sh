#!/bin/bash
#
#
#SBATCH --job-name=pull
#SBATCH -p dpwall
#SBATCH -D /scratch/PI/dpwall/DATA/iHART/kpaskov/PhasingFamilies
#SBATCH -t 1:00:00
#SBATCH --mem=32G

module load py-scipystack/1.0_py36
srun python3 pull_coordinates.py ../split_gen_miss_ssc/chr.$1.gen.coordinates.npy

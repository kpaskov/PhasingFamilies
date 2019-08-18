#!/bin/bash
#
#
#SBATCH --job-name=mask
#SBATCH -p dpwall
#SBATCH -D /scratch/PI/dpwall/DATA/iHART/kpaskov/PhasingFamilies
#SBATCH -t 30:00:00
#SBATCH --mem=64G  

module load py-scipystack/1.0_py36
#srun python3 phase/mask_bad_fitsX.py $1 split_gen_miss phased_ihart
srun python3 phase/mask_bad_fitsX.py $1 split_gen_miss_ssc phased_ssc

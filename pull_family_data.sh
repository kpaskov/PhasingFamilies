#!/bin/bash
#
#
#SBATCH --job-name=pull
#SBATCH -p dpwall
#SBATCH -D /scratch/PI/dpwall/DATA/iHART/kpaskov/PhasingFamilies
#SBATCH -t 30:00:00
#SBATCH --mem=32G

module load py-scipystack/1.0_py36

srun python3 pull_family_data.py ../../vcf/v3.4/$1.reheader.vcf.gz ../../vcf/v3.4/v34.forCompoundHet.ped split_gen $1

#!/bin/bash
#
#
#SBATCH --job-name=pull
#SBATCH -p dpwall
#SBATCH -D /scratch/PI/dpwall/DATA/iHART/kpaskov/PhasingFamilies
#SBATCH -t 20:00:00
#SBATCH --mem=64G

module load python/3.4.3

srun python3 pull_family_data.py ../../vcf/v3.4/$1.reheader.vcf.gz ../../vcf/v3.4/v34.forCompoundHet.ped raw_data $1

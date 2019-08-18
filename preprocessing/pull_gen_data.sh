#!/bin/bash
#
#
#SBATCH --job-name=pull
#SBATCH -p dpwall
#SBATCH -D /scratch/PI/dpwall/DATA/iHART/kpaskov/PhasingFamilies
#SBATCH -t 30:00:00
#SBATCH --mem=32G

module load py-scipystack/1.0_py36

#srun python3 preprocessing/pull_gen_data.py ../../vcf/v3.4/$1.reheader.vcf.gz split_gen_ihart $1

srun python3 preprocessing/pull_gen_data.py ../../vcf/v3.4/$1.reheader.vcf.gz split_gen_ihart $1
srun python3 preprocessing/pull_gen_data.py ../../vcf/SSC/ssc.$1.vcf.gz split_gen_ssc $1

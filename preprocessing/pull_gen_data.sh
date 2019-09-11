#!/bin/bash
#
#
#SBATCH --job-name=pull
#SBATCH --output=logs/pull.out                                                                          
#SBATCH --error=logs/pull.err
#SBATCH -p dpwall
#SBATCH -D /oak/stanford/groups/dpwall/users/kpaskov/PhasingFamilies                                       
#SBATCH -t 30:00:00
#SBATCH --mem=32G

module load py-scipystack/1.0_py36

#srun python3 preprocessing/pull_gen_data.py ../../vcf/v3.4/$1.reheader.vcf.gz split_gen_ihart $1

#srun python3 preprocessing/pull_gen_data.py ../../vcf/v3.4/$1.reheader.vcf.gz split_gen_ihart $1

#srun python3 preprocessing/pull_gen_data.py ../../vcf/SSC/ssc.$1.vcf.gz split_gen_ssc $1

srun python3 preprocessing/pull_gen_data.py /oak/stanford/groups/dpwall/simons_spark/snp/SPARK.30K.array_genotype.20190818.phaseable.passing.vcf.gz split_gen_spark $1

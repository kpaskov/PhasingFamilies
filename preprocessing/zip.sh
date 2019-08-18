#!/bin/bash
#
#
#SBATCH --job-name=pull
#SBATCH --output=logs/zip%A.out
#SBATCH --error=logs/zip%A.err
#SBATCH -p dpwall
#SBATCH -D /oak/stanford/groups/dpwall/users/kpaskov/PhasingFamilies
#SBATCH -t 30:00:00
#SBATCH --mem=32G

srun gzip /oak/stanford/groups/dpwall/simons_spark/snp/SPARK.30K.array_genotype.20190818.phaseable.passing.vcf

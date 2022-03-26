#!/bin/bash
#
#
#SBATCH --job-name=af
#SBATCH --output=logs/af.out
#SBATCH --error=logs/af.err
#SBATCH --array=1-22
#SBATCH -p dpwall
#SBATCH -D /oak/stanford/groups/dpwall/users/kpaskov/PhasingFamilies
#SBATCH -t 3:00:00
#SBATCH --mem=8G 

module load py-numpy/1.14.3_py36
module load py-scipy/1.1.0_py36
module load biology
module load py-pysam/0.15.3_py36

#srun python3 preprocessing/pull_af_gnomad.py ../DATA/ancestry/genotypes /scratch/PI/dpwall/DATA/gnomAD/r3.hg38/gnomad.genomes.r3.0.sites.chr$1.vcf.bgz 71702 $SLURM_ARRAY_TASK_ID 0

#srun python3 preprocessing/pull_af_gnomad.py ../DATA/ihart/genotypes /scratch/PI/dpwall/DATA/gnomAD/r2.1.1/gnomad.genomes.r2.1.1.sites.vcf.bgz 15708 $SLURM_ARRAY_TASK_ID $1

#srun python3 preprocessing/pull_af_gnomad.py ../DATA/platinum/genotypes /scratch/PI/dpwall/DATA/gnomAD/r2.1.1/gnomad.genomes.r2.1.1.sites.vcf.bgz 15708 $SLURM_ARRAY_TASK_ID 0

#srun python3 preprocessing/pull_af_gnomad.py ../DATA/ihart.ms2/genotypes /scratch/PI/dpwall/DATA/gnomAD/3.1/gnomad.site.all.gz 71702 $1 $SLURM_ARRAY_TASK_ID

srun python3 preprocessing/pull_af_gnomad.py ../DATA/spark20190423/genotypes /scratch/PI/dpwall/DATA/gnomAD/3.1/gnomad.genomes.v3.1.1.sites.chr$SLURM_ARRAY_TASK_ID.vcf.bgz 71702 $SLURM_ARRAY_TASK_ID 0

#srun python3 preprocessing/pull_af_gnomad.py ../DATA/spark/genotypes /scratch/PI/dpwall/DATA/gnomAD/r3.hg38/gnomad.genomes.r3.0.sites.chr$SLURM_ARRAY_TASK_ID.vcf.bgz 71702 $SLURM_ARRAY_TASK_ID $1 

#srun python3 preprocessing/pull_af_gnomad.py ../DATA/spark.exome/genotypes /scratch/PI/dpwall/DATA/gnomAD/r3.hg38/gnomad.genomes.r3.0.sites.chr$SLURM_ARRAY_TASK_ID.vcf.bgz 71702 $SLURM_ARRAY_TASK_ID $1 

#srun python3 preprocessing/pull_af_gnomad.py ../DATA/mssng/genotypes /scratch/PI/dpwall/DATA/gnomAD/r3.hg38/gnomad.genomes.r3.0.sites.chr$SLURM_ARRAY_TASK_ID.vcf.bgz 71702 $SLURM_ARRAY_TASK_ID $1

#srun python3 preprocessing/pull_af_gnomad.py ../DATA/ssc.hg38/genotypes/phase1-1 /scratch/PI/dpwall/DATA/gnomAD/r3.hg38/gnomad.genomes.r3.0.sites.chr$SLURM_ARRAY_TASK_ID.vcf.bgz 71702 $SLURM_ARRAY_TASK_ID $1

#srun python3 preprocessing/pull_af_gnomad.py ../DATA/ssc.hg38/genotypes/phase1-2 /scratch/PI/dpwall/DATA/gnomAD/r3.hg38/gnomad.genomes.r3.0.sites.chr$SLURM_ARRAY_TASK_ID.vcf.bgz 71702 $SLURM_ARRAY_TASK_ID $1

#srun python3 preprocessing/pull_af_gnomad.py ../DATA/ssc.hg38/genotypes/phase1-3 /scratch/PI/dpwall/DATA/gnomAD/r3.hg38/gnomad.genomes.r3.0.sites.chr$SLURM_ARRAY_TASK_ID.vcf.bgz 71702 $SLURM_ARRAY_TASK_ID $1

#srun python3 preprocessing/pull_af_gnomad.py ../DATA/ssc.hg38/genotypes/phase1-4 /scratch/PI/dpwall/DATA/gnomAD/r3.hg38/gnomad.genomes.r3.0.sites.chr$SLURM_ARRAY_TASK_ID.vcf.bgz 71702 $SLURM_ARRAY_TASK_ID $1

#srun python3 preprocessing/pull_af_gnomad.py ../DATA/ssc.hg38/genotypes/phase1-5 /scratch/PI/dpwall/DATA/gnomAD/r3.hg38/gnomad.genomes.r3.0.sites.chr$SLURM_ARRAY_TASK_ID.vcf.bgz 71702 $SLURM_ARRAY_TASK_ID $1

#srun python3 preprocessing/pull_af_gnomad.py ../DATA/ssc.hg38/genotypes/phase1-7 /scratch/PI/dpwall/DATA/gnomAD/r3.hg38/gnomad.genomes.r3.0.sites.chr$SLURM_ARRAY_TASK_ID.vcf.bgz 71702 $SLURM_ARRAY_TASK_ID $1

#srun python3 preprocessing/pull_af_gnomad.py ../DATA/ssc.hg38/genotypes/phase2 /scratch/PI/dpwall/DATA/gnomAD/r3.hg38/gnomad.genomes.r3.0.sites.chr$SLURM_ARRAY_TASK_ID.vcf.bgz 71702 $SLURM_ARRAY_TASK_ID $1

#srun python3 preprocessing/pull_af_gnomad.py ../DATA/ssc.hg38/genotypes/phase2_B01 /scratch/PI/dpwall/DATA/gnomAD/r3.hg38/gnomad.genomes.r3.0.sites.chr$SLURM_ARRAY_TASK_ID.vcf.bgz 71702 $SLURM_ARRAY_TASK_ID $1

#srun python3 preprocessing/pull_af_gnomad.py ../DATA/ssc.hg38/genotypes/phase2_Replacements /scratch/PI/dpwall/DATA/gnomAD/r3.hg38/gnomad.genomes.r3.0.sites.chr$SLURM_ARRAY_TASK_ID.vcf.bgz 71702 $SLURM_ARRAY_TASK_ID $1

#srun python3 preprocessing/pull_af_gnomad.py ../DATA/ssc.hg38/genotypes/phase3_1 /scratch/PI/dpwall/DATA/gnomAD/r3.hg38/gnomad.genomes.r3.0.sites.chr$SLURM_ARRAY_TASK_ID.vcf.bgz 71702 $SLURM_ARRAY_TASK_ID $1

#srun python3 preprocessing/pull_af_gnomad.py ../DATA/ssc.hg38/genotypes/phase3_1_B02 /scratch/PI/dpwall/DATA/gnomAD/r3.hg38/gnomad.genomes.r3.0.sites.chr$SLURM_ARRAY_TASK_ID.vcf.bgz 71702 $SLURM_ARRAY_TASK_ID $1

#srun python3 preprocessing/pull_af_gnomad.py ../DATA/ssc.hg38/genotypes/phase3_2 /scratch/PI/dpwall/DATA/gnomAD/r3.hg38/gnomad.genomes.r3.0.sites.chr$SLURM_ARRAY_TASK_ID.vcf.bgz 71702 $SLURM_ARRAY_TASK_ID $1

#srun python3 preprocessing/pull_af_gnomad.py ../DATA/ssc.hg38/genotypes/phase4 /scratch/PI/dpwall/DATA/gnomAD/r3.hg38/gnomad.genomes.r3.0.sites.chr$SLURM_ARRAY_TASK_ID.vcf.bgz 71702 $SLURM_ARRAY_TASK_ID $1

#srun python3 preprocessing/pull_af_gnomad.py ../DATA/ssc.hg38/genotypes/pilot /scratch/PI/dpwall/DATA/gnomAD/r3.hg38/gnomad.genomes.r3.0.sites.chr$SLURM_ARRAY_TASK_ID.vcf.bgz 71702 $SLURM_ARRAY_TASK_ID $1

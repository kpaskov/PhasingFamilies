#!/bin/bash
#
#
#SBATCH --job-name=pull
#SBATCH --output=pullcoord%A_%a.out
#SBATCH --error=pullcoord%A_%a.err
#SBATCH --array=1-22
#SBATCH -p dpwall
#SBATCH -D /scratch/PI/dpwall/DATA/iHART/kpaskov/PhasingFamilies
#SBATCH -t 3:00:00
#SBATCH --mem=32G

module load py-scipystack/1.0_py36

# Print this sub-job's task ID
echo "My SLURM_ARRAY_TASK_ID is " $SLURM_ARRAY_TASK_ID
#srun python3 maya/pull_coordinates.py split_gen_miss/chr.$SLURM_ARRAY_TASK_ID.gen.variants.txt.gz 
#srun python3 maya/pull_coordinates.py split_gen_miss_psp/chr.$SLURM_ARRAY_TASK_ID.gen.variants.txt.gz
#srun python3 maya/pull_coordinates.py split_gen_miss_ssc/chr.$SLURM_ARRAY_TASK_ID.gen.variants.txt.gz 
srun python3 maya/pull_coordinates.py split_gen_miss_1kg/chr.$SLURM_ARRAY_TASK_ID.gen.variants.txt.gz

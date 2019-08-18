#!/bin/bash
#
#
#SBATCH --job-name=pull
#SBATCH --output=pull%A_%a.out
#SBATCH --error=pull%A_%a.err
#SBATCH --array=1-22
#SBATCH -p dpwall
#SBATCH -D /scratch/PI/dpwall/DATA/iHART/kpaskov/PhasingFamilies
#SBATCH -t 30:00:00
#SBATCH --mem=32G

module load py-numpy/1.14.3_py36
module load py-scipy/1.1.0_py36

# Print this sub-job's task ID
echo "My SLURM_ARRAY_TASK_ID is " $SLURM_ARRAY_TASK_ID

#srun python3 preprocessing/pull_positions_of_interest.py split_gen_ihart data/23andme_positions.txt split_gen_ihart_23andme $SLURM_ARRAY_TASK_ID

srun python3 preprocessing/pull_positions_of_interest.py split_gen_ihart data/ancestry_positions.txt split_gen_ihart_ancestry $SLURM_ARRAY_TASK_ID

#srun python3 preprocessing/pull_positions_of_interest.py split_gen_ihart data/23andme_ancestry_positions.txt split_gen_ihart_23andme_ancestry $SLURM_ARRAY_TASK_ID

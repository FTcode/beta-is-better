#!/bin/bash
#SBATCH --job-name=primalattack
#SBATCH --output=logs/primalattack_%A_%a.out
#SBATCH --error=logs/primalattack_%A_%a.err
#SBATCH --partition=cpu
#SBATCH --time=20:00:00
#SBATCH --mem-per-cpu=4G
#SBATCH --cpus-per-task=1
#SBATCH --array=0-7799       # 26 block sizes * 300 trials = 7800 tasks

# Map array index to blocksize and trial
blocksize=$(( 45 + SLURM_ARRAY_TASK_ID / 300 ))
trial=$(( SLURM_ARRAY_TASK_ID % 300 ))

mkdir -p ./data/n100/dg
mkdir -p logs

export LD_LIBRARY_PATH=$HOME/QD-install/lib:$LD_LIBRARY_PATH
module load python
source g6k/activate

python3 -u primalattack.py n=100 m=104 q=257 dist=DiscreteGaussian\(sqrt\(2/3\)\) blocksize=${blocksize} tours=10 \
  > ./data/n100/dg/blocksize${blocksize}_tours10_trial${trial}.txt \
  2>&1
#!/bin/bash
#SBATCH --job-name=primalattack
#SBATCH --output=logs/%A_%a.out
#SBATCH --error=logs/%A_%a.err
#SBATCH --partition=cpu
#SBATCH --time=20:00:00
#SBATCH --mem-per-cpu=4G
#SBATCH --cpus-per-task=1
#SBATCH --array=0-4999       # 5 tour options * 1000 trials = 5,000 tasks

# Map array index to correct params: tours and trial
idx=${SLURM_ARRAY_TASK_ID}

tours_index=$(( idx / 1000 ))
trial=$(( idx % 1000 ))
tours_options=(1 5 10 15 20)
tours=${tours_options[$tours_index]}

# Ensure directories exist
mkdir -p ./data/n93/cbin
mkdir -p logs

# Shared output and lock file
shared_out=./data/n93/cbin/pbkz_t${tours}_all.txt
lock_file=./data/n93/cbin/pbkz_t${tours}.lock
touch "${lock_file}"

export LD_LIBRARY_PATH=$HOME/QD-install/lib:$LD_LIBRARY_PATH
module load python
source g6k/activate

# Per-job output file
perjob_out=./data/n93/cbin/pbkz_t${tours}_trial${trial}.txt

# -u ensures Python does not buffer output
python3 -u primalattack.py n=93 m=105 q=257 dist=CentredBinary\(\) \
  progressive=True max_blocksize=70 tours=${tours} verbose=True \
  > "${perjob_out}" 2>&1

# After Python finishes, append only the final line to the shared file (thread-safe)
last_line=$(tail -n 1 "${perjob_out}")
flock "${lock_file}" -c "echo \"${last_line}\" >> \"${shared_out}\""
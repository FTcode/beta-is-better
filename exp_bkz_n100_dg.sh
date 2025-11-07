#!/bin/bash
#SBATCH --job-name=primalattack
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --partition=cpu
#SBATCH --time=20:00:00
#SBATCH --mem-per-cpu=4G
#SBATCH --cpus-per-task=1
#SBATCH --array=0-20799       # 4 tour options * 26 block sizes * 200 trials = 20,800 tasks

# Map array index to correct params: tours, blocksize and trial
idx=${SLURM_ARRAY_TASK_ID}

tours_index=$(( idx / 5200 ))          # 26 blocks * 200 trials = 5200 tasks per tour option
remainder=$(( idx % 5200 ))            
blocksize=$(( 45 + remainder / 200 ))  # 200 trials per blocksize
trial=$(( remainder % 200 ))           
tours_options=(5 10 15 20)
tours=${tours_options[$tours_index]}

# Ensure directories exist
mkdir -p ./data/n100/dg
mkdir -p logs

export LD_LIBRARY_PATH=$HOME/QD-install/lib:$LD_LIBRARY_PATH
module load python
source g6k/activate

# Per-job output file
perjob_out=./data/n100/dg/bkz${blocksize}_t${tours}_trial${trial}.txt

# Shared output file
shared_out=./data/n100/dg/bkz${blocksize}_t${tours}_all.txt
touch "${shared_out}"
lock_file=./data/n100/dg/bkz${blocksize}_t${tours}.lock
touch "${lock_file}"

# Run experiment
python3 -u primalattack.py n=100 m=104 q=257 dist=DiscreteGaussian\(sqrt\(2/3\)\) \
    blocksize=${blocksize} tours=${tours} verbose=True \
    > "${perjob_out}" 2>&1

# Append only the final line to the shared file
last_line=$(tail -n 1 "${perjob_out}")
flock "${lock_file}" -c "echo \"${last_line}\" >> \"${shared_out}\""
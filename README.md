# beta-is-better

Artefact for Section 3 of the paper _"Refined Modelling of the Primal Attack, and Variants against Module-LWE"_, Paola de Perthuis and Filip Trenkić ([eprint](https://eprint.iacr.org/2025/2195))

`primalattack.py` runs the primal attack against a single instance of LWE. The LWE parameters, and strategy (progressive or fixed-blocksize), are specified on the command line:

```
python3 primalattack.py n=100 m=104 q=257 dist=DiscreteGaussian(sqrt(2/3)) progressive=True tours=1 verbose=True

python3 primalattack.py n=100 m=104 q=257 dist=DiscreteGaussian(sqrt(2/3)) blocksize=60 tours=5 verbose=True
```

`successprob.py` contains the functions `primal_success(...)` and `primal_success_progressive(...)` which estimate the simulated success probabilities of the primal attack. For example, for the two parameterisations above, the corresponding estimates are given by:

```
# Returns the cumulative mass function for the successful blocksize
p1 = primal_success_progressive(100, 104, 257, DiscreteGaussian(sqrt(2/3)), tours=1)

# Returns the success probability using this blocksize
p2 = primal_success(100, 104, 257, DiscreteGaussian(sqrt(2/3)), blocksize=60, tours=5)
```

## Installation

1. Make sure packages `gmp`, `mpfr`, `python`, `libtool`, `qd` are installed.
2. Build [g6k](https://github.com/fplll/g6k):
    ```
    git submodule add "https://github.com/fplll/g6k"
    cd g6k
    PYTHON=python3 ./bootstrap.sh
    cd ..
    ```
3. Add `scipy` to the virtual environment: `source g6k/activate; pip install scipy`.
4. To run the python scripts, make sure to activate the virutal environment each time.

## Reproducing experiments

The scripts to reproduce experiments from the paper are designed to run in parallel, utilising the full available resources of a compute cluster using the `slurm` scheduler.

To run all experiments from the paper:

```
# Progressive BKZ
sbatch exp_pbkz_n100_dg.sh
sbatch exp_pbkz_n100_tern.sh
sbatch exp_pbkz_n93_dg.sh
sbatch exp_pbkz_n93_cbin.sh

# Fixed-blocksize BKZ
sbatch exp_bkz_n100_dg.sh
sbatch exp_bkz_n100_tern.sh
sbatch exp_bkz_n93_dg.sh
sbatch exp_bkz_n93_cbin.sh
```

To collect all data:

```
python3 collect_data.py
```

This generates the CSV files
1. `data/beta_model_data.csv` collecting all experimental and simulated success probabilities,
2. `data/pbkz_statdist.csv` and `data/bkz_statdist.csv`, listing the total squared errors, and statistical distances, between experiments and simulations.

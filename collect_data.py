import csv
import successprob
from LWEgen import DiscreteGaussian, CentredBinary, Ternary
from math import sqrt
import numpy as np
from scipy.special import betainc
from math import floor

MAX_BLOCKSIZE = 70
PLOTS_BLOCKSIZE_RANGE = list(range(40, MAX_BLOCKSIZE+1))


PARAM_SETS = [
    ("n100", "dg", 100, 104, 257, DiscreteGaussian(sqrt(2/3))),
    ("n100", "tern", 100, 104, 257, Ternary()),
    ("n93", "dg", 93, 105, 257, DiscreteGaussian(1)),
    ("n93", "cbin", 93, 105, 257, CentredBinary()),
]

PROG_TRIALS = 1000
PROG_TOURS = [1, 5, 10, 15, 20]

BKZ_TRIALS = 200
BKZ_TOURS = [5, 10, 15, 20]
BKZ_BLOCKSIZE_RANGE = list(range(45, MAX_BLOCKSIZE+1))

OUT_FILE = "./data/beta_model_data.csv"

csv_cols = []
csv_cols.append(["blocksize", *PLOTS_BLOCKSIZE_RANGE])

PROG_SD_FILE = './data/pbkz_statdist.csv'
BKZ_SD_FILE = './data/bkz_statdist.csv'
prog_statdist_csv_cols = []
prog_statdist_csv_cols.append(["tours", *PROG_TOURS])
bkz_statdist_csv_cols = []
bkz_statdist_csv_cols.append(["tours", *BKZ_TOURS])


def cmf_to_pmf(cmf):
    if isinstance(cmf, dict):
        cmf_list = []
        prob_so_far = 0
        for i in range(1, MAX_BLOCKSIZE + 1):
            if not (i in cmf.keys()):
                cmf_list.append(prob_so_far)
            else:
                prob_so_far = cmf[i]
                cmf_list.append(cmf[i])
        return cmf_to_pmf(cmf_list)
    elif isinstance(cmf, list):
        pmf = []
        for i in range(len(cmf)):
            if i == 0:
                pmf.append(float(cmf[i]))
            else:
                pmf.append(float(cmf[i] - cmf[i-1]))
                assert cmf[i] - cmf[i-1] >= 0
        return pmf
    else:
        raise ValueError()

def stat_dist(pmf1, pmf2):
    assert len(pmf1) == len(pmf2)
    statdist = 0

    for i in range(len(pmf1)):
        statdist += abs(pmf1[i] - pmf2[i])

    pmf1_sum = sum(pmf1)
    pmf2_sum = sum(pmf2)
    # Assume worst case for missing probabilities
    statdist += abs(pmf1_sum - pmf2_sum)

    return statdist/2

def square_error(pmf1, pmf2):
    assert len(pmf1) == len(pmf2)
    err = 0

    for i in range(len(pmf1)):
        err += abs(pmf1[i] - pmf2[i])**2

    pmf1_sum = sum(pmf1)
    pmf2_sum = sum(pmf2)
    # Assume worst case for missing probabilities
    err += abs(pmf1_sum - pmf2_sum)**2

    return err

def binom_cmf(n, p, k):
    # proba that X \sim Bin(n,p) \leq k
    return betainc(n+floor(k), 1+floor(k), 1-p)


def expec_self_dist(n, exp_distrib):
    # return sum(exp_distrib[i]*((1-betainc(n-floor(exp_distrib[i]*n), floor(exp_distrib[i]*n), 1-exp_distrib[i]))/(1-betainc(n-floor(exp_distrib[i]*n), 1+floor(exp_distrib[i]*n), 1-exp_distrib[i])) - (betainc(n-floor(exp_distrib[i]*n), floor(exp_distrib[i]*n), 1-exp_distrib[i]))/ betainc(n-floor(exp_distrib[i]*n), 1+floor(exp_distrib[i]*n), 1-exp_distrib[i]))/4 if (1-exp_distrib[i])*exp_distrib[i]!=0 else 0 for i in range(len(exp_distrib)))
    # return sum(exp_distrib[i]*((1-betainc(n-floor(exp_distrib[i]*n), floor(exp_distrib[i]*n), 1-exp_distrib[i])) - (betainc(n-floor(exp_distrib[i]*n), floor(exp_distrib[i]*n), 1-exp_distrib[i])) - ((1-exp_distrib[i])**floor(n*exp_distrib[i])-(1-exp_distrib[i])**(n-floor(n*exp_distrib[i]))))/2 if (1-exp_distrib[i])*exp_distrib[i]!=0 else 0 for i in range(len(exp_distrib)))
    return sum(exp_distrib[i]*( betainc(n-floor(n*exp_distrib[i]), 1+floor(n*exp_distrib[i]), 1-exp_distrib[i]) - betainc(n-floor(n*exp_distrib[i]), floor(n*exp_distrib[i]), 1-exp_distrib[i]) ) if exp_distrib[i]*(1-exp_distrib[i])!=0 else 0 for i in range(len(exp_distrib)))

pv21_pmfs = {} # since pv21 estimator is blind to the distribution, cache the pBKZ pmfs
pv21_bkzprobs = {} # same for the bkz success probs

for n_str, dist_str, n, m, q, dist in PARAM_SETS:
    dir = f"./data/{n_str}/{dist_str}/"

    #============== Progressive-BKZ ===============#

    our_statdist_col_pbkz = [f'{n_str}_{dist_str}']
    pv21_statdist_col_pbkz = [f'{n_str}_{dist_str}_pv21']
    witness_exp_statdist_col_pbkz = [f'{n_str}_{dist_str}_witness']

    our_squareerr_col_pbkz = [f'{n_str}_{dist_str}_sq']
    pv21_squareerr_col_pbkz = [f'{n_str}_{dist_str}_sq_pv21']
    witness_exp_squareerr_col_pbkz = [f'{n_str}_{dist_str}_sq_witness']

    for tours in PROG_TOURS:

        print(f"Collecting exp_{n_str}_pbkz_t{tours}_{dist_str}")

        filename = dir + f"pbkz_t{tours}_all.txt"
        with open(filename, 'r') as file:
            lines = file.readlines()

        assert len(lines) == PROG_TRIALS

        # Read in the experimental data
        wins = []
        for line in lines:
            assert line.startswith("success = ")
            success = None
            exec(line)
            if success:
                wins.append(success)

        # Build the corresponding probability cmf and pmf
        cmf = [len([*filter(lambda x : x[1] <= blocksize, wins)])/PROG_TRIALS for blocksize in PLOTS_BLOCKSIZE_RANGE]
        pmf = [len([*filter(lambda x : x[1] == blocksize, wins)])/PROG_TRIALS for blocksize in range(1, MAX_BLOCKSIZE+1)]

        # Write to CSV
        csv_cols.append([f"exp_{n_str}_pbkz_t{tours}_{dist_str}", *cmf])

        # Run the estimator
        print(f"Collecting pred_{n_str}_pbkz_t{tours}_{dist_str}")
        pred_cmf = successprob.primal_success_progressive(n, m, q, dist, tours, max_blocksize = MAX_BLOCKSIZE)
        csv_cols.append([f"pred_{n_str}_pbkz_t{tours}_{dist_str}"]+['{:.6f}'.format(pred_cmf[b] if b in pred_cmf.keys() else 1) for b in PLOTS_BLOCKSIZE_RANGE])

        pred_pmf = cmf_to_pmf(pred_cmf)

        # Run the PV21 estimator
        if dist_str == "dg":
            print(f"Collecting pv21pred_{n_str}_pbkz_t{tours}_{dist_str}")
            pv21pred_cmf = successprob.primal_success_progressive_PV21(n, m, q, dist.sigma, tours, max_blocksize = MAX_BLOCKSIZE)
            pv21pred_pmf = cmf_to_pmf(pv21pred_cmf)
            pv21_pmfs[f"pv21pred_{n_str}_pbkz_t{tours}_{dist_str}"] = pv21pred_pmf
            csv_cols.append([f"pv21pred_{n_str}_pbkz_t{tours}_{dist_str}"]+['{:.6f}'.format(pv21pred_cmf[b] if b in pv21pred_cmf.keys() else 1) for b in PLOTS_BLOCKSIZE_RANGE])
        else:
            # already computed pv21 for discrete gaussian
            pv21pred_pmf = pv21_pmfs[f"pv21pred_{n_str}_pbkz_t{tours}_dg"]

        # Statistical distances and square errors
        our_statdist_col_pbkz.append(stat_dist(pmf, pred_pmf))
        pv21_statdist_col_pbkz.append(stat_dist(pmf, pv21pred_pmf))
        witness_exp_statdist_col_pbkz.append(expec_self_dist(PROG_TRIALS, pmf))
        
        our_squareerr_col_pbkz.append(square_error(pmf, pred_pmf))
        pv21_squareerr_col_pbkz.append(square_error(pmf, pv21pred_pmf))
        witness_exp_squareerr_col_pbkz.append(sum(pmf[i] * (1 - pmf[i]) for i in range(len(pmf)))/PROG_TRIALS)

    prog_statdist_csv_cols.append(our_statdist_col_pbkz)
    prog_statdist_csv_cols.append(pv21_statdist_col_pbkz)
    prog_statdist_csv_cols.append(witness_exp_statdist_col_pbkz)
    prog_statdist_csv_cols.append(our_squareerr_col_pbkz)
    prog_statdist_csv_cols.append(pv21_squareerr_col_pbkz)
    prog_statdist_csv_cols.append(witness_exp_squareerr_col_pbkz)

    #============== BKZ-beta ===============#

    our_statdist_col_bkz = [f'{n_str}_{dist_str}']
    pv21_statdist_col_bkz = [f'{n_str}_{dist_str}_pv21']
    witness_exp_statdist_col = [f'{n_str}_{dist_str}_witness']

    our_squareerr_col_bkz = [f'{n_str}_{dist_str}_sq']
    pv21_squareerr_col_bkz = [f'{n_str}_{dist_str}_sq_pv21']
    witness_exp_squareerr_col_bkz = [f'{n_str}_{dist_str}_sq_witness']

    for tours in BKZ_TOURS:
        print(f"Collecting pred_{n_str}_bkz_t{tours}_{dist_str}")
        pred = [successprob.primal_success(n, m, q, dist, blocksize, tours) for blocksize in PLOTS_BLOCKSIZE_RANGE]
        csv_cols.append([f"pred_{n_str}_bkz_t{tours}_{dist_str}"] + pred)

        if dist_str == "dg":
            print(f"Collecting pv21pred_{n_str}_bkz_t{tours}_{dist_str}")
            predpv21 = [successprob.primal_success_PV21(n, m, q, dist.sigma, blocksize, tours) for blocksize in PLOTS_BLOCKSIZE_RANGE]
            csv_cols.append([f"pv21pred_{n_str}_bkz_t{tours}_{dist_str}"] + predpv21)
            pv21_bkzprobs[f"pv21pred_{n_str}_bkz_t{tours}_{dist_str}"] = predpv21
        else:
            predpv21 = pv21_bkzprobs[f"pv21pred_{n_str}_bkz_t{tours}_dg"]

        print(f"Collecting exp_{n_str}_bkz_t{tours}_{dist_str}")

        exp = []
        for blocksize in PLOTS_BLOCKSIZE_RANGE:
            if not (blocksize in BKZ_BLOCKSIZE_RANGE):
                exp.append(0)
                continue
            filename = dir + f"bkz{blocksize}_t{tours}_all.txt"
            with open(filename, 'r') as file:
                lines = file.readlines()
            wins = 0
            for line in lines:
                assert line.startswith("success =")
                success = None
                exec(line)
                if success:
                    wins += 1
            exp.append(wins / BKZ_TRIALS)
        csv_cols.append([f"exp_{n_str}_bkz_t{tours}_{dist_str}"]+exp)

        # Compute the distances between experiments and predictions
        l = len(exp)
        assert l == len(predpv21) == len(pred) == len(PLOTS_BLOCKSIZE_RANGE)
        our_statdist_col_bkz.append(sum(abs(exp[i] - pred[i])/2 for i in range(l)))
        pv21_statdist_col_bkz.append(sum(abs(exp[i] - predpv21[i])/2 for i in range(l)))
        witness_exp_statdist_col.append(expec_self_dist(BKZ_TRIALS, exp))

        our_squareerr_col_bkz.append(sum(abs(exp[i] - pred[i])**2 for i in range(l)))
        pv21_squareerr_col_bkz.append(sum(abs(exp[i] - predpv21[i])**2 for i in range(l)))
        witness_exp_squareerr_col_bkz.append(sum(exp[i] * (1 - exp[i]) for i in range(l))/BKZ_TRIALS)

    bkz_statdist_csv_cols.append(our_statdist_col_bkz)
    bkz_statdist_csv_cols.append(pv21_statdist_col_bkz)
    bkz_statdist_csv_cols.append(witness_exp_statdist_col)
    bkz_statdist_csv_cols.append(our_squareerr_col_bkz)
    bkz_statdist_csv_cols.append(pv21_squareerr_col_bkz)
    bkz_statdist_csv_cols.append(witness_exp_squareerr_col_bkz)
    
assert all(len(x)==32 for x in csv_cols)
writer = csv.writer(open(OUT_FILE, 'w'))
writer.writerows(zip(*csv_cols))

writer = csv.writer(open(PROG_SD_FILE, 'w'))
writer.writerows(zip(*prog_statdist_csv_cols))

writer = csv.writer(open(BKZ_SD_FILE, 'w'))
writer.writerows(zip(*bkz_statdist_csv_cols))

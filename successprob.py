from BKZsim import initial_Z_shape_PV21, BKZsim_CN11
from scipy.stats import chi2, binom
from scipy.special import betainc, gamma
from math import sqrt, exp, log
import matplotlib.pyplot as plt
from numpy import array
import numpy as np
from functools import cache
from LWEgen import LWEDistribution, DiscreteGaussian, Ternary, CentredBinary

def projected_norm_cdf(full_dimension, full_norm, proj_dimension, proj_norm):
    """
    Cumulative density function for the projected norm (proj_norm) when a vector
    in dimension full_dimension, of norm full_norm, is projected onto a random subspace of
    dimension proj_dimension.
    """

    n = full_dimension
    m = proj_dimension
    v = full_norm
    x = proj_norm

    if x <= 0:
        return 0.0
    if x >= v:
        return 1.0

    alpha = m / 2.0
    beta = (n - m) / 2.0
    z = (x / v) ** 2

    return betainc(alpha, beta, z)

def primal_success_PV21(n, m, q, sigma, blocksize, tours, embedding_coeff = 1):
    """
    [PV21, Algorithm 6]
    """

    rank = n + m + 1
    profile = initial_Z_shape_PV21(q, rank, m, embedding_coeff)
    p_tot = 0

    for tour in range(tours):
        # Run 1 tour of BKZ simulator
        profile = BKZsim_CN11(profile, tours = 1, blocksize = blocksize)

        # Get prob chisquare is shorter
        scale = (sigma**2)
        quantile = exp(2 * profile[-blocksize]) / scale
        p_new = chi2.cdf(x=quantile,df=blocksize)
        p_tot += (1 - p_tot) * p_new

    return p_tot

def primal_success_progressive_PV21(n, m, q, sigma, tours, max_blocksize = None, embedding_coeff = 1):
    """
    [PV21, Algorithm 5]
    """

    p_tot = 0
    cmf = dict()

    rank = n + m + 1
    blocksize = 3

    if max_blocksize is None:
        max_blocksize = rank

    profile = initial_Z_shape_PV21(q, rank, m, embedding_coeff)

    while blocksize < 40:
        profile = BKZsim_CN11(profile, tours = tours, blocksize = blocksize)
        blocksize += 1

    p_tot = 0
    while blocksize <= max_blocksize:
        for tour in range(tours):
            # Run 1 tour of BKZ simulator
            profile = BKZsim_CN11(profile, tours = 1, blocksize = blocksize)

            # Get prob chisquare is shorter
            scale = (sigma**2)
            quantile = exp(2 * profile[-blocksize] - log(scale))
            p_new = chi2.cdf(x=quantile,df=blocksize)
            p_tot += (1 - p_tot) * p_new

        cmf[blocksize] = p_tot

        if p_tot >= 0.9999:
            for _ in range(blocksize + 1, max_blocksize + 1):
                cmf[_] = 1
            break

        blocksize += 1

    return cmf

@cache
def primal_success_fixednorm(n, m, q, v_norm2, blocksize, tours, embedding_coeff = 1):
    rank = n + m + 1
    v_norm = sqrt(v_norm2)
    profile = initial_Z_shape_PV21(q, rank, m, embedding_coeff)
    p_tot = 0

    for tour in range(tours):
        # Run 1 tour of BKZ simulator
        profile = BKZsim_CN11(profile, tours = 1, blocksize = blocksize)

        quantile = exp(profile[-blocksize])
        p_new = projected_norm_cdf(rank, v_norm, blocksize, quantile)
        p_tot += (1 - p_tot) * p_new

    return p_tot

def primal_success(n, m, q, lwe_distribution : LWEDistribution, blocksize, tours):
    rank = n + m + 1

    vnorm2_pmf = lwe_distribution.squarednorm_pmf(n+m)
    success_per_vnorm2 = {vnorm2 : 0 for vnorm2 in vnorm2_pmf.keys()}

    profile = initial_Z_shape_PV21(q, rank, m, lwe_distribution.embedding_coeff())

    for tour in range(tours):
        # Run 1 tour of BKZ simulator
        profile = BKZsim_CN11(profile, tours = 1, blocksize = blocksize)

        # Update probabilities
        quantile = exp(profile[-blocksize])

        for vnorm2, p_tot in success_per_vnorm2.items():
            p_new = projected_norm_cdf(rank, sqrt(vnorm2), blocksize, quantile)
            success_per_vnorm2[vnorm2] += (1 - p_tot) * p_new


    sum_result = 0
    for vnorm2, prob in vnorm2_pmf.items():
        sum_result += prob * success_per_vnorm2[vnorm2]

    return sum_result

def primal_success_progressive(n, m, q, lwe_distribution : LWEDistribution, tours, max_blocksize = None, thresh = 0.999):
    rank = n + m + 1

    if max_blocksize is None:
        max_blocksize = rank

    vnorm2_pmf = lwe_distribution.squarednorm_pmf(n+m)

    # Table where keys are ||v||^2 and entires are p_tot
    p_tot_table = {vnorm2 : 0 for vnorm2 in vnorm2_pmf.keys()}
    # Table where keys are ||v||^2 and entires are cmfs for blocksize
    cmf_table = {vnorm2 : dict() for vnorm2 in vnorm2_pmf.keys()}

    profile = initial_Z_shape_PV21(q, rank, m, lwe_distribution.embedding_coeff())


    blocksize = 3
    while blocksize < 40:
        profile = BKZsim_CN11(profile, tours = tours, blocksize = blocksize)
        blocksize += 1

    loop = True
    while loop and blocksize <= max_blocksize:
        for tour in range(tours):
            profile = BKZsim_CN11(profile, tours = 1, blocksize = blocksize)

            quantile = exp(profile[-blocksize])

            # Update the pmfs for each vnorm2, counting how many have reached threshold
            num_above_thresh = 0
            for vnorm2, p_tot in p_tot_table.items():
                if p_tot >= thresh:
                    num_above_thresh += 1
                    continue

                p_new = projected_norm_cdf(rank, sqrt(vnorm2), blocksize, quantile)
                p_tot += (1 - p_tot) * p_new
                p_tot_table[vnorm2] = p_tot

        # If all above threshold then we are done
        if num_above_thresh == len(vnorm2_pmf.items()):
            loop = False

        # Update the cmfs
        for vnorm2, cmf in cmf_table.items():
            cmf[blocksize] = p_tot_table[vnorm2]

        blocksize += 1

    blocksize_range = range(40,blocksize)
    avg_cmf = dict()

    # Average the cmfs for each vnorm2
    for blocksize in blocksize_range:
        avg = 0
        for vnorm2, cmf in cmf_table.items():
            if blocksize in cmf.keys():
                avg += cmf[blocksize] * vnorm2_pmf[vnorm2]
        avg_cmf[blocksize] = avg

    return avg_cmf



if __name__ == '__main__':
    n = 93
    m = 105
    q = 257

    colors = {
        1 : "purple",
        5 : 'red',
        10 : "#C46800",
        15 : 'green',
        20 : 'black',
        30 : 'blue'
    }

    experimental = {}

    for tours in [1, 5, 10, 15, 20]:

        x_axis = range(40, 71)
        pv21 = primal_success_progressive_PV21(n, m, q, 1, tours, max_blocksize = 70)
        ours = primal_success_progressive(n, m, q, DiscreteGaussian(1), tours, max_blocksize = 70)

        plt.plot(pv21.keys(), pv21.values(), ':', label = f"pv21 {tours}", color = colors[tours])
        plt.plot(ours.keys(), ours.values(), '--', label = f"ours {tours}", color = colors[tours])

        if tours in experimental.keys():
            data = experimental[tours]
            samples = 1000
            plt.plot(data.keys(), [v/samples for v in data.values()], 'x', label = f"experiment", color = colors[tours])

        #if tours in experimental.keys():
        #    plt.plot(x_axis, experimental[tours], 'x', label = f"experimental {tours}", color = colors[tours])

    #pv21_1 = {45: 0.000273943847335349, 46: 0.000627351173392779, 47: 0.00150811599509220, 48: 0.00344011620682035, 49: 0.00729728356408436, 50: 0.0144379141924790, 51: 0.0268422973752590, 52: 0.0472178460553758, 53: 0.0789998231397309, 54: 0.126125060231109, 55: 0.192411625756455, 56: 0.280395372626043, 57: 0.389651967409082, 58: 0.515058577462635, 59: 0.646040672910428, 60: 0.768133016984571, 61: 0.867364610995125, 62: 0.935804575399678, 63: 0.974637070709936, 64: 0.992142271443691, 65: 0.998174078300678, 66: 0.999696892020078, 67: 1, 68: 1, 69: 1, 70: 1}
    #pv21_5 = {45: 0.00943967817217647, 46: 0.0229214497110410, 47: 0.0492531815273277, 48: 0.0897216752354710, 49: 0.147026112688856, 50: 0.224646460137345, 51: 0.325181338242616, 52: 0.447900860928648, 53: 0.585922880498910, 54: 0.724703216583440, 55: 0.844721311964474, 56: 0.929754932259813, 57: 0.976244993350728, 58: 0.994491310459706, 59: 0.999210358772763, 60: 1, 61: 1, 62: 1, 63: 1, 64: 1, 65: 1, 66: 1, 67: 1, 68: 1, 69: 1, 70: 1}

    #plt.plot(pv21_1.keys(), pv21_1.values(), label = "Real pv21 1")
    #plt.plot(pv21_5.keys(), pv21_5.values(), label = "Real pv21 5")

    plt.grid(color='#eeeeee', linestyle='-', linewidth=2)
    plt.legend(loc = "upper left")

    plt.show()
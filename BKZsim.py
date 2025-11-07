from numpy import array, zeros, copy, allclose, concatenate
from scipy.special import betainc, gamma
import math
from math import log, sqrt, lgamma
import matplotlib.pyplot as plt

# Rank where heuristics kick in. Experimental data used for smaller ranks.
heur_rank = 45

# Taken from [PV21]. 
# Note they use base-2 logarithms, we use base-e, so we multiply by log(2) to account for this.
hkz_profile_45 = log(2) * array((
    0.789527997160000,
    0.780003183804613,
    0.750872218594458,
    0.706520454592593,
    0.696345241018901,
    0.660533841808400,
    0.626274718790505,
    0.581480717333169,
    0.553171463433503,
    0.520811087419712,
    0.487994338534253,
    0.459541470573431,
    0.414638319529319,
    0.392811729940846,
    0.339090376264829,
    0.306561491936042,
    0.276041187709516,
    0.236698863270441,
    0.196186341673080,
    0.161214212092249,
    0.110895134828114,
    0.0678261623920553,
    0.0272807162335610,
    -0.0234609979600137,
    -0.0320527224746912,
    -0.0940331032784437,
    -0.129109087817554,
    -0.176965384290173,
    -0.209405754915959,
    -0.265867993276493,
    -0.299031324494802,
    -0.349338597048432,
    -0.380428160303508,
    -0.427399405474537,
    -0.474944677694975,
    -0.530140672818150,
    -0.561625221138784,
    -0.612008793872032,
    -0.669011014635905,
    -0.713766731570930,
    -0.754041787011810,
    -0.808609696192079,
    -0.859933249032210,
    -0.884479963601658,
    -0.886666930030433,
))

def BKZsim_CN11(profile, tours, blocksize):
    """
    [CN11, Algorithm 2] with additional functionality to allow blocksize<45
    """
    rank = len(profile)

    # Memoize log-gaussian heuristic for dimension up to blocksize
    gh = [0] + [hkz_profile_45[-i] - sum(hkz_profile_45[-i:])/i for i in range(1, heur_rank)]
    gh += [log_gaussian_heuristic(d) for d in range(heur_rank, blocksize+1)]

    for _ in range(tours):
        new_profile = copy(profile)
        phi = True
        terminal = min(heur_rank, blocksize)

        for k in range(rank - terminal):
            local_dim = min(blocksize, rank - k)
            f = min(k + blocksize - 1, rank - 1)
            local_ln_V = sum(profile[:f+1]) - sum(new_profile[:k])

            pred = gh[local_dim] + local_ln_V/local_dim

            if phi:
                if pred < profile[k]:
                    new_profile[k] = pred
                    phi = False
            else:
                new_profile[k] = pred
        
        if phi:
            break
        
        # Account for terminal block
        local_ln_V = sum(profile) - sum(new_profile[:-terminal])
        new_profile[-terminal:] = (hkz_profile_45[-terminal:]-sum(hkz_profile_45[-terminal:]/terminal)) + (local_ln_V/terminal)

        assert allclose(sum(new_profile), sum(profile))
        profile = copy(new_profile)
    
    return profile

def initial_Z_shape_PV21(q, rank, m, embedding_coeff):
    """
    [PV21, Algorithm 7]
    """
    deltaLLL = 1.0219
    log_alpha = -2 * log(deltaLLL)
    log_vol = m * log(q) + log(embedding_coeff)

    slope = [log(q) + i * log_alpha for i in range(1, int(log(q)/-log_alpha) + 1)]
    assert slope[-1] >= 0
    assert slope[-1] + log_alpha < 0

    # Edge case: no head or tail
    if len(slope) > rank:
        slope = array(slope[:rank])
        slope += (log_vol -sum(slope))/len(slope)
        return slope

    l = len(slope)
    v = sum(slope)
    head = []

    while v + sum(head) + log(q) < m * log(q) and l + len(head) < rank:
        head.append(log(q))
    
    l = l + len(head)
    v = v + sum(head)

    tail = []
    while l + len(tail) < rank:
        tail.append(0)

    slope = array(slope)
    slope += (log_vol - sum(head)-sum(tail)-sum(slope))/len(slope)

    assert allclose(sum(concatenate([head,slope,tail])), log_vol)
    return concatenate([head, slope, tail])

def log_gaussian_heuristic(dim):
    return (lgamma(dim / 2.0 + 1) * (1.0 / dim) - (1/2)*log(math.pi))
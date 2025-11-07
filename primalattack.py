import sys

sys.path.insert(1, './g6k')

import fpylll
from fpylll import IntegerMatrix, LLL
from fpylll.fplll.bkz_param import BKZParam
from fpylll.algorithms.bkz2 import BKZReduction
from numpy import array, concatenate, zeros, identity, allclose
from LWEgen import gen_lwe_instance, DiscreteGaussian, CentredBinary, Ternary
import time
from math import sqrt

def construct_BaiGalbraith_basis(n, m, q, A, b, embedding_coeff):
    """
    Given a LWE instance (assuming short secret and error)
    Construct the Bai-Galbraith basis
     (qI_m   -A^T    b   )
     (        I_n        )
     (               e.c.)
    or more precisely, since we are working in column form, its transpose.
    """
    M = zeros((n+m+1, n+m+1), dtype=int)

    M[:m,:m] = q*identity(m, dtype=int)
    M[m:, m:] = identity(n+1, dtype=int)
    M[-1, -1] = embedding_coeff
    M[m:m+n,:m] = -A.T
    M[-1, :m] = b

    return M

def test_primal_success(n, m, q, A, s, e, b, embedding_coeff, blocksize, tours, ft = "ld"):
    """
    Run the primal attack on a LWE instance.
    If embedded vector found on tour t, returns (True, t)
    otherwise, returns False
    """

    # Note down the secret vector and its norm
    v = concatenate([e, s, [embedding_coeff]])
    v_norm2 = v @ v
    logmsg(f"v_norm2={v_norm2}")

    # Generate the Bai-Galbraith basis
    B = construct_BaiGalbraith_basis(n, m, q, A, b, embedding_coeff)

    params_fplll = BKZParam(block_size=blocksize, strategies=fpylll.BKZ.DEFAULT_STRATEGY,
                            flags=0 | fpylll.BKZ.AUTO_ABORT | fpylll.BKZ.GH_BND | fpylll.BKZ.MAX_LOOPS,
                            max_loops=1)

    try:
        M = fpylll.GSO.Mat(IntegerMatrix.from_matrix(B), float_type=ft)
        bkz = BKZReduction(M)

        for _ in range(tours):
            logmsg(f"Tour {_+1}")

            # Run one tour BKZ, then LLL. Record time taken.
            tourstart = time.time()
            bkz.tour(params_fplll)
            LLL.reduction(M.B)
            tourend = time.time()
            logmsg(f"Time: {tourend - tourstart}")

            # Succeed if a vector of length same as v is found anywhere in the basis
            for i in range(M.B.nrows):
                bi = array(M.B[i])
                if (bi @ bi) == v_norm2:
                    logmsg(f"Found embedded vector at position {i}")
                    return True, _+1
    except Exception as err:
        if str(err) in ["b'infinite loop in babai'", "math domain error", "b'success'"]:
            new_ft = next_float_type(ft)
            if new_ft is None:
                # If plagued by precision errors, rerandomise the instance
                A,s,e,b = gen_lwe_instance(n, m, q, dist)
                new_ft = "ld"
                logmsg("New instance")
            else:
                logmsg(f"Increasing precision to {new_ft}")
            return test_primal_success(n, m, q, A, s, e, b, embedding_coeff, blocksize, tours, ft = new_ft)
        else:
            raise err

    return False

def test_primal_success_progressive(n, m, q, A, s, e, b, embedding_coeff, tours, max_blocksize = None, ft = "ld"):
    """
    Run primal attack with pBKZ on LWE instance.
    If embedded vector found on tour t of blocksize b, returns (True, b, t)
    otherwise returns False
    """
    if max_blocksize is None:
        max_blocksize = n + m + 1

    # Note down the secret vector and its norm
    v = concatenate([e, s, [embedding_coeff]])
    v_norm2 = v @ v
    logmsg(f"v_norm2={v_norm2}")

    # Generate the Bai-Galbraith basis
    B = construct_BaiGalbraith_basis(n, m, q, A, b, embedding_coeff)

    params_fplll = {'strategies':fpylll.BKZ.DEFAULT_STRATEGY,
                    'flags':0 | fpylll.BKZ.AUTO_ABORT | fpylll.BKZ.GH_BND | fpylll.BKZ.MAX_LOOPS,
                    'max_loops': tours}

    try:
        M = fpylll.GSO.Mat(IntegerMatrix.from_matrix(B), float_type=ft)
        bkz = BKZReduction(M)
        for blocksize in range(3, max_blocksize + 1):
            logmsg(f"Blocksize {blocksize}")
            for _ in range(tours):
                logmsg(f"Tour {_+1}")
                tourstart = time.time()
                bkz.tour(BKZParam(block_size=blocksize, **params_fplll))
                LLL.reduction(M.B)
                tourend = time.time()
                logmsg(f"Time: {tourend - tourstart}")

                # Succeed if a vector of length same as v is found anywhere in the basis
                for i in range(M.B.nrows):
                    bi = array(M.B[i])
                    if allclose(bi @ bi, v_norm2):
                        logmsg(f"Found embedded vector at position {i}")
                        return True, blocksize, _+1
    except Exception as err:
        if str(err) in ["b'infinite loop in babai'", "math domain error", "b'success'"]:
            new_ft = next_float_type(ft)
            if new_ft is None:
                # If plagued by precision errors, rerandomise the instance
                A,s,e,b = gen_lwe_instance(n, m, q, dist)
                new_ft = "d"
                logmsg("New instance")
            else:
                logmsg(f"Increasing precision to {new_ft}")
            return test_primal_success_progressive(n, m, q, A, s, e, b, embedding_coeff, tours, max_blocksize, ft = new_ft)
        else:
            raise err
    return False

def next_float_type(curr_ft):
    if curr_ft == "d":
        return "ld"
    elif curr_ft == "ld":
        return "dd"
    elif curr_ft == "dd":
        return "qd"
    elif curr_ft == "qd":
        fpylll.FPLLL.set_precision(300)
        return "mpfr"
    elif curr_ft == "mpfr":
        return None

n = 100
m = 104
q = 257
dist = DiscreteGaussian(sqrt(2/3))
blocksize = 40
progressive = False
tours = 5
ft = "d"
max_blocksize = None

verbose = False
def logmsg(*args, override = False, **kwargs):
    if verbose or override:
        print(*args, **kwargs)

# Get command line arguments
for arg in sys.argv[1:]:
    exec(arg)

if __name__ == '__main__':
    logmsg(f"Primal attack experiment: n={n}, m={m}, q={q}, dist={dist}")
    if progressive:
        logmsg(f"Strategy: Progressive-BKZ, tours={tours}, max_blocksize={max_blocksize}")
    else:
        logmsg(f"Strategy: BKZ, blocksize={blocksize}, tours={tours}")

    A,s,e,b = gen_lwe_instance(n, m, q, dist)
    try:
        start_time = time.time()
        if progressive:
            experiment = test_primal_success_progressive(n, m, q, A, s, e, b, embedding_coeff = dist.embedding_coeff(), tours = tours, max_blocksize = max_blocksize, ft = ft)
        else:
            experiment = test_primal_success(n, m, q, A, s, e, b, embedding_coeff = dist.embedding_coeff(), blocksize = blocksize, tours = tours, ft = ft)
        end_time = time.time()
        logmsg(f"minutes = {(end_time - start_time)/60}")
        logmsg(f"success = {experiment}", override = True)
    except Exception as err:
        logmsg(f"Experiment crashed! {err}", override = True)
        raise err

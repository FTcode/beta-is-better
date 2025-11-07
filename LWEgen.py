from numpy import random, array, ceil, arange, exp, allclose
from scipy.stats import binom, chi2
from math import sqrt

class LWEDistribution:
    """
    Base class for a distribution of the integers used to sample vectors (coordinate-wise) for LWE.
    """

    def sample_vector(self, dimension : int) -> array:
        """
        Sample a vector from the distribution, of dimension d
        """
        pass

    def embedding_coeff(self) -> int:
        """
        Which embedding coefficient to use in Bai-Galbraith
        (positive integer closest to s.d. of each coordinate)
        """
        return 1

    def squarednorm_pmf(self, dimension : int, tol : float = 0.001) -> dict:
        """
        Probability mass function for the squared norm of the size (dimension+1) embedded vector (e | s | embedding_coeff)
        when e,s are sampled from this distribution, with argument dimension=dim(e)+dim(s)
        If the distribution has tails, only consider region with (1 - 2*tol) of probability.
        """
        pass

    def __str__(self) -> str:
        pass

class CentredBinary(LWEDistribution):
    def __init__(self):
        self.sigma = 1

    def sample_vector(self, dimension):
        return random.choice([-1, 1], dimension)

    def squarednorm_pmf(self, dimension, tol = 0):
        return {dimension+1: 1}

    def __str__(self):
        return "CentredBinary()"

class Ternary(LWEDistribution):
    def __init__(self):
        self.sigma = 1

    def sample_vector(self, dimension):
        return random.choice([-1, 0, 1], dimension)

    def squarednorm_pmf(self, dimension, tol = 0.001):
        pmf = dict()

        # Define parameters for binomial distribution
        loc = 1
        trials = dimension
        trial_prob = 2/3

        # Compute tails: only sum over region which contains (1 - 2 * tol) of probability
        lower_tail = int(binom.ppf(q = tol, n = trials, p = trial_prob, loc = loc))
        upper_tail = int(binom.isf(q = tol, n = trials, p = trial_prob, loc = loc))

        for v_norm2 in range(lower_tail, upper_tail + 1):
            if v_norm2 == lower_tail:
                pmf[v_norm2] = binom.cdf(k = v_norm2, n = trials, p = trial_prob, loc = loc)
            elif v_norm2 == upper_tail:
                pmf[v_norm2] = 1 - binom.cdf(k = v_norm2 - 1, n = trials, p = trial_prob, loc = loc)
            else:
                pmf[v_norm2] = binom.pmf(k = v_norm2, n = trials, p = trial_prob, loc = loc)

        assert allclose(sum(pmf.values()), 1)
        return pmf

    def __str__(self):
        return "Ternary()"

class DiscreteGaussian(LWEDistribution):
    def __init__(self, sigma, cutoff = 15):
        """
        sigma is the standard deviation (such that sigma**2 is variance)
        cutoff is the number of standard deviations to consider as the support
        """
        self.sigma = sigma

        # Define support range
        max_k = int(ceil(cutoff * sigma))
        support = arange(-max_k, max_k + 1)

        # Compute probabilities
        probs = exp(-support**2 / (2 * sigma**2))
        probs /= probs.sum()  # normalize
        probs = probs.tolist()

        self.probs = probs
        self.support = support

    def sample_vector(self, dimension):
        return random.choice(self.support, p=self.probs, size=dimension)

    def squarednorm_pmf(self, dimension, tol = 0.001):
        pmf = dict()

        # Parameters for chi-squared distribution
        df = dimension
        scale = self.sigma**2
        loc = self.embedding_coeff()**2

        # Compute tails: only sum over region which contains (1 - 2 * tol) of probability
        lower_tail = int(chi2.ppf(q = tol, df = df, scale = scale, loc = loc))
        upper_tail = int(chi2.isf(q = tol, df = df, scale = scale, loc = loc))

        for v_norm2 in range(lower_tail, upper_tail + 1):
            if v_norm2 == lower_tail:
                pmf[v_norm2] = chi2.cdf(v_norm2 + 1/2, df=df, scale=scale, loc=loc)
            elif v_norm2 == upper_tail:
                pmf[v_norm2] = 1 - chi2.cdf(v_norm2 - 1/2, df=df, scale=scale, loc=loc)
            else:
                pmf[v_norm2] = chi2.cdf(v_norm2 + 1/2, df=df, scale=scale, loc=loc) - chi2.cdf(v_norm2 - 1/2, df=df, scale=scale, loc=loc)

        assert allclose(sum(pmf.values()), 1)
        return pmf

    def embedding_coeff(self):
        return max(round(self.sigma), 1)

    def __str__(self):
        return f"DiscreteGaussian({self.sigma})"


def gen_lwe_instance(n, m, q, distribution):
    """
    Generate an LWE instance in normal form (secret and error follow the same distribution.)
    """
    A = random.randint(low = 0, high = q, size = (m, n))
    s = distribution.sample_vector(n)
    e = distribution.sample_vector(m)

    b = s @ A.T + e
    b %= q

    return A, s, e, b
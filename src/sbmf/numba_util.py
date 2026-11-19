"""
numba_util.py  --  scalar numba helpers shared by the E-step kernels
====================================================================

Split out of bae_util so the compiled search (bae_search) depends only on
numpy + numba.  Previously `bae_search` imported `bae_util` for these two
functions, which dragged torch, torchvision, scikit-learn, cvxpy, networkx and
matplotlib into every import of the kernel.  `bae_util` re-exports everything
here, so `bae_util.log_ndtr` etc. keep working.
"""

import math

import numpy as np
from numba import njit


@njit
def _seed_numba(s):
    # numba keeps its OWN global RNG, untouched by Python's np.random.seed; the
    # E-step search kernels (old_bae_search / bae_search) draw their stochastic
    # flips from it, so this must be seeded too for runs to be paired across ranks.
    np.random.seed(s)


@njit
def log_ndtr(x: float) -> float:
    """Robust log standard normal CDF for Numba."""
    if x > -5.0:
        return math.log(0.5 * math.erfc(-x / math.sqrt(2.0)))
    else:
        return -0.5 * x**2 - math.log(-x) - 0.5 * math.log(2 * math.pi)

@njit
def sample_trunc_norm(mu: float, nu: float) -> float:
    """Samples from a normal distribution truncated to [0, inf).

    Hardened against the degenerate inputs that show up in spike-and-slab fits:
    when a weight column dies (gram diagonal -> 0) or a structured prior drives
    the effective precision non-positive, the caller's nu = sqrt(sigma2/wjj) and
    mu = .../wjj come through as inf/NaN.  A NaN mu fails `mu >= 0.0`, falls into
    the mu<0 branch, and its accept test `log(rand()) <= NaN` is never true -- so
    the original unbounded `while True` spins forever inside the (numba) E-step
    and forces a kernel restart.  Here a non-positive/non-finite nu or non-finite
    mu has no proper truncated normal, so we return the mode (max(mu, 0)); and
    both rejection loops are capped at `max_tries` with the same mode fallback, so
    the sampler can never stall the search.  For well-conditioned inputs the loops
    accept far inside the cap, leaving behaviour identical to before."""
    max_tries = 10000          # local constant (keep the 2-arg signature: snmf
                               # dispatches sample/mode/mean through one njit
                               # function pointer, so all three must match arity)
    # Degenerate scale/location -> no proper truncated normal; return the mode.
    if not (nu > 0.0) or not math.isfinite(mu):
        return mu if (math.isfinite(mu) and mu > 0.0) else 0.0

    if mu >= 0.0:
        for _ in range(max_tries):
            z = mu + nu * np.random.randn()
            if z >= 0.0:
                return z
        return mu                                    # mode fallback (mu >= 0)
    else:
        alpha_opt = (-mu + math.sqrt(mu**2 + 4 * nu**2)) / (2 * nu**2)
        z_max = mu + alpha_opt * nu**2
        log_rho_max = -(z_max - mu)**2 / (2 * nu**2) + alpha_opt * z_max
        for _ in range(max_tries):
            z = np.random.exponential(1.0 / alpha_opt)
            log_rho = -(z - mu)**2 / (2 * nu**2) + alpha_opt * z
            if math.log(np.random.rand()) <= log_rho - log_rho_max:
                return z
        return 0.0                                   # mode fallback (mu < 0)

@njit
def mode_trunc_norm(mu: float, nu: float) -> float:
    """Mode of a normal distribution truncated to [0, inf)."""
    if mu >= 0.0:
        return mu
    else:
        return 0.0

@njit
def mean_trunc_norm(mu: float, nu: float) -> float:
    """Mean of a normal distribution truncated to [0, inf)."""
    
    alpha = mu / nu

    Z = math.sqrt(2 * math.pi) * 0.5 * ( 1 + math.erf(alpha / math.sqrt(2.0)))
    return mu + nu * math.exp(-0.5 * alpha**2) / Z



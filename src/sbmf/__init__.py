"""sbmf -- sparse and semi-binary matrix factorization.

Binary autoencoder / binary matrix factorization models (SemiBMF, JBMF, BiPCA,
SCPD, RRBMF, KernelBMF), their compiled E-step kernels and priors, and the
simulation / analysis utilities in `df_util` and `bae_util`.

Submodules load lazily on first access, so `import sbmf` is instant:

    from sbmf.bae_models import BiPCA, SemiBMF
    from sbmf.bae_util import impcv, bicv
    from sbmf import df_util

Import cost (heaviest dependency each pulls in):
    util, numba_util, bae_search, bae_priors   numpy / scipy / networkx / numba
    bae_weights, bae_models                    + torch
    df_util, bae_util                          + torch, cvxpy, scikit-learn
"""

import importlib

__version__ = "0.1.0"

_SUBMODULES = (
    "bae_models", "bae_weights", "bae_priors", "bae_search", "numba_util",
    "bae_util", "df_util", "util",
)

__all__ = list(_SUBMODULES)


def __getattr__(name):
    if name in _SUBMODULES:
        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals()) + list(_SUBMODULES))

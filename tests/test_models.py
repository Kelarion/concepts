"""Small end-to-end fits + regression tests for the bugs fixed in the audit."""
import numpy as np
import pytest

from sbmf import bae_models, bae_util
from sbmf.bae_priors import AdaptiveTemp


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    S = (rng.random((60, 3)) > 0.5).astype(float)
    W = rng.standard_normal((12, 3))
    X = S @ W.T + 0.1 * rng.standard_normal((60, 12))
    return X / X.std()


FIT = dict(max_iter=40, verbose=False)


def test_semibmf_fits_and_compiles_once(data):
    mod = bae_models.SemiBMF(3, tree_reg=0.0)
    en = mod.fit(data, lr=0.1, scl_lr=1e-2, **FIT)
    assert len(en) == 40 and np.isfinite(en).all() and en[-1] < en[0]
    # sigma_x used to start as the int 1 -> the kernel compiled twice
    assert len(mod.operator._kernel.signatures) == 1


def test_bipca_multichain_collapse_with_adaptive_temp(data):
    mod = bae_models.BiPCA(3, n_chains=3)
    mod.fit(data, lr=0.25, prior_schedule=AdaptiveTemp(), **FIT)
    assert np.ndim(mod.latent_prior.temp) == 1          # per-chain temperature
    mod.collapse()
    assert mod.S.shape == (60, 3)
    mod.sample(data, n_samp=2, burnin=2)                # used to raise TypeError


def test_impcv_and_bicv_run(data):
    tr, te = bae_util.impcv(bae_models.SemiBMF(2, tree_reg=0.0), data, folds=3,
                            seed=0, lr=0.1, **FIT)
    assert np.isfinite([tr, te]).all()
    tr, te = bae_util.gabriel_bicv(bae_models.SemiBMF(2, tree_reg=0.0), data, seed=0,
                                   initial_temp=1, decay_rate=0.5, period=2,
                                   fold_iter=5, n_samp=2, lr=0.1)
    assert np.isfinite([tr, te]).all()


@pytest.mark.parametrize("kernel_input", [False, True])
def test_kernelbmf_loss_matches_mstep_energy(data, kernel_input):
    X = data @ data.T if kernel_input else data
    mod = bae_models.KernelBMF(3, tree_reg=0.0, kernel_input=kernel_input)
    en = mod.fit(X, **FIT)
    assert np.isfinite(en).all()
    # loss() and the M-step's return value are the same normalized energy; the
    # feature branch of loss() had a centering bug and a precedence bug.
    energy = mod.MStep(X, mod.S, learn_prior=False)
    assert mod.loss(X) == pytest.approx(energy, rel=1e-8)


def test_fit_rejects_undefined_schedule(data):
    with pytest.raises(ValueError, match="max_iter"):
        bae_models.SemiBMF(2).fit(data, max_iter=None, verbose=False)


def test_missing_optional_module_message():
    from sbmf.bae_priors import mrf
    try:
        mrf.build_fields
    except ModuleNotFoundError as e:                    # not shipped by default
        assert "mrf_samplers.py" in str(e)

"""The sparse E-step kernels: tree-table decomposition, batched vs exact sampling,
and the StS bookkeeping."""
import numpy as np
import pytest
from numba import njit

from sbmf import bae_search as bs
from sbmf.numba_util import _seed_numba


@njit
def _exact_inhib(S, i, j, StS, N):
    return bs._tree_inhib(S, i, j, StS, N, 1.0)


def test_tree_tables_match_exact_term():
    rng = np.random.default_rng(0)
    n, m = 200, 15
    S = (rng.random((n, m)) < rng.uniform(0.05, 0.6, m)).astype(float)
    StS = S.T @ S
    base, D = np.zeros((2, m)), np.zeros((2, m, m))
    bs._tree_tables(StS, float(n), base, D)
    for i in range(n):
        act = np.flatnonzero(S[i])
        for j in range(m):
            s = int(S[i, j])
            assert base[s, j] + D[s, j, act].sum() == _exact_inhib(S, i, j, StS, float(n))


def _problem(seed=0, n=300, d=40, m=12):
    rng = np.random.default_rng(seed)
    S0 = (rng.random((n, m)) < 0.2).astype(float)
    W = rng.standard_normal((d, m)) / np.sqrt(d) * 2
    X = S0 @ W.T + rng.standard_normal((n, d))
    Jc = rng.normal(0, 0.5, (m, m)); Jc = (Jc + Jc.T) / 2; np.fill_diagonal(Jc, 0)
    return S0, X @ W, W.T @ W, Jc, rng.normal(0, 0.5, m)


def _marginals(tree_batch, seed, sweeps=300, burn=50):
    S0, XW, WtW, Jc, hc = _problem()
    k = bs.make_dense_search(*bs.BINARY_LINK, bs.PRIOR_BOLTZMANN)
    np.random.seed(seed); _seed_numba(seed)
    S = S0.copy(); StS = S.T @ S; acc = np.zeros_like(S)
    for it in range(sweeps):
        k(XW, S, S, WtW, StS, len(S), 1.0, 0.3, 1.0, 0.0, 1.5, Jc, hc, True, None, 1.5,
          tree_batch)
        if it >= burn:
            acc += S
    assert np.array_equal(StS, S.T @ S)                  # live StS stays exact
    return acc / (sweeps - burn)


def test_batched_tree_matches_exact_marginals():
    exact_a, exact_b = _marginals(0, 1), _marginals(0, 2)
    batched = _marginals(20, 3)
    floor = np.abs(exact_a - exact_b).mean()
    assert np.abs(batched - exact_a).mean() < 1.5 * floor
    assert abs(batched.mean() - exact_a.mean()) < 0.01


def test_sweep_moves_and_depends_on_seed():
    S0, XW, WtW, Jc, hc = _problem()
    k = bs.make_dense_search(*bs.BINARY_LINK, bs.PRIOR_PLAIN)
    out = []
    for seed in (1, 1, 2):
        np.random.seed(seed); _seed_numba(seed)
        S = S0.copy()
        k(XW, S, S, WtW, S.T @ S, len(S), 1.0, 0.0, 1.0, 0.0, 1.0, Jc, hc, True, None, 1.0, 20)
        out.append(S)
    assert not np.array_equal(out[0], S0)
    assert np.array_equal(out[0], out[1]) and not np.array_equal(out[0], out[2])


@pytest.mark.parametrize("kernel_input", [False, True])
def test_kernel_scaffold_keeps_StS(kernel_input):
    S0, XW, WtW, Jc, hc = _problem(n=80, m=6)
    rng = np.random.default_rng(1)
    X = rng.standard_normal((80, 10))
    k = bs.make_kernel_search(bs.PRIOR_BOLTZMANN, kernel_input=kernel_input)
    S = S0.copy(); StS = S.T @ S
    StX = np.zeros((6, 0)) if kernel_input else S.T @ X
    data = X @ X.T if kernel_input else X
    for _ in range(5):
        k(data, S, StS, StX, 80, np.ones(6), 1.0, 1.0, 0.1, 1.0, Jc, hc, np.zeros((80, 6)),
          True, 1.0, 20)
    assert np.array_equal(StS, S.T @ S)
    if not kernel_input:
        assert np.allclose(StX, S.T @ X)

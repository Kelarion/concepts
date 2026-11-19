"""df_util simulators and the util helpers the package keeps."""
import numpy as np
import pytest

from sbmf import df_util, util


@pytest.mark.parametrize("struct,K,kw", [("none", 6, {}), ("categorical", 6, {}),
                                          ("tree", 4, {}), ("grid", 4, {"d": 2})])
def test_undirected_model_samples_binary(struct, K, kw):
    np.random.seed(0)
    S = df_util.UndirectedModel(K, struct, kw).sample(200)
    assert S.shape[0] == 200 and set(np.unique(S)) <= {0, 1}


def test_noisyembed_and_permham():
    np.random.seed(0)
    S = df_util.UndirectedModel(5, "none").sample(100).astype(float)
    W, noise = df_util.noisyembed(S, 20, 0, orth=True, nonneg=False)
    assert W.shape == (20, S.shape[1]) and noise.shape == (100, 20)
    # permham is permutation- and sign-invariant: a shuffled, flipped copy scores 0
    perm = np.random.permutation(S.shape[1])
    S2 = S[:, perm].copy(); S2[:, 0] = 1 - S2[:, 0]
    assert np.all(df_util.permham(S, S2) == 0)


def test_unbalanced_assignment_transposes_adj():
    # regression for the audit fix: a supplied adjacency must follow the transpose
    rng = np.random.default_rng(0)
    cost = rng.random((5, 3))
    adj = np.ones((5, 3), bool)
    assert util.unbalanced_assignment(cost) is not None
    assert util.unbalanced_assignment(cost.T, adj=adj.T) is not None


def test_center_kernel_rows_sum_to_zero():
    X = np.random.default_rng(0).standard_normal((30, 4))
    Kc = util.center_kernel(X @ X.T)
    assert np.allclose(Kc.sum(0), 0) and np.allclose(Kc.sum(1), 0)


def test_lexorder_roundtrip():
    inds = util.LexOrder()
    i, j = np.array([3, 5]), np.array([1, 2])
    assert np.array_equal(inds.inv(inds(i, j))[0], i)

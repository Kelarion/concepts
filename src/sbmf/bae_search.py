"""
bae_search.py  --  the compiled E-step
======================================

The compositional search: one
shared coordinate-descent *scaffold* + two injected, inlined "link" plugins,
so that the spike-only kernel (sbmf) and the spike-and-slab kernel (snmf) are
two instances of the SAME kernel rather than two hand-written copies.

The point being tested: inlining is preserved by specializing the kernel when
the configuration is known (a closure factory closing over the link), instead
of passing the link as a runtime argument (which would be an indirect,
un-inlinable call in the n*m*m hot loop).

A "link" is two element-wise pieces, both on scalars:
    score(E, wjj, tau, sigma2)  -> (logodds, mu, nu)   [no RNG]
    aux_value(s, mu, nu)        -> z                   [RNG for the slab]
`score` runs before the Bernoulli flip; `aux_value` after it (the only place the
slab draws its continuous magnitude), and the scaffold stores z in Z, so the
RNG-draw order matches the originals exactly.
Two latent arrays, not three: `S` is the binary spike (drives the tree/StS
combinatorics) and `Z` is the *effective continuous latent* read by the field
accumulation.  The slab keeps the invariant Z==0 wherever the spike is off, so Z
already equals S*magnitude.  A no-slab model just initializes Z = S and the
binary link keeps Z == S.
Plugins take and return scalars only: a call that passes arrays costs reference-
count updates every time, which in the hot loop is several times the work of the
arithmetic.  For the same reason the sparse bookkeeping below is written out in
the sweeps rather than called.
The scaffold owns everything shared: the permuted i,j loop, the neighbour-field
accumulation from the gram (off Z), the tree/sparsity A/B/C/D regularizer, the
StS bookkeeping (off S), and the sigmoid flip.

One factory builds the dense E-step kernel for any number of chains:
`make_dense_search(*link, prior, diag_gram=, debug=, parallel=)`.  Both the serial
(2-D state, plain @njit) and the chain-batched (leading chain axis + `prange`)
builds run the SAME per-chain sweep body (`_make_dense_sweep`); `parallel` just
picks whether it runs directly or inside `prange(C)`.  The operator
(bae_weights) passes parallel=(n_chains > 1), so a single chain keeps the exact
serial RNG stream (bit-for-bit vs old_bae_search.sbmf / snmf).

Run `python bae_search.py` (with the real src on PYTHONPATH) to compare the
serial scaffold against old_bae_search.sbmf / old_bae_search.snmf and the parallel one
against C serial calls, and to time both.
"""

import math
from functools import lru_cache

import numpy as np
from numba import njit, prange

from . import numba_util


# ---------------------------------------------------------------------------
#  Link plugins  (marked inline='always' so LLVM folds them into the loop)
# ---------------------------------------------------------------------------
#
# Binary spike (recovers old_bae_search.sbmf): the field is linear, magnitude is
# fixed at 1, nothing extra is sampled.

@njit(inline='always')
def score_binary(E, wjj, tau, sigma2):
    # logodds at the Gaussian precision 1/sigma2 (sbmf's XW/sigma2, WtW/sigma2);
    # tau unused.  sigma2 == 1 -> exactly E - 0.5*wjj (a/1.0 is exact).
    return (E - 0.5 * wjj) / sigma2, 0.0, 0.0


@njit(inline='always')
def aux_binary(s, mu, nu):
    return s                                # effective latent == the spike


# Spike-and-slab (recovers old_bae_search.snmf): the field is the partially
# collapsed marginal over the continuous magnitude, and when the spike turns on
# we draw that magnitude from a truncated normal.

@njit(inline='always')
def score_slab(E, wjj, tau, sigma2):
    # wjj is the TRUE gram diagonal ||W[:,j]||^2 (the prior on S is now a separate
    # additive term, no longer folded into WtW), so the only degeneracy left is a
    # genuinely dead weight column (wjj -> 0) or an overflowing W (wjj/E/sigma2 ->
    # inf/NaN).  Either makes nu = sqrt(sigma2/wjj) and mu = .../wjj non-finite,
    # which (a) divides by zero in this kernel and (b) feeds NaN/inf magnitudes to
    # the truncated-normal sampler.  In that case the column carries no usable
    # signal, so report the spike as strongly OFF with a bounded, well-defined
    # (mu, nu): the flip turns off, no magnitude is drawn, nothing blows up.  A
    # no-op when wjj is a healthy positive precision, so well-behaved fits are
    # unchanged.
    if (not (math.isfinite(wjj) and wjj > 1e-12)
            or not (math.isfinite(sigma2) and sigma2 > 0.0)
            or not math.isfinite(E)):
        return -1e18, 0.0, 1.0
    mu = (E - tau * sigma2) / wjj
    nu = math.sqrt(sigma2 / wjj)
    logodds = (math.log(tau * nu * math.sqrt(2 * math.pi))
               + (mu ** 2) / (2 * (sigma2 / wjj))
               + numba_util.log_ndtr(mu / nu))
    return logodds, mu, nu


@njit(inline='always')
def aux_slab(s, mu, nu):
    if s > 0.5:
        return numba_util.sample_trunc_norm(mu, nu)
    return 0.0                              # maintains Z==0 off-spike


# ---------------------------------------------------------------------------
#  Prior terms  (the latent prior's additive contribution to the log-odds)
# ---------------------------------------------------------------------------
#
# The conditional log-odds for the spike S_ij factors as
#     score(E, wjj)                                  # LIKELIHOOD, integrates the slab Z
#   + [ -alpha - beta*inhib(S, StS) + coupling ]     # PRIOR on S
# The bracket reads the binary spikes S_ik (never the magnitude Z_ik) and never
# touches wjj.  The scaffold owns the two terms every prior shares -- sparsity
# (-alpha) and the tree/StS penalty (-beta*inhib) -- and the prior PLUGIN supplies
# only its coupling, as a pair of scalar functions (field, pair): the scaffold
# computes  (field(hc_j) + sum_{k active, k != j} pair(Jc_jk)) / prior_temp,
# summing over the row's ACTIVE latents (the Ising conditional is linear in S_ik,
# so the inactive ones contribute nothing).  A prior with no coupling is None,
# and the scaffold compiles the coupling out entirely.
#
# Tree penalty, two ways (chosen at run time by `tree_batch`):
#   tree_batch == 0  exact: `_tree_inhib`, a dense O(m) pass over the live StS.
#   tree_batch  > 0  batched: StS is read only at the start of each batch of
#                    `tree_batch` rows (it is still updated live).  With StS fixed,
#                    each k's contribution depends on the row only through
#                    (S_ij, S_ik), so for s = S_ij
#                        inhib = base[s, j] + sum_{k active} D[s, j, k]
#                    with base[s, j] = sum_k f(j, k | s, 0) and
#                    D[s, j, k] = f(j, k | s, 1) - f(j, k | s, 0) (zero for k == j).
#                    O(m^2) per batch, then O(nnz) per site.  (Evaluating D per
#                    active pair instead, as the 0.0.14 Cython sampler did, was no
#                    faster: the baseline is most of each rebuild.)  Exact for the frozen
#                    StS; batches much larger than ~50 rows can destabilise the
#                    sampler (stale counts switch many latents at once).

@njit(inline='always')
def _tree_rule(Sjk, Sjj, Skk, N, Sij, Sik):
    # the A/B/C/D StS combinatorics for one (j, k), verbatim from old_bae_search.snmf,
    # on scalars (StS[j,k], StS[j,j], StS[k,k]) so calls in hot loops pass no arrays
    A = Sjk - Sij * Sik
    B = Sjj - A - Sij
    C = Skk - A
    D = N - A - B - C
    out = 0.0
    if A < min(B, C - 1, D):
        out += Sik
    if B < min(A, C, D - 1):
        out += 1 - Sik
    if C <= min(A, B, D):
        out -= Sik
    if D <= min(A, B, C):
        out -= 1 - Sik
    return out


@njit(inline='always')
def _tree_contrib(StS, N, j, k, Sij, Sik):
    return _tree_rule(StS[j, k], StS[j, j], StS[k, k], N, Sij, Sik)


@njit(inline='always')
def _tree_inhib(S, i, j, StS, N, beta):
    """Exact tree penalty: sum over ALL k (incl. k == j) against the live StS.
    Gated on beta so a no-tree model pays nothing."""
    if not (beta > 1e-6):
        return 0.0
    Sij = S[i, j]
    Sjj = StS[j, j]
    inhib = 0.0
    for k in range(S.shape[1]):
        inhib += _tree_rule(StS[j, k], Sjj, StS[k, k], N, Sij, S[i, k])
    return inhib


@njit
def _tree_tables(StS, N, base, D):
    """Fill base (2, m) and D (2, m, m) from the current StS (see above)."""
    m = StS.shape[0]
    for j in range(m):
        Sjj = StS[j, j]
        for s in range(2):
            b = 0.0
            for k in range(m):
                Sjk, Skk = StS[j, k], StS[k, k]
                if k == j:                       # S_ik is S_ij itself
                    b += _tree_rule(Sjk, Sjj, Skk, N, s, s)
                    D[s, j, k] = 0.0
                else:
                    f0 = _tree_rule(Sjk, Sjj, Skk, N, s, 0)
                    b += f0
                    D[s, j, k] = _tree_rule(Sjk, Sjj, Skk, N, s, 1) - f0
            base[s, j] = b


@njit(inline='always')
def _ising_field(h):
    return h


@njit(inline='always')
def _ising_pair(J):
    return J


@njit(inline='always')
def _zero(x):
    return 0.0


# LatentPrior: no coupling (sparsity and tree live in the scaffold).
prior_unstructured = None
# BoltzmannPrior: the Ising conditional.  Jc is the symmetric {0,1}-coupling (zero
# diagonal) and hc the {0,1}-field, i.e. (2J, 2h) for a prior
# log p(S) ~ S'JS + 2h'S, so the scaffold accumulates its conditional log-odds
# 2*(J S + h)_j over the active latents.  sigma2-independent: J is learned from the
# spike statistics alone.  `prior_temp` divides the coupling only (high prior_temp
# = weak structure); the sparsity/tree terms are not scaled.
prior_boltzmann = (_ising_field, _ising_pair)


# ---------------------------------------------------------------------------
#  Active sets  (the sparse row representation the sweeps share)
# ---------------------------------------------------------------------------
#
# Row i's on-latents are act[i, :size[i]], unordered; pos[i, j] is j's slot in
# act[i] or -1.  Add and remove are O(1) (remove swaps the last entry into the
# hole), so every neighbour sum -- likelihood field, coupling, batched tree term,
# StS update -- costs O(nnz) instead of O(m).  Rebuilt from S at each call.

@njit
def _active_sets(S):
    n, m = S.shape
    act = np.zeros((n, m), np.int64)
    pos = -np.ones((n, m), np.int64)
    size = np.zeros(n, np.int64)
    for i in range(n):
        for j in range(m):
            if S[i, j] > 0.5:
                act[i, size[i]] = j
                pos[i, j] = size[i]
                size[i] += 1
    return act, pos, size


# Add and remove are written out inside the sweeps rather than called as helpers:
# a call that takes arrays (inline='always' or not) costs reference-count updates
# on every call, which made the sweep ~2.3x slower at 64 latents.


# ---------------------------------------------------------------------------
#  The scaffold factory
# ---------------------------------------------------------------------------
#
# Closes over (score, aux_value, prior) so they are compile-time freevar
# constants -> inlinable.  Memoized so each distinct (link, prior) compiles the
# kernel exactly once.  The scaffold owns what is shared: the permuted i, j loop,
# the active sets, the LIKELIHOOD field (off Z, true gram), sparsity and tree
# penalties, the Bernoulli flip, and the StS bookkeeping.

@lru_cache(maxsize=None)
def _make_dense_sweep(score, aux_value, prior, diag_gram=False, debug=False,
                      batched_tree=False):
    """ONE chain's sweep over 2-D state (S/Z (n,m), WtW/StS (m,m), XW (n,m)), scalar
    sigma2 and prior_temp; the serial search calls it directly and the parallel one
    per chain inside `prange(C)`.
    * `diag_gram` -- the gram is diagonal (orthonormal W, e.g. BiPCA), so the
      neighbour sum vanishes and is skipped.
    * `debug` -- samples exactly as normal but records each element's pre-temp
      log-odds (likelihood + prior) into `out`.
    * `batched_tree` -- compile-time choice of the tree penalty: the exact dense
      pass, or the batched tables refreshed every `tree_batch` rows (see "Prior
      terms").  Compiled separately because inlining the unused exact pass into the
      batched loop slowed it ~1.7x.  The search wrapper picks one per call.
    Visits rows and latents in random order and draws one uniform per site (plus
    the slab magnitude when on), the same RNG stream as before the sparse rewrite;
    the neighbour sums now run over active latents only, so results can differ from
    the dense kernel in the last floating-point bit."""
    coupled = prior is not None
    field, pair = prior if coupled else (_zero, _zero)

    @njit
    def sweep(XW, S, Z, WtW, StS, N, temp, alpha, beta, tau, sigma2,
              Jc, hc, inplace, out, prior_temp, tree_batch):
        n, m = S.shape
        regularize = beta > 1e-6
        act, pos, size = _active_sets(S)
        base = np.zeros((2, m))
        D = np.zeros((2, m, m)) if batched_tree else np.zeros((2, 1, 1))
        rows = np.random.permutation(np.arange(n))
        for r in range(n):
            i = rows[r]
            if batched_tree and regularize and r % tree_batch == 0:
                _tree_tables(StS, N, base, D)
            for j in np.random.permutation(np.arange(m)):
                Sij = S[i, j]
                s = 1 if Sij > 0.5 else 0
                p = pos[i, j]                    # remove j from the active set
                if p >= 0:
                    last = act[i, size[i] - 1]
                    act[i, p] = last
                    pos[i, last] = p
                    pos[i, j] = -1
                    size[i] -= 1
                # ONE pass over the other active latents gathers every neighbour sum:
                # the LIKELIHOOD field (off Z, true gram; Z == 0 wherever the spike is
                # off, so this is exact for the slab), the prior coupling, and the
                # batched tree correction (D[s, j, j] == 0, so excluding j is exact).
                E = XW[i, j]
                coup = field(hc[j]) if coupled else 0.0
                inhib = base[s, j] if (batched_tree and regularize) else 0.0
                for l in range(size[i]):
                    k = act[i, l]
                    if not diag_gram:
                        E -= WtW[j, k] * Z[i, k]
                    if coupled:
                        coup += pair(Jc[j, k])
                    if batched_tree and regularize:
                        inhib += D[s, j, k]
                if regularize and not batched_tree:
                    inhib = _tree_inhib(S, i, j, StS, N, beta)   # exact: dense, live StS
                # PRIOR on S: sparsity, coupling, tree
                lp = -alpha - beta * inhib if regularize else -alpha
                if coupled:
                    lp += coup / prior_temp
                logodds, mu, nu = score(E, WtW[j, j], tau, sigma2)
                if debug:
                    out[i, j] = logodds + lp     # record, then sample as usual
                curr = (logodds + lp) / temp
                if curr < -100:
                    prob = 0.0
                elif curr > 100:
                    prob = 1.0
                else:
                    prob = 1.0 / (1.0 + math.exp(-curr))
                new_Sij = 1.0 * (np.random.rand() < prob)
                if new_Sij > 0.5:                # add j to the active set
                    act[i, size[i]] = j
                    pos[i, j] = size[i]
                    size[i] += 1
                ds = new_Sij - Sij
                # StS bookkeeping (the prior's state; maintained by the scaffold)
                if regularize and inplace and ds != 0.0:
                    StS[j, j] += ds
                    for l in range(size[i]):
                        k = act[i, l]
                        if k != j:
                            StS[j, k] += ds
                            StS[k, j] += ds
                S[i, j] = new_Sij
                Z[i, j] = aux_value(new_Sij, mu, nu)
        return S, Z
    return sweep


def _build_dense_search(score, aux_value, prior, diag_gram=False, debug=False,
                        parallel=False):
    """The dense E-step kernel, in one factory.  `parallel` (compile time) selects
    between two thin wrappers around the shared `_make_dense_sweep` body:
      parallel=False -- a plain @njit over 2-D state; the single-chain path, with the
                        serial RNG stream.
      parallel=True  -- an @njit(parallel=True) running the SAME sweep per chain in
                        `prange(C)` over 3-D state (leading chain axis; sigma2 and
                        prior_temp (C,)).  Chains own all their state, so this is
                        race-free.
    Two wrappers because numba types a function on its argument shapes (a 2-D and a
    3-D `S` cannot share one compiled body).  Memoized via make_dense_search on
    (link, prior, diag_gram, debug, parallel).  The operator (bae_weights) passes
    parallel=(n_chains > 1)."""
    sweep_exact = _make_dense_sweep(score, aux_value, prior, diag_gram, debug, False)
    sweep_batch = _make_dense_sweep(score, aux_value, prior, diag_gram, debug, True)

    @njit
    def sweep(XW, S, Z, WtW, StS, N, temp, alpha, beta, tau, sigma2,
              Jc, hc, inplace, out, prior_temp, tree_batch):
        if tree_batch > 0 and beta > 1e-6:
            return sweep_batch(XW, S, Z, WtW, StS, N, temp, alpha, beta, tau, sigma2,
                               Jc, hc, inplace, out, prior_temp, tree_batch)
        return sweep_exact(XW, S, Z, WtW, StS, N, temp, alpha, beta, tau, sigma2,
                           Jc, hc, inplace, out, prior_temp, tree_batch)

    if parallel:
        @njit(parallel=True)
        def search(XW, S, Z, WtW, StS, N, temp, alpha, beta, tau, sigma2,
                   Jc, hc, inplace=True, out=None, prior_temp=None, tree_batch=0):
            C = S.shape[0]
            for c in prange(C):
                # `out` is (C,n,m) under debug (else None); the branch is pruned
                # since `debug` is a compile-time constant.
                if debug:
                    sweep(XW[c], S[c], Z[c], WtW[c], StS[c], N, temp, alpha, beta,
                          tau, sigma2[c], Jc[c], hc[c], inplace, out[c], prior_temp[c],
                          tree_batch)
                else:
                    sweep(XW[c], S[c], Z[c], WtW[c], StS[c], N, temp, alpha, beta,
                          tau, sigma2[c], Jc[c], hc[c], inplace, out, prior_temp[c],
                          tree_batch)
            return S, Z
    else:
        @njit
        def search(XW, S, Z, WtW, StS, N, temp, alpha, beta, tau, sigma2,
                   Jc, hc, inplace=True, out=None, prior_temp=1.0, tree_batch=0):
            sweep(XW, S, Z, WtW, StS, N, temp, alpha, beta, tau, sigma2,
                  Jc, hc, inplace, out, prior_temp, tree_batch)
            return S, Z
    search._dense_sweep = sweep    # exposed for the inlining self-test
    return search


make_dense_search = lru_cache(maxsize=None)(_build_dense_search)


# ---------------------------------------------------------------------------
#  The kernel scaffold factory  (Gram factorization -- NOT a linear operator)
# ---------------------------------------------------------------------------
#
# KernelBMF factorizes a Gram/kernel matrix: K ~ center(S diag(scl) S^T).  The
# reconstruction is QUADRATIC in the binary latents S, so there is no linear
# operator (no forward/drive/gram) and it cannot ride make_dense_search.  But the
# conditional log-odds still factors the same way,
#     curr = ( LIKELIHOOD field  +  additive S-prior ) / temp ,
# and the additive S-prior is *exactly* the same plugin the dense models use
# (sparsity + tree combinatorics + any Boltzmann coupling).  So this scaffold
# reuses `prior` verbatim -- a plain LatentPrior reproduces old_bae_search.kerbmf3
# bit-for-bit, and a BoltzmannPrior gives KernelBMF a structured Ising prior for
# free.  Only the LIKELIHOOD field is kernel-specific and lives here.
#
# The field is the response of <data, S diag(scl) S^T> to flipping S_ij, weighted
# by the per-feature scale scl[j] (self) and scl[k] (neighbours), verbatim from
# old_bae_search.kerbmf3 (feature input) / kerbmf2 (kernel input).  `kernel_input`
# (compile-time) selects the input: a feature matrix X (n, d), for which the
# running StX = S^T X is maintained live and the field costs O(n d); or a
# precomputed kernel K (n, n), which needs no StX and costs O(n^2).  The two forms
# compute the identical field when K = X X^T.

def _build_kernel_search(prior, kernel_input=False, currents=False, debug=False):
    coupled = prior is not None
    field, pair = prior if coupled else (_zero, _zero)

    # `currents` (compile-time) turns the kernel into a diagnostic: instead of
    # sampling, it writes the raw per-element log-odds (like + prior, pre-temp)
    # into `out` and does NOT flip -- so every (i,j) is scored against the SAME
    # fixed S.  Used by test_kernelbmf2_port.py to check the field against the
    # exact change in log-likelihood.
    #
    # `debug` (compile-time) is the live-fit counterpart: it samples EXACTLY as the
    # normal path (flips, updates StS) but ALSO records each element's log-odds into
    # `out` as it sweeps -- so you can watch the current magnitudes evolve over a
    # real fit (out[i,j] is the log-odds at the moment (i,j) was resampled).  Both
    # off (the normal sampling path) never touches `out`, so callers pass a dummy.

    @njit
    def search(data, S, StS, StX, N, scl, sigma2, temp, alpha, beta, Jc, hc, out,
               inplace=True, prior_temp=1.0, tree_batch=0):
        n, m = S.shape
        t = (N - 1) / N
        regularize = beta > 1e-6
        batched = regularize and tree_batch > 0
        act, pos, size = _active_sets(S)
        base = np.zeros((2, m))
        D = np.zeros((2, m, m)) if batched else np.zeros((2, 1, 1))
        rows = np.random.permutation(np.arange(n))
        for r in range(n):
            i = rows[r]
            if batched and r % tree_batch == 0:
                _tree_tables(StS, N, base, D)
            for j in np.random.permutation(np.arange(m)):
                Sij = S[i, j]
                s = 1 if Sij > 0.5 else 0
                S_j = (StS[j, j] - Sij) / (N - 1)
                pij = scl[j]
                # LIKELIHOOD (kernel-specific and dense: every latent enters through
                # its centred value S_ik - S_k, and the data term is O(n) or O(d))
                if kernel_input:
                    inp = (1 - 2 * Sij) * data[i, i] / t
                    for k in range(n):
                        inp += 2 * data[i, k] * S[k, j] / t
                else:
                    inp = 0.0
                    d = StX.shape[1]
                    for k in range(d):
                        inp += (2 * StX[j, k] * data[i, k]
                                + (1 - 2 * Sij) * data[i, k] ** 2) / t
                dot = pij * (t * (2 * (N - 2) * S_j * (1 - S_j) + 1) * (0.5 - Sij))
                for k in range(m):
                    Sik = S[i, k]
                    S_k = (StS[k, k] - Sik) / (N - 1)
                    term = 2 * (StS[j, k] - Sij * Sik
                                + t * (0.5 - S_j - S_k - (N - 2) * S_j * S_k)) * (Sik - S_k)
                    term -= t * (1 - S_k) * S_k * (2 * S_j - 1)
                    dot += scl[k] * term
                like = pij * (inp - dot) / sigma2
                # PRIOR: the same terms and plugins as the dense scaffold
                lp = -alpha
                if coupled:
                    coup = field(hc[j])
                    for l in range(size[i]):
                        k = act[i, l]
                        if k != j:
                            coup += pair(Jc[j, k])
                    lp += coup / prior_temp
                if regularize:
                    if batched:
                        inhib = base[s, j]
                        for l in range(size[i]):
                            inhib += D[s, j, act[i, l]]      # D[s, j, j] == 0
                    else:
                        inhib = _tree_inhib(S, i, j, StS, N, beta)
                    lp -= beta * inhib
                if currents:
                    out[i, j] = like + lp        # diagnostic: score, do not flip
                    continue
                if debug:
                    out[i, j] = like + lp        # record, then sample as usual
                curr = (like + lp) / temp
                if curr < -100:
                    prob = 0.0
                elif curr > 100:
                    prob = 1.0
                else:
                    prob = 1.0 / (1.0 + math.exp(-curr))
                ds = (1.0 * (np.random.rand() < prob)) - Sij
                S[i, j] += ds
                if ds > 0:                       # add j to the active set
                    act[i, size[i]] = j
                    pos[i, j] = size[i]
                    size[i] += 1
                elif ds < 0:                     # remove j
                    p = pos[i, j]
                    last = act[i, size[i] - 1]
                    act[i, p] = last
                    pos[i, last] = p
                    pos[i, j] = -1
                    size[i] -= 1
                if inplace and ds != 0.0:
                    StS[j, j] += ds
                    for l in range(size[i]):
                        k = act[i, l]
                        if k != j:
                            StS[j, k] += ds
                            StS[k, j] += ds
                    if not kernel_input:
                        d = StX.shape[1]
                        for k in range(d):
                            StX[j, k] += data[i, k] * ds

        return S

    return search


make_kernel_search = lru_cache(maxsize=None)(_build_kernel_search)

# A model composes a "link" -- the (score, aux_value) pair from the slab side --
# with a "prior" -- the additive S-prior from the LatentPrior.  no-slab models use
# BINARY_LINK, slab models SLAB_LINK; unstructured priors use PRIOR_PLAIN,
# Boltzmann-structured ones PRIOR_BOLTZMANN.
BINARY_LINK = (score_binary, aux_binary)
SLAB_LINK = (score_slab, aux_slab)
PRIOR_PLAIN = prior_unstructured
PRIOR_BOLTZMANN = prior_boltzmann

# prebuilt kernels (compiled lazily on first call); unstructured prior here, the
# structured ones are composed by the model from its prior's plugin.  The `par_*`
# variants are the chain-batched (prange) builds the operators use when n_chains>1.
sbmf_search = make_dense_search(*BINARY_LINK, PRIOR_PLAIN)
snmf_search = make_dense_search(*SLAB_LINK, PRIOR_PLAIN)
par_sbmf_search = make_dense_search(*BINARY_LINK, PRIOR_PLAIN, parallel=True)
par_snmf_search = make_dense_search(*SLAB_LINK, PRIOR_PLAIN, parallel=True)


# ===========================================================================
#  self-test  (vs old_bae_search.sbmf / old_bae_search.snmf)
# ===========================================================================

@njit
def _seed(s):
    np.random.seed(s)


def _selftest():
    import time
    from . import old_bae_search     # legacy kernels; not shipped by default

    rng = np.random.RandomState(0)
    n, d, m = 200, 50, 8
    W = rng.randn(d, m)
    b = rng.randn(d)
    X = rng.randn(n, d)
    XW = (X - b) @ W
    WtW = W.T @ W
    S0 = 1.0 * (rng.rand(n, m) > 0.5)
    temp, alpha, beta = 0.7, 0.0, 1e-2
    SEED = 12345

    def StS_of(S):
        return S.T @ S

    # ---- BINARY: unified vs sbmf -----------------------------------------
    S_a, StS_a = S0.copy(), StS_of(S0)
    _seed(SEED)
    old_bae_search.sbmf(XW.copy(), S_a, WtW.copy(), temp=temp,
                    StS=StS_a, N=n, beta=beta, alpha=alpha, inplace=True)

    S_b, Z_b, StS_b = S0.copy(), S0.copy(), StS_of(S0)   # no-slab: Z = S
    zeroJ, zeroh = np.zeros((m, m)), np.zeros(m)
    _seed(SEED)
    sbmf_search(XW.copy(), S_b, Z_b, WtW.copy(), StS_b, n,
                temp, alpha, beta, 1.0, 1.0, zeroJ, zeroh, True)

    bin_equal = np.array_equal(S_a, S_b)
    bin_frac = float(np.mean(S_a == S_b))
    print(f"[binary]  S array_equal vs sbmf : {bin_equal}   "
          f"(agreement {bin_frac:.5f})")

    # ---- SLAB: unified vs snmf -------------------------------------------
    Z0 = 1.0 * S0 * np.abs(rng.randn(n, m))
    tau, sigma2 = 1.0, 1.0

    S_c, Z_c, StS_c = S0.copy(), Z0.copy(), StS_of(S0)
    _seed(SEED)
    old_bae_search.snmf(XW.copy(), S_c, Z_c, WtW.copy(), temp=temp,
                    sigma2=sigma2, tau=tau, StS=StS_c, N=n,
                    beta=beta, alpha=alpha, inplace=True)

    S_d = S0.copy()
    Z_d = Z0.copy()                          # Z0 already zeroed off-spike
    StS_d = StS_of(S0)
    _seed(SEED)
    snmf_search(XW.copy(), S_d, Z_d, WtW.copy(), StS_d, n,
                temp, alpha, beta, tau, sigma2, zeroJ, zeroh, True)

    slab_S_equal = np.array_equal(S_c, S_d)
    slab_Z_equal = np.allclose(Z_c, Z_d)
    slab_Z_exact = np.array_equal(Z_c, Z_d)
    print(f"[slab]    S array_equal vs snmf : {slab_S_equal}")
    print(f"[slab]    Z array_equal vs snmf : {slab_Z_exact}   "
          f"(allclose {slab_Z_equal})")

    # ---- inlining evidence: scan the compiled LLVM IR --------------------
    def link_calls(kernel):
        llvm = kernel.inspect_llvm()
        txt = "\n".join(llvm.values())
        # any 'call' line that references the link helpers by name
        hits = [ln.strip() for ln in txt.splitlines()
                if " call " in ln and ("score_" in ln or "aux_" in ln
                                       or "log_ndtr" in ln
                                       or "sample_trunc_norm" in ln)]
        return hits

    print(f"[inolg]   binary kernel: residual link calls = "
          f"{len(link_calls(sbmf_search._dense_sweep))}")
    print(f"[inolg]   slab   kernel: residual link calls = "
          f"{len(link_calls(snmf_search._dense_sweep))}")

    # ---- timing -----------------------------------------------------------
    def timeit(fn, reps=50):
        fn()                                   # warm
        t0 = time.perf_counter()
        for _ in range(reps):
            fn()
        return (time.perf_counter() - t0) / reps * 1e3   # ms/sweep

    t_sbmf = timeit(lambda: old_bae_search.sbmf(
        XW.copy(), S0.copy(), WtW.copy(), temp=temp,
        StS=StS_of(S0), N=n, beta=beta, alpha=alpha, inplace=True))
    t_bin = timeit(lambda: sbmf_search(
        XW.copy(), S0.copy(), S0.copy(), WtW.copy(),
        StS_of(S0), n, temp, alpha, beta, 1.0, 1.0, zeroJ, zeroh, True))
    print(f"[time]    sbmf {t_sbmf:.3f} ms  | unified-binary {t_bin:.3f} ms  "
          f"({t_bin / t_sbmf:.2f}x)")

    t_snmf = timeit(lambda: old_bae_search.snmf(
        XW.copy(), S0.copy(), Z0.copy(), WtW.copy(), temp=temp,
        sigma2=sigma2, tau=tau, StS=StS_of(S0), N=n,
        beta=beta, alpha=alpha, inplace=True))
    t_slab = timeit(lambda: snmf_search(
        XW.copy(), S0.copy(), Z0.copy(), WtW.copy(),
        StS_of(S0), n, temp, alpha, beta, tau, sigma2, zeroJ, zeroh, True))
    print(f"[time]    snmf {t_snmf:.3f} ms  | unified-slab   {t_slab:.3f} ms  "
          f"({t_slab / t_snmf:.2f}x)")


def _selftest_parallel():
    """C parallel chains vs C serial calls.  Per-thread RNG in a prange means we
    can't match a specific serial call bit-for-bit, so we check that (a)
    shapes/dtypes line up, (b) every spike is a valid {0,1} draw, and (c) the
    parallel sweep reduces the per-chain energy like the serial one."""
    import time

    rng = np.random.RandomState(0)
    C, n, d, m = 6, 200, 50, 8
    temp, alpha, beta, tau, sig = 0.7, 0.0, 1e-2, 1.0, 1.0

    W = rng.randn(C, d, m)                 # distinct W per chain -> chains differ
    X = rng.randn(n, d)
    XW = np.einsum('nd,cdm->cnm', X, W)
    WtW = np.einsum('cdm,cdk->cmk', W, W)
    S0 = 1.0 * (rng.rand(C, n, m) > 0.5)
    StS0 = np.einsum('cnm,cnk->cmk', S0, S0)
    Jc = np.zeros((C, m, m))
    hc = np.zeros((C, m))
    sigma2 = np.full(C, sig)
    prior_temp = np.ones(C)          # per-chain prior temperature ((C,), like sigma2)

    Sp, Zp, StSp = S0.copy(), S0.copy(), StS0.copy()
    par_sbmf_search(XW.copy(), Sp, Zp, WtW.copy(), StSp, n,
                    temp, alpha, beta, tau, sigma2, Jc, hc, True, None, prior_temp)

    Ss = S0.copy()
    for c in range(C):
        Sc, Zc, StSc = S0[c].copy(), S0[c].copy(), StS0[c].copy()
        sbmf_search(XW[c].copy(), Sc, Zc, WtW[c].copy(), StSc, n,
                    temp, alpha, beta, tau, sig,
                    np.zeros((m, m)), np.zeros(m), True)
        Ss[c] = Sc

    print(f"[par shape]   parallel S {Sp.shape}  serial-stack S {Ss.shape}")
    print(f"[par valid]   all spikes in {{0,1}}: {np.all((Sp == 0) | (Sp == 1))}")

    def energy(S):
        recon = np.einsum('cnm,cdm->cnd', S, W)
        return ((X[None] - recon) ** 2).mean((1, 2))

    e0, ep, es = energy(S0), energy(Sp), energy(Ss)
    print(f"[par energy]  init          : {np.round(e0, 3)}")
    print(f"[par energy]  parallel sweep: {np.round(ep, 3)}  "
          f"(all decreased: {np.all(ep < e0)})")
    print(f"[par energy]  serial  sweep : {np.round(es, 3)}  "
          f"(all decreased: {np.all(es < e0)})")

    def timeit(fn, reps=30):
        fn()
        t0 = time.perf_counter()
        for _ in range(reps):
            fn()
        return (time.perf_counter() - t0) / reps * 1e3

    t_par = timeit(lambda: par_sbmf_search(
        XW.copy(), S0.copy(), S0.copy(), WtW.copy(), StS0.copy(), n,
        temp, alpha, beta, tau, sigma2, Jc, hc, True, None, prior_temp))

    def serial_all():
        for c in range(C):
            sbmf_search(XW[c].copy(), S0[c].copy(), S0[c].copy(),
                        WtW[c].copy(), StS0[c].copy(), n,
                        temp, alpha, beta, tau, sig,
                        np.zeros((m, m)), np.zeros(m), True)
    t_ser = timeit(serial_all)
    print(f"[par time]    {C} chains: parallel {t_par:.3f} ms | serial {t_ser:.3f} "
          f"ms ({t_ser / t_par:.2f}x)")


if __name__ == "__main__":
    _selftest()
    print()
    _selftest_parallel()
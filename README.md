# sbmf

Sparse and semi-binary matrix factorization: binary autoencoder / binary matrix
factorization models, their numba-compiled Gibbs E-steps and latent priors, and
the simulation and cross-validation utilities.

| Module | Contents |
|---|---|
| `bae_models` | `SemiBMF`, `JBMF`, `BiPCA`, `SCPD`, `RRBMF`, `KernelBMF` |
| `bae_weights` | decoders: affine, Procrustes, CP, reduced-rank, torch |
| `bae_priors` | `LatentPrior`, `BoltzmannPrior`, `MRFPrior`, temperature schedules |
| `bae_search` | numba E-step kernels |
| `bae_util` | cross-validation (`impcv`, `bicv`, `gabriel_bicv`, ...), annealing |
| `df_util` | latent-structure simulators (`UndirectedModel`, `noisyembed`), cut decompositions, `permham` |
| `util`, `numba_util` | internal helpers |

## Use it from another project

```bash
uv add --editable ../concepts                 # local clone; edits show up immediately
uv add git+https://github.com/Kelarion/concepts --branch sbmf-0.1   # or straight from GitHub
pip install -e ../concepts                    # without uv
```

```python
from sbmf.bae_models import BiPCA, SemiBMF
from sbmf.bae_util import impcv
from sbmf import df_util
```

Optional extras: `sbmf[plot]` (matplotlib, for `df_util.plotgraph` / `plotwalk`),
`sbmf[graph]` (pydot, for graphviz layouts), `sbmf[polytope]` (pypoman, for
`df_util.enumerate_cuts`), or `sbmf[all]`.

## E-step sampler

Every dense model's E-step is a Gibbs sweep that keeps each row's active latents
in a list, so neighbour sums (likelihood field, Boltzmann coupling, tree penalty)
cost O(active) rather than O(latents) per site. The tree penalty (`tree_reg`)
reads the co-occurrence counts `StS` once per batch of `tree_batch` rows
(default 20) instead of after every flip; `tree_batch=0` gives the exact sampler,
which draws the same samples as 0.1.0 for a given seed.

```python
model = SemiBMF(50, tree_reg=1.0)                 # batched tree penalty
model = SemiBMF(50, tree_reg=1.0, tree_batch=0)   # exact
```

Batches of 10-50 rows gave the same recovery, held-out error and stability as
the exact sampler in simulations and on real data. Much larger batches can make
the sampler oscillate (stale counts switch many latents on at once), so keep it
well below the number of rows. Table rebuilds cost O(latents^2) each, so with
many rows and latents a larger batch is faster.

## Develop

```bash
uv sync
uv run pytest
uv run ruff check src tests
uv build
```

## Changes from 0.0.14

0.0.14 shipped only the Cython extension `sbmf.sparse` (`BiMat`,
`GaussianSampler`, `KernelSampler`). 0.1.0 replaces it with the pure-Python +
numba package above, so no compiler is needed; the Cython code is in the git
history. `util.py` keeps only the 20 helpers the package uses (of 131).

Not included: `mrf_samplers.py` and `old_bae_search.py`. Without them
`MRFPrior` (`J_prior='mrf'`) and `ConvOperator` raise an error naming the missing
file; copy the file into `src/sbmf/` (changing `import util` to
`from . import util`) to enable them.

## Known issues (not fixed, they change results)

- `bae_util.impcv`, `fpcv` and `multifit` do not reset the model between folds or
  chains, so later folds start from weights trained on held-out entries. `bicv`
  and `loocv` do reset (`model.initialized = False`).
- Multi-chain fits are not reproducible from a seed: inside `prange` each thread
  has its own numba RNG, and seeding reaches only the calling thread.
- `Procrustes.init_params` hot-starts from the uncentered data.
- `numba_util.log_ndtr` has a 0.037 jump at `x = -5`.
- Unfinished drafts: `bae_util.kercv` (raises `NotImplementedError`),
  `df_util.reassign` with `pi` given, `ConvOperator` / `ConvBMF`.

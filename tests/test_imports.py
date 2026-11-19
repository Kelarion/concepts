"""Every module imports cleanly, and the light ones stay light."""
import importlib
import subprocess
import sys

import pytest

MODULES = [
    "util", "df_util", "numba_util", "bae_util", "bae_search",
    "bae_priors", "bae_weights", "bae_models",
]


@pytest.mark.parametrize("name", MODULES)
def test_module_imports(name):
    importlib.import_module(f"sbmf.{name}")


def _loaded_after(stmt):
    code = f"import sys; {stmt}; print(' '.join(m for m in ('torch','cvxpy','numba','matplotlib') if m in sys.modules))"
    return subprocess.run([sys.executable, "-c", code], capture_output=True,
                          text=True, check=True).stdout.split()


def test_top_level_import_is_lazy():
    assert _loaded_after("import sbmf") == []


def test_util_needs_no_heavy_deps():
    assert _loaded_after("from sbmf import util") == []


def test_search_kernels_do_not_pull_torch():
    # bae_search used to import bae_util -> torch, torchvision, cvxpy, ...
    loaded = _loaded_after("from sbmf import bae_search, bae_priors")
    assert "torch" not in loaded and "cvxpy" not in loaded


def test_df_util_does_not_import_matplotlib():
    assert "matplotlib" not in _loaded_after("from sbmf import df_util")

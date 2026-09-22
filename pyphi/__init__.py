"""
pyphi: A Python package for advanced latent variable data analysis: PCA/PLS/JYPLS/JRPLS/TPLS/PLS-CCA (OPLS)
"""
__version__ = "6.0.7"
import importlib

from pyphi import calc

__all__ = ["calc", "batch", "plots"]

_LAZY = {"batch", "plots"}

def __getattr__(name):
    if name in _LAZY:
        return importlib.import_module(f"pyphi.{name}")
    raise AttributeError(f"module 'pyphi' has no attribute {name!r}")

def __dir__():
    return sorted(set(globals()) | _LAZY)

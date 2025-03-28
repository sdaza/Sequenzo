"""
@Author  : Yuqi Liang 梁彧祺
@File    : __init__.py
@Time    : 11/02/2025 16:42
@Desc    : Sequenzo Package Initialization
"""
# sequenzo/__init__.py (Top-level)

import importlib.util
import platform
import sys
from pathlib import Path

__version__ = "0.1.3"

# Construct the expected path to the extension module.
_lib_path = Path(__file__).parent / "sequenzo" / ("_rust_fast_cluster" + {
    "Windows": ".pyd",
    "Darwin": ".so",  # Might need .cpython-39-darwin.so, see note below
    "Linux": ".so"
}[platform.system()])

from sequenzo import datasets, visualization, clustering, dissimilarity_measures, define_sequence_data, big_data


def __getattr__(name):
    try:
        if name == "datasets":
            from sequenzo import datasets
            return datasets
        elif name == "visualization":
            from sequenzo import visualization
            return visualization
        elif name == "clustering":
            from sequenzo import clustering
            return clustering
        elif name == "dissimilarity_measures":
            from sequenzo import dissimilarity_measures
            return dissimilarity_measures
        elif name == "SequenceData":
            from sequenzo.define_sequence_data import SequenceData
            return SequenceData
        elif name == "big_data":
            from sequenzo.big_data import clara
            return clara
        elif name == "_rust_fast_cluster" and _extension_loaded:
            return sys.modules["sequenzo._rust_fast_cluster"]

        # Provide a helpful error message.
        if name == "clustering" and not _extension_loaded:
            raise ImportError(
                "The Rust extension failed to load, so fast clustering is unavailable. "
                "The 'clustering' module is not available without the extension. "
                "Please ensure you have built the package correctly using `maturin develop` or `python -m build`."
            )
        elif not _extension_loaded:
            raise ImportError(
                f"The Rust extension failed to load. Please ensure you have "
                "built the package correctly using `maturin develop` or `python -m build`."
            )

    except ImportError as e:
        raise AttributeError(f"Could not import {name}: {e}")

    raise AttributeError(f"module 'sequenzo' has no attribute '{name}'")


SequenceData = sequenzo.define_sequence_data.SequenceData


# But *don't* directly import clustering here.  It's handled by __getattr__.
# These are the public APIs of the package, but use __getattr__ for lazy imports
__all__ = [
    'datasets',
    'visualization',
    'clustering',  # Included, but it is lazy-loaded via __getattr__
    'dissimilarity_measures',
    'SequenceData',
    'big_data',
]

# NOTE:  You *could* choose to make _rust_fast_cluster directly accessible
# via __all__ as well, *if* you want users to be able to directly interact
# with the Rust functions.  If you do, add "_rust_fast_cluster" to __all__.
# If you *don't* add it, it will still be accessible via
# `sequenzo._rust_fast_cluster`, but it won't be considered part of the
# "public" API.  This is a design choice.
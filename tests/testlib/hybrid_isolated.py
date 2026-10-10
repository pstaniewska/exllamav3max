"""
Load the hybrid MoE host-side modules (model/moe_expert_cache.py, moe_hybrid_ref.py,
moe_cpu_host.py, moe_numa_hosts.py, util/numa_*) without the CUDA extension and without running
the exllamav3 package __init__ (which builds or loads the extension). The package directories are
mapped under a private name with empty package modules, and `exllamav3_ext` is a stub exposing
only what these modules call at import time or on CPU paths. Used by the nogpu hybrid MoE tests;
on a GPU host the same tests run unchanged.
"""

from __future__ import annotations
import importlib
import os
import sys
import types

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "exllamav3")
PKG = "_exl3_hybrid_isolated"


class _ExtStub(types.ModuleType):
    """exllamav3_ext stand-in: CPU-side queries answer like a host without pinning, anything
    else raises (a test reaching a CUDA call is a test bug)"""

    def exl3_moe_cpu_core_order(self):
        return [], 0

    def exl3_moe_cpu_set_memops(self, enabled):
        pass

    def exl3_moe_cpu_swizzle_group(self, K):
        return 0

    def exl3_moe_cpu_planar_layout(self, K):
        return 0

    def __getattr__(self, name):
        raise AttributeError(f"exllamav3_ext stub: {name} needs the CUDA extension")


def load():
    """Returns a namespace of the isolated modules (cached per process)"""
    if PKG in sys.modules and hasattr(sys.modules[PKG], "_loaded"):
        return sys.modules[PKG]._loaded
    root = types.ModuleType(PKG)
    root.__path__ = [ROOT]
    sys.modules[PKG] = root
    for sub in ("model", "util", "modules"):
        m = types.ModuleType(f"{PKG}.{sub}")
        m.__path__ = [os.path.join(ROOT, sub)]
        sys.modules[f"{PKG}.{sub}"] = m
    ext_pkg = types.ModuleType(f"{PKG}.ext")
    ext_pkg.exllamav3_ext = _ExtStub("exllamav3_ext")
    sys.modules[f"{PKG}.ext"] = ext_pkg
    ns = types.SimpleNamespace(
        ref = importlib.import_module(f"{PKG}.model.moe_hybrid_ref"),
        cache = importlib.import_module(f"{PKG}.model.moe_expert_cache"),
        topo = importlib.import_module(f"{PKG}.util.numa_topology"),
        numa_mem = importlib.import_module(f"{PKG}.util.numa_mem"),
        host = importlib.import_module(f"{PKG}.model.moe_cpu_host"),
        numa_hosts = importlib.import_module(f"{PKG}.model.moe_numa_hosts"),
        dispatch = importlib.import_module(f"{PKG}.model.moe_hybrid_dispatch"),
        ext = ext_pkg.exllamav3_ext,
    )
    root._loaded = ns
    return ns

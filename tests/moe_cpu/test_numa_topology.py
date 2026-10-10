"""
NUMA topology discovery (exllamav3/util/numa_topology.py) and NUMA memory placement
(exllamav3/util/numa_mem.py) for the CPU MoE expert offload. Topology tests run against synthetic
sysfs/procfs/cgroup trees; the syscall tests only check that the wrappers behave (return an errno,
never raise) on whatever this machine is. No GPU, model or compiled extension needed: both modules are
loaded by file path so importing them does not run exllamav3/__init__.
"""

import ctypes
import importlib.util
import mmap
import os
import platform
import sys

import pytest

pytestmark = pytest.mark.nogpu

_UTIL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "exllamav3", "util")


def _load(name: str):
    mod_name = f"_exl3_test_{name}"
    if mod_name in sys.modules:
        return sys.modules[mod_name]
    spec = importlib.util.spec_from_file_location(mod_name, os.path.join(_UTIL, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod  # dataclasses resolve string annotations through sys.modules
    spec.loader.exec_module(mod)
    return mod


nt = _load("numa_topology")
nm = _load("numa_mem")


# --------------------------------------------------------------------------------------------------
# Synthetic trees

def _write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok = True)
    with open(path, "w") as f:
        f.write(text)


def _meminfo(node, total_kb, free_kb, inactive_file_kb):
    return (
        f"Node {node} MemTotal:       {total_kb} kB\n"
        f"Node {node} MemFree:        {free_kb} kB\n"
        f"Node {node} MemUsed:        {total_kb - free_kb} kB\n"
        f"Node {node} Inactive(file): {inactive_file_kb} kB\n"
        f"Node {node} HugePages_Total:     0\n"
    )


def make_two_socket(root):
    """
    Two sockets, 4 cores each, SMT2, Linux enumeration order: primaries 0-7 (0-3 socket 0, 4-7 socket 1),
    siblings 8-15 (cpu N+8 is cpu N's sibling). node0 = 0-3,8-11, node1 = 4-7,12-15
    """
    sysfs = os.path.join(root, "sys")
    for cpu in range(16):
        prim = cpu % 8
        pkg = prim // 4
        t = os.path.join(sysfs, "devices", "system", "cpu", f"cpu{cpu}", "topology")
        _write(os.path.join(t, "core_id"), f"{prim % 4}\n")
        _write(os.path.join(t, "physical_package_id"), f"{pkg}\n")
        _write(os.path.join(t, "thread_siblings_list"), f"{prim},{prim + 8}\n")
    nd = os.path.join(sysfs, "devices", "system", "node")
    _write(os.path.join(nd, "node0", "cpulist"), "0-3,8-11\n")
    _write(os.path.join(nd, "node1", "cpulist"), "4-7,12-15\n")
    _write(os.path.join(nd, "node0", "distance"), "10 21\n")
    _write(os.path.join(nd, "node1", "distance"), "21 10\n")
    _write(os.path.join(nd, "node0", "meminfo"), _meminfo(0, 65536, 1024, 512))
    _write(os.path.join(nd, "node1", "meminfo"), _meminfo(1, 65536, 32768, 0))
    _write(os.path.join(nd, "online"), "0-1\n")
    cg = os.path.join(root, "cgroup")
    os.makedirs(cg, exist_ok = True)
    return sysfs, cg


# --------------------------------------------------------------------------------------------------
# cpulist

@pytest.mark.parametrize("text, cpus", [
    ("", []),
    ("\n", []),
    ("0", [0]),
    ("0-3,8,10-11\n", [0, 1, 2, 3, 8, 10, 11]),
    (" 4-5 , 1 ,1\n", [1, 4, 5]),
    ("0-63,128-191", list(range(64)) + list(range(128, 192))),
])
def test_cpulist_roundtrip(text, cpus):
    assert nt.parse_cpulist(text) == cpus
    assert nt.parse_cpulist(nt.format_cpulist(cpus)) == cpus


def test_format_cpulist_compact():
    assert nt.format_cpulist([11, 10, 8, 3, 2, 1, 0]) == "0-3,8,10-11"
    assert nt.format_cpulist([]) == ""


@pytest.mark.parametrize("bad", ["a", "3-1", "1-", "-1"])
def test_parse_cpulist_rejects_garbage(bad):
    with pytest.raises(ValueError):
        nt.parse_cpulist(bad)


# --------------------------------------------------------------------------------------------------
# Discovery

def test_two_socket_smt(tmp_path):
    sysfs, cg = make_two_socket(str(tmp_path))
    topo = nt.discover(sysfs_root = sysfs, cgroup_root = cg, allowed_cpus = list(range(16)))
    assert topo.source == "sysfs"
    assert topo.is_numa
    assert [n.id for n in topo.cpu_nodes()] == [0, 1]
    n0, n1 = topo.node(0), topo.node(1)
    assert n0.cpus == [0, 1, 2, 3, 8, 9, 10, 11]
    assert n0.cores == [[0, 8], [1, 9], [2, 10], [3, 11]]
    assert n0.physical_cpus == [0, 1, 2, 3]
    assert n0.smt_siblings == [8, 9, 10, 11]
    assert n1.physical_cpus == [4, 5, 6, 7]
    assert n1.smt_siblings == [12, 13, 14, 15]
    assert n0.distances == [10, 21] and n1.distances == [21, 10]
    assert n0.mem_total == 65536 * 1024
    assert n0.mem_free == (1024 + 512) * 1024
    assert n1.mem_free == 32768 * 1024
    assert topo.node_of_cpu(9) == 0 and topo.node_of_cpu(12) == 1 and topo.node_of_cpu(99) is None
    assert topo.node(7) is None
    assert topo.cgroup_mem_limit is None
    d = topo.describe()
    assert "node 0: cpus 0-3,8-11" in d and "node 1: cpus 4-7,12-15" in d and "4 cores" in d


def test_memory_only_node_and_cgroup_limit(tmp_path):
    sysfs, cg = make_two_socket(str(tmp_path))
    nd = os.path.join(sysfs, "devices", "system", "node", "node2")
    _write(os.path.join(nd, "cpulist"), "\n")
    _write(os.path.join(nd, "meminfo"), _meminfo(2, 1024, 1024, 0))
    _write(os.path.join(cg, "memory.max"), "8589934592\n")
    topo = nt.discover(sysfs_root = sysfs, cgroup_root = cg, allowed_cpus = list(range(16)))
    assert [n.id for n in topo.nodes] == [0, 1, 2]
    assert topo.node(2).cpus == [] and topo.node(2).cores == []
    assert [n.id for n in topo.cpu_nodes()] == [0, 1]
    assert topo.cgroup_mem_limit == 8 << 30


def test_cgroup_unlimited(tmp_path):
    sysfs, cg = make_two_socket(str(tmp_path))
    _write(os.path.join(cg, "memory.max"), "max\n")
    topo = nt.discover(sysfs_root = sysfs, cgroup_root = cg, allowed_cpus = list(range(16)))
    assert topo.cgroup_mem_limit is None


def test_restricted_cpuset(tmp_path):
    # A cpuset holding node 0's first two cores (both LPs) and one LP of node 1
    sysfs, cg = make_two_socket(str(tmp_path))
    topo = nt.discover(sysfs_root = sysfs, cgroup_root = cg, allowed_cpus = [0, 1, 8, 9, 5])
    n0, n1 = topo.node(0), topo.node(1)
    assert n0.cpus == [0, 1, 8, 9]
    assert n0.cores == [[0, 8], [1, 9]]
    assert n1.cpus == [5] and n1.cores == [[5]] and n1.smt_siblings == []
    assert topo.allowed_cpus == [0, 1, 5, 8, 9]
    # Only siblings allowed on node 1: the sibling is that core's first allowed LP
    topo = nt.discover(sysfs_root = sysfs, cgroup_root = cg, allowed_cpus = [0, 12, 13])
    assert topo.node(1).physical_cpus == [12, 13]
    # Nothing allowed on node 1: not a CPU node, so not NUMA from this process's point of view
    topo = nt.discover(sysfs_root = sysfs, cgroup_root = cg, allowed_cpus = [0, 1, 2])
    assert [n.id for n in topo.cpu_nodes()] == [0] and not topo.is_numa


def test_cgroup_cpuset_effective_default(tmp_path, monkeypatch):
    sysfs, cg = make_two_socket(str(tmp_path))
    _write(os.path.join(cg, "cpuset.cpus.effective"), "0-1,8\n")
    monkeypatch.setattr(nt.os, "sched_getaffinity", lambda pid: set(range(16)), raising = False)
    monkeypatch.setattr(nt, "_own_cgroup_paths", lambda controller: ["/"])
    topo = nt.discover(sysfs_root = sysfs, cgroup_root = cg)
    assert topo.allowed_cpus == [0, 1, 8]
    assert topo.node(0).cores == [[0, 8], [1]]


def test_topology_fallback_to_siblings_and_singletons(tmp_path):
    sysfs, cg = make_two_socket(str(tmp_path))
    for cpu in (0, 8):
        os.remove(os.path.join(sysfs, "devices", "system", "cpu", f"cpu{cpu}", "topology", "core_id"))
    for f in ("core_id", "physical_package_id", "thread_siblings_list"):
        for cpu in (1, 9):
            os.remove(os.path.join(sysfs, "devices", "system", "cpu", f"cpu{cpu}", "topology", f))
    topo = nt.discover(sysfs_root = sysfs, cgroup_root = cg, allowed_cpus = list(range(16)))
    assert topo.node(0).cores == [[0, 8], [1], [2, 10], [3, 11], [9]]


def test_cpu_map_override(tmp_path):
    sysfs, cg = make_two_socket(str(tmp_path))
    topo = nt.discover(sysfs_root = sysfs, cgroup_root = cg, allowed_cpus = list(range(16)),
                       cpu_map = "0:0-7; 3:8-15")
    assert topo.source == "override"
    assert [n.id for n in topo.nodes] == [0, 1, 3]
    assert topo.node(0).cpus == list(range(8))
    assert topo.node(0).cores == [[c] for c in range(8)]
    assert topo.node(1).cpus == []
    assert topo.node(0).mem_total == 65536 * 1024
    assert topo.node(3).cpus == list(range(8, 16))
    assert topo.node(3).mem_total is None and topo.node(3).mem_free is None and topo.node(3).distances is None
    assert [n.id for n in topo.cpu_nodes()] == [0, 3]


@pytest.mark.parametrize("cpu_map, match", [
    ("0:0-3;1:4-20", "20"),      # outside allowed
    ("0:0-5;1:4-7", "CPU 4"),    # overlap
    ("0:0-3;0:4-7", "more than once"),
    ("0-3", "node:cpulist"),
    ("x:0-3", "malformed"),
])
def test_cpu_map_invalid(tmp_path, cpu_map, match):
    sysfs, cg = make_two_socket(str(tmp_path))
    with pytest.raises(ValueError, match = match):
        nt.discover(sysfs_root = sysfs, cgroup_root = cg, allowed_cpus = list(range(16)), cpu_map = cpu_map)


def test_fallback_without_node_dir(tmp_path):
    sysfs, cg = make_two_socket(str(tmp_path))
    import shutil
    shutil.rmtree(os.path.join(sysfs, "devices", "system", "node"))
    topo = nt.discover(sysfs_root = sysfs, cgroup_root = cg, allowed_cpus = [0, 1, 8, 9])
    assert topo.source == "fallback"
    assert len(topo.nodes) == 1 and topo.nodes[0].id == 0
    assert topo.nodes[0].cpus == [0, 1, 8, 9] and topo.nodes[0].cores == [[0, 8], [1, 9]]
    assert topo.nodes[0].mem_total is None and topo.nodes[0].mem_free is None
    assert not topo.is_numa


def test_empty_sysfs_never_raises(tmp_path):
    topo = nt.discover(sysfs_root = str(tmp_path / "nosys"), cgroup_root = str(tmp_path / "nocg"),
                       allowed_cpus = [0, 1])
    assert topo.source == "fallback" and topo.nodes[0].cores == [[0], [1]]
    assert topo.cgroup_mem_limit is None


def test_real_machine_discovery():
    topo = nt.discover()
    assert topo.allowed_cpus and topo.cpu_nodes()
    assert set(c for n in topo.cpu_nodes() for c in n.cpus) <= set(topo.allowed_cpus)
    assert topo.describe()
    assert nt.get_topology() is nt.get_topology()


def test_numa_debug(monkeypatch):
    monkeypatch.setenv("EXL3_NUMA_DEBUG", "0")
    assert not nt.numa_debug()
    monkeypatch.setenv("EXL3_NUMA_DEBUG", "1")
    assert nt.numa_debug()
    monkeypatch.delenv("EXL3_NUMA_DEBUG")
    assert not nt.numa_debug()


# --------------------------------------------------------------------------------------------------
# GPU locality

BUS = "0000:41:00.0"


def _pci(sysfs, bus, node):
    _write(os.path.join(sysfs, "bus", "pci", "devices", bus, "numa_node"), f"{node}\n")


def _gpu_node(tmp_path, env = None, bus_id = BUS):
    return nt.gpu_numa_node(0, sysfs_root = str(tmp_path / "sys"), proc_root = str(tmp_path / "proc"),
                            env = env or {}, bus_id = bus_id)


def test_gpu_node_env_override(tmp_path):
    _pci(str(tmp_path / "sys"), BUS, 0)
    assert _gpu_node(tmp_path, env = {"EXL3_NUMA_GPU_NODE": "0:1,1:0"}) == 1
    # Override for another device only: falls through to sysfs
    assert _gpu_node(tmp_path, env = {"EXL3_NUMA_GPU_NODE": "1:1"}) == 0
    with pytest.raises(ValueError):
        _gpu_node(tmp_path, env = {"EXL3_NUMA_GPU_NODE": "zero"})


def test_gpu_node_sysfs(tmp_path):
    _pci(str(tmp_path / "sys"), BUS, 1)
    assert _gpu_node(tmp_path) == 1
    # nvidia-smi style 8-digit domain, uppercase
    assert _gpu_node(tmp_path, bus_id = "00000000:41:00.0".upper()) == 1


def test_gpu_node_sysfs_minus_one(tmp_path):
    _pci(str(tmp_path / "sys"), BUS, -1)
    assert _gpu_node(tmp_path) is None


def test_gpu_node_missing(tmp_path):
    assert _gpu_node(tmp_path) is None
    # No bus id at all and no proc entries
    assert _gpu_node(tmp_path, bus_id = "") is None


def _proc_gpu(proc, bus):
    _write(os.path.join(proc, "driver", "nvidia", "gpus", bus, "information"),
           f"Model: \t\t NVIDIA Test GPU\nIRQ:   \t\t 100\nBus Location: \t {bus}\nDevice Minor: \t 0\n")


def test_gpu_node_proc_single_gpu(tmp_path):
    _pci(str(tmp_path / "sys"), BUS, 1)
    _proc_gpu(str(tmp_path / "proc"), BUS)
    assert _gpu_node(tmp_path, bus_id = "") == 1


def test_gpu_node_proc_two_gpus(tmp_path):
    _pci(str(tmp_path / "sys"), BUS, 1)
    _pci(str(tmp_path / "sys"), "0000:c1:00.0", 0)
    _proc_gpu(str(tmp_path / "proc"), BUS)
    _proc_gpu(str(tmp_path / "proc"), "0000:c1:00.0")
    assert _gpu_node(tmp_path, bus_id = "") is None


def test_gpu_pci_bus_id_never_raises():
    r = nt.gpu_pci_bus_id(0)
    assert r is None or isinstance(r, str)


# --------------------------------------------------------------------------------------------------
# numa_mem

def _linux_supported():
    return sys.platform.startswith("linux") and platform.machine().lower() in ("x86_64", "amd64", "aarch64", "arm64")


def test_available_matches_platform():
    assert nm.available() == _linux_supported()


def test_policy_scope_none_is_noop():
    before = nm._get_mempolicy()
    with nm.policy_scope(None):
        assert nm._get_mempolicy() == before


def test_policy_scope_restores_on_exception():
    if not nm.available() or not os.path.isdir("/sys/devices/system/node/node0"):
        pytest.skip("NUMA syscalls not available here")
    before = nm._get_mempolicy()
    with pytest.raises(KeyError):
        with nm.policy_scope(0):
            raise KeyError("x")
    assert nm._get_mempolicy() == before


def test_mbind_anonymous_mapping():
    if not _linux_supported():
        assert nm.mbind(0, 4096, [0]) != 0
        assert nm.pages_per_node(0, 4096) is None
        return
    if not os.path.isdir("/sys/devices/system/node/node0"):
        pytest.skip("no NUMA node 0 on this machine")
    n = 16 * mmap.PAGESIZE
    buf = mmap.mmap(-1, n)
    c = ctypes.c_char.from_buffer(buf)
    try:
        addr = ctypes.addressof(c)
        err = nm.mbind(addr, n, [0])
        assert isinstance(err, int)
        # 0, or what a sandbox may answer (seccomp / gVisor / no NUMA support compiled in)
        assert err in (0, nm.errno.EPERM, nm.errno.ENOSYS, nm.errno.EINVAL)
        # Unaligned start and length round to whole pages internally
        assert isinstance(nm.mbind(addr + 100, 5000, 0), int)
        buf[:] = b"\x01" * n  # fault the pages in
        r = nm.pages_per_node(addr, n, sample = 8)
        assert r is None or (isinstance(r, dict) and sum(r.values()) == 8)
        if err == 0 and r is not None:
            assert set(r) <= {0, -1}
        assert isinstance(nm.bind_range_or_warn(addr, n, 0, "test arena"), bool)
    finally:
        del c
        buf.close()


class _FakeTopo:
    def __init__(self, free):
        self.free = free

    def node_free_bytes(self, node):
        return self.free


def test_check_node_memory(monkeypatch):
    monkeypatch.delenv("EXL3_HOST_MEM_RESERVE_MB", raising = False)
    with pytest.raises(RuntimeError, match = "NUMA node 1") as e:
        nm.check_node_memory(1 << 30, 1, "expert arena", topo = _FakeTopo(64 << 20))
    msg = str(e.value)
    assert "expert arena" in msg and "1024 MiB" in msg and "64 MiB" in msg
    assert "EXL3_NUMA_MEMORY_POLICY" in msg and "EXL3_HOST_MEM_RESERVE_MB=0" in msg
    monkeypatch.setenv("EXL3_HOST_MEM_RESERVE_MB", "0")
    nm.check_node_memory(1 << 30, 1, "expert arena", topo = _FakeTopo(64 << 20))
    nm.check_node_memory(1 << 30, None, "expert arena")


def test_check_node_memory_unknown_node_runs_global_check(monkeypatch):
    # Node free unknown: only the system-wide check runs, which a small request passes
    monkeypatch.delenv("EXL3_HOST_MEM_RESERVE_MB", raising = False)
    nm.check_node_memory(4096, 5, "small buffer", topo = _FakeTopo(None))
    nm.check_node_memory(4096, None, "small buffer")


def test_node_free_bytes_from_topology(tmp_path):
    sysfs, cg = make_two_socket(str(tmp_path))
    topo = nt.discover(sysfs_root = sysfs, cgroup_root = cg, allowed_cpus = list(range(16)))
    assert nm.node_free_bytes(1, topo) == 32768 * 1024
    # Re-read, not the discovery snapshot
    _write(os.path.join(sysfs, "devices", "system", "node", "node1", "meminfo"), _meminfo(1, 65536, 100, 0))
    assert nm.node_free_bytes(1, topo) == 100 * 1024
    assert nm.node_free_bytes(9, topo) is None

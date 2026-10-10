"""
NUMA topology discovery for the CPU MoE expert offload.

On a multi-socket host the offload runs one worker process per NUMA node, pinned to that node's CPUs
with its weight arena bound to that node's memory, so every expert matmul streams weights from local
DRAM. That needs, per node: the CPUs this process may actually use (affinity and cgroup cpuset, not
just what the node owns), how they group into physical cores (workers take one LP per core first),
how much memory the node has free, and which node the GPU hangs off (its staging buffers should be
local to the PCIe root complex).

Everything here is read from sysfs/procfs with plain file reads, so it imports without torch and
works in containers. A file that cannot be read means "unknown" (None / empty), never an error and
never a guess: in particular a GPU whose NUMA node cannot be determined reports None, not node 0,
because binding staging memory to the wrong socket is worse than not binding it at all.

Environment
    EXL3_NUMA_CPU_MAP     override node -> CPU assignment, "node:cpulist;node:cpulist", e.g.
                          "0:0-63,128-191;1:64-127,192-255" (nodes separated by ';', CPUs in the
                          usual Linux cpulist syntax). Every CPU must be allowed for this process and
                          nodes must be disjoint
    EXL3_NUMA_GPU_NODE    override GPU -> node, "dev:node,dev:node", e.g. "0:1"
    EXL3_NUMA_DEBUG       non-empty and not "0": callers print topology/placement details
"""
from __future__ import annotations
import os
import re
from dataclasses import dataclass


# --------------------------------------------------------------------------------------------------
# cpulist syntax

def parse_cpulist(s: str) -> list[int]:
    """
    Parse a Linux cpulist ("0-3,8,10-11") into a sorted list of unique CPU ids. Empty or whitespace
    input is the empty list (a memory-only node's cpulist file holds just a newline). Raises
    ValueError on malformed input, since the same parser validates user-supplied overrides.
    """
    out = set()
    for part in s.replace("\n", ",").split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            a, b = int(a.strip()), int(b.strip())
            if a < 0 or b < a:
                raise ValueError(f"Invalid CPU range '{part}' in cpulist '{s.strip()}'")
            out.update(range(a, b + 1))
        else:
            c = int(part)
            if c < 0:
                raise ValueError(f"Invalid CPU '{part}' in cpulist '{s.strip()}'")
            out.add(c)
    return sorted(out)


def format_cpulist(cpus) -> str:
    """Inverse of parse_cpulist: compact ranges, e.g. [0, 1, 2, 3, 8, 10, 11] -> "0-3,8,10-11" """
    cpus = sorted(set(int(c) for c in cpus))
    parts = []
    i = 0
    while i < len(cpus):
        j = i
        while j + 1 < len(cpus) and cpus[j + 1] == cpus[j] + 1:
            j += 1
        parts.append(str(cpus[i]) if i == j else f"{cpus[i]}-{cpus[j]}")
        i = j + 1
    return ",".join(parts)


# --------------------------------------------------------------------------------------------------
# File helpers: every read degrades to None

def _read(path: str) -> str | None:
    try:
        with open(path) as f:
            return f.read()
    except (OSError, UnicodeDecodeError):
        return None


def _read_int(path: str) -> int | None:
    s = _read(path)
    if s is None:
        return None
    try:
        return int(s.strip())
    except ValueError:
        return None


def _read_cpulist(path: str) -> list[int] | None:
    s = _read(path)
    if s is None:
        return None
    try:
        return parse_cpulist(s)
    except ValueError:
        return None


def _read_node_meminfo(sysfs_root: str, node_id: int) -> tuple[int | None, int | None]:
    """
    (MemTotal, MemFree + Inactive(file)) in bytes from node{N}/meminfo, None where unreadable. Lines
    look like "Node 0 MemTotal:       65536 kB". Free memory counts inactive page cache because the
    kernel reclaims it before refusing a node-bound allocation; MemFree alone reads near zero on any
    host that has been reading model files
    """
    s = _read(os.path.join(sysfs_root, "devices", "system", "node", f"node{node_id}", "meminfo"))
    if s is None:
        return None, None
    fields = {}
    for line in s.splitlines():
        parts = line.split()
        if len(parts) < 4 or parts[0] != "Node" or not parts[2].endswith(":"):
            continue
        try:
            v = int(parts[3])
        except ValueError:
            continue
        if len(parts) > 4 and parts[4].lower() == "kb":
            v *= 1024
        fields[parts[2][:-1]] = v
    total = fields.get("MemTotal")
    free = fields.get("MemFree")
    if free is not None:
        free += fields.get("Inactive(file)", 0)
    return total, free


# --------------------------------------------------------------------------------------------------
# Topology model

@dataclass
class NumaNode:
    id: int
    cpus: list[int]                 # allowed logical CPUs of this node, sorted
    cores: list[list[int]]          # allowed sibling LPs per physical core, cores ordered by first LP
    mem_total: int | None           # bytes, None if unreadable
    mem_free: int | None            # bytes, MemFree + Inactive(file), None if unreadable
    distances: list[int] | None     # node*/distance row

    @property
    def physical_cpus(self) -> list[int]:
        """One LP per physical core (the first allowed sibling)"""
        return [c[0] for c in self.cores]

    @property
    def smt_siblings(self) -> list[int]:
        """Every allowed LP that is not its core's first"""
        return sorted(lp for c in self.cores for lp in c[1:])


@dataclass
class NumaTopology:
    nodes: list[NumaNode]           # every sysfs node, memory-only nodes included (cpus == [])
    allowed_cpus: list[int]
    cgroup_mem_limit: int | None    # bytes, None = unlimited / unknown
    source: str                     # "sysfs", "override" or "fallback"

    def cpu_nodes(self) -> list[NumaNode]:
        """Nodes this process can run on, ascending id"""
        return sorted((n for n in self.nodes if n.cpus), key = lambda n: n.id)

    def node(self, node_id) -> NumaNode | None:
        for n in self.nodes:
            if n.id == node_id:
                return n
        return None

    def node_of_cpu(self, cpu) -> int | None:
        for n in self.nodes:
            if cpu in n.cpus:
                return n.id
        return None

    @property
    def is_numa(self) -> bool:
        return len(self.cpu_nodes()) > 1

    def node_free_bytes(self, node_id: int) -> int | None:
        """
        Free memory on a node (MemFree + Inactive(file)) re-read from sysfs now, since the snapshot taken
        at discovery goes stale as soon as weights start loading. Falls back to the snapshot when the
        node has no readable meminfo
        """
        _, free = _read_node_meminfo(getattr(self, "_sysfs_root", "/sys"), node_id)
        if free is None:
            n = self.node(node_id)
            free = n.mem_free if n is not None else None
        return free

    def describe(self) -> str:
        def gib(b):
            return "?" if b is None else f"{b / 1024 ** 3:.1f} GiB"
        cn = self.cpu_nodes()
        lim = "none" if self.cgroup_mem_limit is None else gib(self.cgroup_mem_limit)
        lines = [
            f"NUMA topology ({self.source}): {len(self.nodes)} node(s), {len(cn)} with allowed CPUs, "
            f"allowed CPUs {format_cpulist(self.allowed_cpus) or '-'}, cgroup memory limit {lim}"
        ]
        for n in sorted(self.nodes, key = lambda n: n.id):
            lines.append(
                f"  node {n.id}: cpus {format_cpulist(n.cpus) or '-'} "
                f"({len(n.cores)} cores, {len(n.smt_siblings)} SMT siblings), "
                f"mem {gib(n.mem_free)} free / {gib(n.mem_total)} total, "
                f"distances {' '.join(str(d) for d in n.distances) if n.distances else '?'}"
            )
        return "\n".join(lines)


# --------------------------------------------------------------------------------------------------
# cgroup

def _own_cgroup_paths(controller: str | None) -> list[str]:
    """
    This process's cgroup path for a v1 controller (controller name) or the v2 hierarchy (None), from
    /proc/self/cgroup, as the list [own path, parent, ..., "/"] so a limit set on an ancestor is still
    found. Just ["/"] when unreadable (also what a container with a cgroup namespace sees)
    """
    path = "/"
    s = _read("/proc/self/cgroup")
    if s is not None:
        for line in s.splitlines():
            parts = line.split(":", 2)
            if len(parts) != 3:
                continue
            if controller is None and parts[0] == "0" and parts[1] == "":
                path = parts[2] or "/"
                break
            if controller is not None and controller in parts[1].split(","):
                path = parts[2] or "/"
                break
    out = []
    p = "/" + path.strip("/")
    while True:
        out.append(p)
        if p == "/":
            break
        p = os.path.dirname(p)
    return out


def _cgroup_files(cgroup_root: str, sub: str, controller: str | None, name: str) -> list[str]:
    return [os.path.join(cgroup_root, sub, p.lstrip("/"), name) for p in _own_cgroup_paths(controller)]


def _cgroup_cpuset(cgroup_root: str) -> set[int] | None:
    """cgroup v2 cpuset.cpus.effective of the nearest cgroup that has one (unified or hybrid mount)"""
    for sub in ("", "unified"):
        for f in _cgroup_files(cgroup_root, sub, None, "cpuset.cpus.effective"):
            cpus = _read_cpulist(f)
            if cpus:
                return set(cpus)
    return None


def _cgroup_mem_limit(cgroup_root: str) -> int | None:
    """
    Tightest memory limit on this process's cgroup or its ancestors: v2 memory.max ("max" =
    unlimited), else v1 memory.limit_in_bytes (unlimited reads as a huge page-rounded number)
    """
    limits = []
    for sub in ("", "unified"):
        for f in _cgroup_files(cgroup_root, sub, None, "memory.max"):
            s = _read(f)
            if s is not None and s.strip().isdigit():
                limits.append(int(s.strip()))
    if not limits:
        for f in _cgroup_files(cgroup_root, "memory", "memory", "memory.limit_in_bytes"):
            v = _read_int(f)
            if v is not None and 0 < v < (1 << 60):
                limits.append(v)
    return min(limits) if limits else None


def _default_allowed_cpus(cgroup_root: str) -> list[int]:
    try:
        cpus = set(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        cpus = set(range(os.cpu_count() or 1))
    cs = _cgroup_cpuset(cgroup_root)
    if cs and (cpus & cs):
        cpus &= cs
    return sorted(cpus)


# --------------------------------------------------------------------------------------------------
# Discovery

def _core_key(sysfs_root: str, cpu: int):
    """
    Grouping key for the physical core of `cpu`: (package, core_id) when readable, else the first LP of
    thread_siblings_list, else the CPU itself (each LP its own core)
    """
    topo = os.path.join(sysfs_root, "devices", "system", "cpu", f"cpu{cpu}", "topology")
    core = _read_int(os.path.join(topo, "core_id"))
    pkg = _read_int(os.path.join(topo, "physical_package_id"))
    if core is not None and pkg is not None:
        return ("core", pkg, core)
    sib = _read_cpulist(os.path.join(topo, "thread_siblings_list"))
    if sib:
        return ("sib", sib[0])
    return ("cpu", cpu)


def _group_cores(cpus: list[int], keys: dict) -> list[list[int]]:
    groups = {}
    for c in sorted(cpus):
        groups.setdefault(keys[c], []).append(c)
    return sorted(groups.values(), key = lambda g: g[0])


def _parse_cpu_map(cpu_map: str) -> dict[int, list[int]]:
    """'0:0-63,128-191;1:64-127,192-255' -> {0: [...], 1: [...]}; ValueError on bad syntax"""
    out = {}
    for entry in cpu_map.split(";"):
        entry = entry.strip()
        if not entry:
            continue
        if ":" not in entry:
            raise ValueError(f"EXL3_NUMA_CPU_MAP entry '{entry}' is not 'node:cpulist' "
                             f"(expected e.g. '0:0-63,128-191;1:64-127,192-255')")
        nid, cl = entry.split(":", 1)
        try:
            nid = int(nid.strip())
            cpus = parse_cpulist(cl)
        except ValueError as e:
            raise ValueError(f"EXL3_NUMA_CPU_MAP entry '{entry}' is malformed: {e}") from None
        if nid < 0:
            raise ValueError(f"EXL3_NUMA_CPU_MAP entry '{entry}': node id must be >= 0")
        if nid in out:
            raise ValueError(f"EXL3_NUMA_CPU_MAP names node {nid} more than once")
        out[nid] = cpus
    if not out:
        raise ValueError("EXL3_NUMA_CPU_MAP is empty")
    return out


def discover(
    sysfs_root: str = "/sys",
    cgroup_root: str = "/sys/fs/cgroup",
    allowed_cpus: list[int] | None = None,
    cpu_map: str | None = None,
) -> NumaTopology:
    """
    Read the NUMA topology. allowed_cpus defaults to this process's affinity intersected with its cgroup
    v2 effective cpuset. cpu_map ("node:cpulist;node:cpulist") replaces the sysfs node -> CPU assignment
    (CPUs it leaves out belong to no node); it must only name allowed CPUs and its nodes must be
    disjoint, else ValueError. Without any node directory (non-NUMA kernel, some containers) the result
    is one synthetic node 0 holding every allowed CPU, source "fallback".
    """
    if allowed_cpus is None:
        allowed_cpus = _default_allowed_cpus(cgroup_root)
    allowed = sorted(set(int(c) for c in allowed_cpus))
    allowed_set = set(allowed)
    keys = {c: _core_key(sysfs_root, c) for c in allowed}

    node_dir = os.path.join(sysfs_root, "devices", "system", "node")
    sysfs_ids = []
    try:
        for e in os.listdir(node_dir):
            m = re.fullmatch(r"node(\d+)", e)
            if m and os.path.isdir(os.path.join(node_dir, e)):
                sysfs_ids.append(int(m.group(1)))
    except OSError:
        pass
    sysfs_ids.sort()

    def mk_node(nid: int, cpus: list[int], from_sysfs: bool) -> NumaNode:
        cpus = sorted(set(cpus) & allowed_set)
        if from_sysfs:
            total, free = _read_node_meminfo(sysfs_root, nid)
            dist = _read(os.path.join(node_dir, f"node{nid}", "distance"))
            try:
                dist = [int(x) for x in dist.split()] if dist is not None and dist.strip() else None
            except ValueError:
                dist = None
        else:
            total, free, dist = None, None, None
        return NumaNode(id = nid, cpus = cpus, cores = _group_cores(cpus, keys),
                        mem_total = total, mem_free = free, distances = dist)

    if cpu_map is not None and cpu_map.strip():
        mapping = _parse_cpu_map(cpu_map)
        bad = sorted(set(c for cpus in mapping.values() for c in cpus) - allowed_set)
        if bad:
            raise ValueError(f"EXL3_NUMA_CPU_MAP lists CPUs this process may not use: {format_cpulist(bad)} "
                             f"(allowed: {format_cpulist(allowed) or 'none'})")
        seen = {}
        for nid, cpus in sorted(mapping.items()):
            for c in cpus:
                if c in seen:
                    raise ValueError(f"EXL3_NUMA_CPU_MAP assigns CPU {c} to both node {seen[c]} and node {nid}")
                seen[c] = nid
        ids = sorted(set(sysfs_ids) | set(mapping))
        nodes = [mk_node(nid, mapping.get(nid, []), nid in sysfs_ids) for nid in ids]
        source = "override"
    elif sysfs_ids:
        nodes = [
            mk_node(nid, _read_cpulist(os.path.join(node_dir, f"node{nid}", "cpulist")) or [], True)
            for nid in sysfs_ids
        ]
        source = "sysfs"
    else:
        nodes = [mk_node(0, allowed, False)]
        source = "fallback"

    topo = NumaTopology(nodes = nodes, allowed_cpus = allowed,
                        cgroup_mem_limit = _cgroup_mem_limit(cgroup_root), source = source)
    topo._sysfs_root = sysfs_root
    return topo


# --------------------------------------------------------------------------------------------------
# GPU locality

def gpu_pci_bus_id(device_index: int) -> str | None:
    """PCI address "dddd:bb:dd.0" of a CUDA device from torch's device properties, None if unavailable"""
    try:
        import torch
        p = torch.cuda.get_device_properties(device_index)
        dom, bus, dev = p.pci_domain_id, p.pci_bus_id, p.pci_device_id
        return "%04x:%02x:%02x.0" % (int(dom), int(bus), int(dev))
    except Exception:
        return None


def _bus_id_candidates(bus_id: str) -> list[str]:
    """The id as given plus its 4- and 8-hex-digit-domain spellings, lowercase (sysfs uses 4, nvidia-smi 8)"""
    b = bus_id.strip().lower()
    out = [b]
    parts = b.split(":")
    if len(parts) == 2:
        parts = ["0"] + parts
    if len(parts) == 3:
        try:
            dom = int(parts[0], 16)
            rest = ":".join(parts[1:])
            out += ["%04x:%s" % (dom, rest), "%08x:%s" % (dom, rest)]
        except ValueError:
            pass
    return list(dict.fromkeys(out))


def _pci_numa_node(sysfs_root: str, bus_id: str) -> int | None:
    for cand in _bus_id_candidates(bus_id):
        v = _read_int(os.path.join(sysfs_root, "bus", "pci", "devices", cand, "numa_node"))
        if v is not None and v >= 0:
            return v
    return None


def _parse_gpu_node_env(s: str) -> dict[int, int]:
    out = {}
    for entry in s.split(","):
        entry = entry.strip()
        if not entry:
            continue
        try:
            dev, node = entry.split(":")
            out[int(dev)] = int(node)
        except ValueError:
            raise ValueError(f"EXL3_NUMA_GPU_NODE entry '{entry}' is not 'device:node' (e.g. '0:1')") from None
    return out


def gpu_numa_node(
    device_index: int,
    sysfs_root: str = "/sys",
    proc_root: str = "/proc",
    env: dict | None = None,
    bus_id: str | None = None,
) -> int | None:
    """
    NUMA node of a CUDA device, or None when it cannot be determined (never a default of 0). In order:
    EXL3_NUMA_GPU_NODE, the PCI device's sysfs numa_node (via `bus_id` or torch), and, only when the
    NVIDIA driver lists exactly one GPU so there is no ambiguity, that GPU's bus location
    """
    env = os.environ if env is None else env
    ov = env.get("EXL3_NUMA_GPU_NODE", "")
    if ov.strip():
        m = _parse_gpu_node_env(ov)
        if device_index in m:
            return m[device_index]

    if bus_id is None:
        bus_id = gpu_pci_bus_id(device_index)
    if bus_id:
        n = _pci_numa_node(sysfs_root, bus_id)
        if n is not None:
            return n

    gdir = os.path.join(proc_root, "driver", "nvidia", "gpus")
    try:
        gpus = [e for e in os.listdir(gdir) if os.path.isdir(os.path.join(gdir, e))]
    except OSError:
        gpus = []
    if len(gpus) == 1:
        loc = None
        info = _read(os.path.join(gdir, gpus[0], "information"))
        if info is not None:
            for line in info.splitlines():
                if line.strip().startswith("Bus Location:"):
                    loc = line.split(":", 1)[1].strip()
                    break
        for cand in (loc, gpus[0]):
            if cand:
                n = _pci_numa_node(sysfs_root, cand)
                if n is not None:
                    return n
    return None


# --------------------------------------------------------------------------------------------------
# Process-wide cache

_topology: NumaTopology | None = None


def get_topology(refresh: bool = False) -> NumaTopology:
    """discover() with EXL3_NUMA_CPU_MAP, cached for the process"""
    global _topology
    if _topology is None or refresh:
        _topology = discover(cpu_map = os.environ.get("EXL3_NUMA_CPU_MAP") or None)
    return _topology


def numa_debug() -> bool:
    return os.environ.get("EXL3_NUMA_DEBUG", "") not in ("", "0")

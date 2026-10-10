"""
NUMA memory placement for the CPU MoE expert offload: thin ctypes wrappers over the Linux mbind,
set_mempolicy, get_mempolicy and move_pages syscalls, plus a per-node free-memory guard.

Each per-node worker binds its weight arena to its own node, so every expert matmul reads local
DRAM instead of crossing the socket interconnect. Binding goes through raw syscalls (libc `syscall`)
rather than libnuma, which is often not installed and would be one more native dependency.

Only Linux on x86_64 and aarch64 is supported (the syscall numbers are per-architecture). Elsewhere
available() is False, the errno-returning calls return ENOSYS without doing anything, and the rest
are no-ops returning False / None. Nothing here raises on a failed syscall: placement is an
optimization, so callers get an errno and decide whether to warn.
"""
from __future__ import annotations
import contextlib
import ctypes
import errno
import mmap
import os
import platform
import sys

MPOL_DEFAULT, MPOL_PREFERRED, MPOL_BIND, MPOL_INTERLEAVE = 0, 1, 2, 3
MPOL_MF_STRICT = 1
MPOL_MF_MOVE = 2

# get_mempolicy() reports the mode with these flag bits ORed in; set_mempolicy() accepts them back
_MPOL_MODE_FLAGS = (1 << 15) | (1 << 14) | (1 << 13)

_SYSCALLS = {
    "x86_64": {"mbind": 237, "set_mempolicy": 238, "get_mempolicy": 239, "move_pages": 279},
    "aarch64": {"mbind": 235, "get_mempolicy": 236, "set_mempolicy": 237, "move_pages": 239},
}

_PAGE = mmap.PAGESIZE

# Mask large enough for any kernel's nr_node_ids when querying a policy (get_mempolicy refuses a mask
# shorter than the number of possible nodes)
_QUERY_MASK_BITS = 1024


def _arch() -> str | None:
    if not sys.platform.startswith("linux"):
        return None
    m = platform.machine().lower()
    m = {"amd64": "x86_64", "arm64": "aarch64"}.get(m, m)
    return m if m in _SYSCALLS else None


_nr = _SYSCALLS.get(_arch() or "")
_syscall = None
if _nr is not None:
    try:
        _libc = ctypes.CDLL(None, use_errno = True)
        _syscall = _libc.syscall
        _syscall.restype = ctypes.c_long
    except (OSError, AttributeError):
        _syscall = None


def available() -> bool:
    """True when the NUMA syscalls can be issued on this platform (they may still fail at runtime, e.g.
    EPERM/ENOSYS under a seccomp filter)"""
    return _syscall is not None


def _call(name: str, *args) -> int:
    """Issue a syscall, 0 on success else the errno. Every argument is passed as a full machine word,
    since `syscall` is variadic and ctypes would otherwise pass Python ints as 32-bit int"""
    ctypes.set_errno(0)
    r = _syscall(ctypes.c_long(_nr[name]), *args)
    if r < 0:
        return ctypes.get_errno() or errno.EINVAL
    return 0


def _nodemask(nodes):
    """
    (mask array, maxnode) for a node or iterable of nodes. The mask covers max node id + 1 rounded up
    to whole 64-bit words; maxnode is that bit count + 1 because the kernel decrements it before use
    """
    if nodes is None:
        return None, 0
    if isinstance(nodes, int):
        nodes = [nodes]
    nodes = sorted(set(int(n) for n in nodes))
    if not nodes:
        return None, 0
    if nodes[0] < 0:
        raise ValueError(f"Invalid NUMA node {nodes[0]}")
    wbits = ctypes.sizeof(ctypes.c_ulong) * 8
    words = (nodes[-1] + wbits) // wbits
    mask = (ctypes.c_ulong * words)()
    for n in nodes:
        mask[n // wbits] |= 1 << (n % wbits)
    return mask, words * wbits + 1


def _mask_ptr(mask):
    return ctypes.cast(mask, ctypes.c_void_p) if mask is not None else ctypes.c_void_p(0)


def mbind(addr: int, length: int, nodes, mode: int = MPOL_BIND, flags: int = 0) -> int:
    """
    Set the memory policy of [addr, addr + length), widened to whole pages. Returns 0 on success, else
    the errno (ENOSYS when unsupported here). Pages already faulted in keep their placement unless flags
    include MPOL_MF_MOVE, so bind before first touch
    """
    if not available():
        return errno.ENOSYS
    if length <= 0:
        return 0
    start = addr & ~(_PAGE - 1)
    end = (addr + length + _PAGE - 1) & ~(_PAGE - 1)
    try:
        mask, maxnode = _nodemask(nodes)
    except (ValueError, TypeError):
        return errno.EINVAL
    return _call("mbind", ctypes.c_ulong(start), ctypes.c_ulong(end - start), ctypes.c_long(mode),
                 _mask_ptr(mask), ctypes.c_ulong(maxnode), ctypes.c_ulong(flags))


def set_mempolicy(mode: int, nodes = None) -> int:
    """Set the calling thread's default memory policy. 0 on success, else the errno"""
    if not available():
        return errno.ENOSYS
    try:
        mask, maxnode = _nodemask(nodes)
    except (ValueError, TypeError):
        return errno.EINVAL
    return _call("set_mempolicy", ctypes.c_long(mode), _mask_ptr(mask), ctypes.c_ulong(maxnode))


def _get_mempolicy() -> tuple[int, list[int]] | None:
    """The calling thread's (mode incl. flag bits, nodes), None if the query fails"""
    if not available():
        return None
    wbits = ctypes.sizeof(ctypes.c_ulong) * 8
    mode = ctypes.c_int(0)
    mask = (ctypes.c_ulong * (_QUERY_MASK_BITS // wbits))()
    err = _call("get_mempolicy", ctypes.c_void_p(ctypes.addressof(mode)), _mask_ptr(mask),
                ctypes.c_ulong(_QUERY_MASK_BITS + 1), ctypes.c_void_p(0), ctypes.c_ulong(0))
    if err:
        return None
    nodes = [i * wbits + b for i, w in enumerate(mask) for b in range(wbits) if (w >> b) & 1]
    return mode.value, nodes


@contextlib.contextmanager
def policy_scope(node: int | None, mode: int = MPOL_BIND):
    """
    Within the block, allocations first touched by the calling thread come from `node` (memory policy is
    per thread). The previous policy is restored on exit, also on exceptions; if it could not be queried,
    or does not restore, the thread goes back to MPOL_DEFAULT. node None, or no support, is a no-op
    """
    if node is None or not available():
        yield
        return
    prev = _get_mempolicy()
    changed = set_mempolicy(mode, [node]) == 0
    try:
        yield
    finally:
        if changed:
            restored = False
            if prev is not None:
                pmode, pnodes = prev
                base = pmode & ~_MPOL_MODE_FLAGS
                if base == MPOL_DEFAULT:
                    restored = set_mempolicy(MPOL_DEFAULT) == 0
                else:
                    restored = set_mempolicy(pmode, pnodes or None) == 0
            if not restored:
                set_mempolicy(MPOL_DEFAULT)


def pages_per_node(addr: int, length: int, sample: int = 64) -> dict[int, int] | None:
    """
    Where a range actually lives: {node: pages} over up to `sample` evenly spaced pages, queried with
    move_pages (nodes = NULL only reports, nothing moves). Pages that are not resident or not mapped
    count under -1. None when unsupported or the query itself fails
    """
    if not available() or length <= 0:
        return None
    start = addr & ~(_PAGE - 1)
    end = (addr + length + _PAGE - 1) & ~(_PAGE - 1)
    n_pages = (end - start) // _PAGE
    count = max(1, min(int(sample), n_pages))
    pages = (ctypes.c_void_p * count)(*[start + (i * n_pages // count) * _PAGE for i in range(count)])
    status = (ctypes.c_int * count)()
    err = _call("move_pages", ctypes.c_long(0), ctypes.c_ulong(count), _mask_ptr(pages),
                ctypes.c_void_p(0), _mask_ptr(status), ctypes.c_long(0))
    if err:
        return None
    out = {}
    for s in status:
        k = s if s >= 0 else -1
        out[k] = out.get(k, 0) + 1
    return out


def _read_node_free(sysfs_root: str, node: int) -> int | None:
    """MemFree + Inactive(file) of a node, bytes, None if unreadable (same estimate as numa_topology)"""
    try:
        with open(os.path.join(sysfs_root, "devices", "system", "node", f"node{node}", "meminfo")) as f:
            text = f.read()
    except (OSError, UnicodeDecodeError):
        return None
    fields = {}
    for line in text.splitlines():
        p = line.split()
        if len(p) >= 4 and p[0] == "Node" and p[2].endswith(":"):
            try:
                fields[p[2][:-1]] = int(p[3]) * (1024 if len(p) > 4 and p[4].lower() == "kb" else 1)
            except ValueError:
                pass
    if "MemFree" not in fields:
        return None
    return fields["MemFree"] + fields.get("Inactive(file)", 0)


def node_free_bytes(node: int, topo = None) -> int | None:
    """
    Free memory on `node` right now (MemFree + Inactive(file)), None when unknown. A topology object
    supplies it through its own node_free_bytes() (NumaTopology re-reads its sysfs root); otherwise it is
    read fresh from /sys, falling back to the topology's snapshot
    """
    fn = getattr(topo, "node_free_bytes", None)
    if callable(fn):
        return fn(node)
    free = _read_node_free("/sys", node)
    if free is None and topo is not None:
        n = topo.node(node)
        free = n.mem_free if n is not None else None
    return free


_check_host_memory = None


def _global_check(nbytes: int, what: str):
    global _check_host_memory
    if _check_host_memory is None:
        if __package__:
            from .memory import check_host_memory
        else:
            # Loaded standalone (by file path, as the tests do): importing the package would run
            # exllamav3/__init__, which needs the compiled extension, so load memory.py on its own
            import importlib.util
            path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "memory.py")
            spec = importlib.util.spec_from_file_location("_exl3_util_memory_standalone", path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            check_host_memory = mod.check_host_memory
        _check_host_memory = check_host_memory
    _check_host_memory(nbytes, what)


def check_node_memory(nbytes: int, node: int | None, what: str, topo = None) -> None:
    """
    Raise before allocating `nbytes` bound to `node` when that node would be left with less than
    EXL3_HOST_MEM_RESERVE_MB (default 2048, <= 0 disables every check) free. A strictly bound allocation
    that does not fit is not spilled to another node: it ends in reclaim storms or the OOM killer, so
    failing early with advice beats a hang. The system-wide check (check_host_memory) runs too; when
    the node's free memory is unknown, or node is None, that is the only check
    """
    if node is None:
        _global_check(nbytes, what)
        return
    reserve_mb = int(os.environ.get("EXL3_HOST_MEM_RESERVE_MB", 2048))
    if reserve_mb <= 0:
        return
    free = node_free_bytes(node, topo)
    if free is not None and nbytes + (reserve_mb << 20) > free:
        raise RuntimeError(
            f"{what} needs {nbytes >> 20} MiB of memory on NUMA node {node}, but only {free >> 20} MiB is "
            f"available there (MemFree + Inactive(file)) and EXL3_HOST_MEM_RESERVE_MB = {reserve_mb} MiB is "
            f"kept free. Offload fewer experts to the CPU, place this part on another node, relax the "
            f"placement with EXL3_NUMA_MEMORY_POLICY (e.g. interleave or preferred, so allocations may spill "
            f"to other nodes), or set EXL3_HOST_MEM_RESERVE_MB=0 to skip this check."
        )
    _global_check(nbytes, what)


_bind_warned = set()


def bind_range_or_warn(addr: int, length: int, node: int, what: str) -> bool:
    """mbind(MPOL_BIND) a range to `node`; on failure print one notice per distinct `what`. Returns
    whether the range is bound. Silent (False) where NUMA binding is not supported at all"""
    if not available():
        return False
    err = mbind(addr, length, [node], MPOL_BIND)
    if err == 0:
        return True
    if what not in _bind_warned:
        _bind_warned.add(what)
        print(f" !! {what}: mbind to node {node} failed ({os.strerror(err)})")
    return False

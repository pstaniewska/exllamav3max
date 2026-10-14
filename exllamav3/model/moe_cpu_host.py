from __future__ import annotations
import errno
import os
import re
import sys
import multiprocessing
from multiprocessing import shared_memory
import numpy as np
import torch

from ..ext import exllamav3_ext as ext
from ..util.device_copy import host_to_device
from ..util.misc import Cleanupper, install_parent_death_signal
from ..util.shm import check_shm_capacity
from ..util.memory import check_host_memory, windows_memory_status
from .moe_cpu_affinity import plan_host_cpus, default_worker_threads, apply_process_affinity, process_cpus
from ..util import numa_mem
from .model_tp_cuda import (
    cuda_host_register,
    cuda_host_unregister,
    cuda_host_get_device_pointer,
    CUDA_HOST_REGISTER_PORTABLE,
    CUDA_HOST_REGISTER_MAPPED,
)

cleanupper = Cleanupper()

"""
Persistent-worker handoff for CPU-offloaded MoE experts, following the native TP backend's
CPU-helper pattern: one spawned child process owns the expert weights and consumes a job ring in
pinned shared memory. The parent's forward pass never blocks on the CPU: for each offloaded
layer it enqueues, onto the CUDA stream, contiguous D2H copies of the staged inputs, a kernel
publishing the job's sequence number, a blocking kernel that waits for the worker's completion
flag, and the contiguous H2D readback (strided GPU<->CPU copies are forbidden here: torch stages
them host-side at enqueue time, outside stream order, which reads/writes the slots at the wrong
time).

Loading is incremental: the child is spawned at the first layer registration and receives layer
specs over a pipe as the parent's loader reaches each MoE layer, loading the expert tensors
concurrently with the parent's GPU loading. After a module commits on the GPU side (no OOM
rollback), the loader waits for the child's ack for that layer, so the progress bar reflects
combined progress and there is no bulk stall at the end. Re-registering a key (autosplit
rollback retry) returns the existing layer index without reloading.

Shared-memory layout constants mirror cpu/moe_handoff.h.
"""

MOE_JOB_RING = 256
MOE_MAX_SLOTS = 8
MOE_JOB_MAX_EXPERTS = 256   # structural capacity of MoeJob.experts[]; see cpu/moe_handoff.h
# sizeof(MoeJob): 7 uint32 fields + experts[MOE_JOB_MAX_EXPERTS] + seq_hi, all 4-byte
# fields so no compiler padding -- recompute this if the struct's fixed fields ever change
MOE_JOB_BYTES = (7 + MOE_JOB_MAX_EXPERTS + 1) * 4
MOE_JOB_SEQ_HI = 7 + MOE_JOB_MAX_EXPERTS   # u32 index of MoeJob.seq_hi
# Flag width (MOE_FLAG_BITS in cpu/moe_handoff.h): ROCm's stream wait has no cyclic compare, so its
# flags are 64-bit and never wrap; CUDA's are 32-bit and compared cyclically
MOE_FLAG_BITS = 64 if torch.version.hip else 32
MOE_CTRL_JOBS_OFFSET = 384
MOE_SLOT_FLAGS_OFFSET = MOE_CTRL_JOBS_OFFSET + MOE_JOB_RING * MOE_JOB_BYTES
MOE_MAX_WSLOTS = 8
MOE_FLAGS_SIZE = 3 * 64 * MOE_MAX_SLOTS + 2 * 64 * MOE_MAX_WSLOTS
MOE_STAGE_RING = 64
MOE_STAGE_TAIL_OFFSET = MOE_SLOT_FLAGS_OFFSET + MOE_FLAGS_SIZE
MOE_STAGE_HEAD_OFFSET = MOE_STAGE_TAIL_OFFSET + 64
MOE_STAGE_JOBS_OFFSET = MOE_STAGE_TAIL_OFFSET + 128
MOE_CTRL_SIZE = MOE_STAGE_JOBS_OFFSET + MOE_STAGE_RING * MOE_JOB_BYTES


def _set_job_seq(job, seq: int):
    """Sequence number into a job descriptor (u32 view) as its two 32-bit halves"""
    job[0] = seq & 0xFFFFFFFF
    job[MOE_JOB_SEQ_HI] = seq >> 32


def _write_flag(u32: np.ndarray, offset: int, value: int):
    """Host-side write of a handoff flag at a byte offset into the region"""
    if MOE_FLAG_BITS == 64:
        u32[offset // 4 : offset // 4 + 2].view(np.uint64)[0] = value
    else:
        u32[offset // 4] = value & 0xFFFFFFFF


def _align64(x):
    return (x + 63) & ~63


def madv_collapse_supported() -> bool:
    """MADV_COLLAPSE landed in Linux 6.1"""
    if not sys.platform.startswith("linux"):
        return False
    m = re.match(r"(\d+)\.(\d+)", os.uname().release)
    return bool(m) and (int(m.group(1)), int(m.group(2))) >= (6, 1)


MADV_POPULATE_WRITE = 23   # Linux 5.14+


class MoeCpuTuning:
    """
    Tunable knobs for the CPU MoE offload path, collected in one place: env vars are read once
    here (at import time) instead of scattered os.environ.get calls, so a config-file migration
    or an automated sweep only has one object to touch. MoeCpuHost copies the values it needs at
    construction time (plain scalars -- this class is never bound into C++; anything the native
    side needs is propagated explicitly as a call or pipe-message parameter).

    For a same-process sweep, mutate fields on the module-level TUNING singleton before
    constructing each MoeCpuHost (each model load constructs a fresh one); env vars only matter
    at the first import.
    """

    def __init__(self):
        # --- CPU worker / staging ---
        self.num_slots = int(os.environ.get("EXL3_MOE_CPU_SLOTS", 4))
        assert 1 <= self.num_slots <= 8, "EXL3_MOE_CPU_SLOTS must be 1..8 (MOE_MAX_SLOTS in moe_handoff.h)"
        self.cap_rows = int(os.environ.get("EXL3_MOE_CPU_SLOT_ROWS", 64))
        # Physical cores kept free of pool workers for the host process, whether the pool pins
        # its workers (EXL3_MOE_CPU_PIN) or lets them float; the host is confined to them once
        # the worker has started, see moe_cpu_affinity.py. 0 disables the reservation
        self.host_cores = int(os.environ.get("EXL3_MOE_HOST_CORES", 1))
        # Thread count fallback chain ends here; config.infer_params.moe_cpu_threads (or the
        # draft/MTP equivalent) takes precedence per host when set (MoeCpuHost.__init__).
        # Default: physical cores minus host_cores; cpu_count/2 when host_cores is 0 or the
        # topology is unreadable
        _, n_phys = ext.exl3_moe_cpu_core_order()
        self.threads = int(os.environ.get(
            "EXL3_MOE_CPU_THREADS",
            default_worker_threads(n_phys, self.host_cores) if n_phys and self.host_cores > 0
            else max(1, (os.cpu_count() or 2) // 2)))
        self.num_wslots = min(int(os.environ.get("EXL3_MOE_CPU_WSLOTS", 2)), MOE_MAX_WSLOTS)
        self.wslot_size = int(os.environ.get("EXL3_MOE_CPU_WSLOT_MB", 32)) * 1024 * 1024
        self.stage_threads = int(os.environ.get("EXL3_MOE_CPU_STAGE_THREADS", 4))
        # madvise(MADV_HUGEPAGE) on the expert-weight arena chunks: with defrag=madvise (the
        # common default), the kernel does SYNCHRONOUS compaction on first touch of a hinted
        # region once easily-compactable free memory runs low, which can stall loading badly.
        # On Windows the same flag makes each arena chunk attempt MEM_LARGE_PAGES at
        # VirtualAlloc time (no post-hoc promotion exists there), negotiating the request
        # size down per chunk and falling back to a plain mapping per chunk
        self.arena_hugepage = os.environ.get("EXL3_MOE_ARENA_HUGEPAGE", "1") != "0"
        # How the Linux arena gets its hugepages: "collapse" = one MADV_COLLAPSE pass after loading
        # (Linux 6.1+), "eager" = MADV_HUGEPAGE on each chunk before its first touch, so faults
        # take 2 MiB pages directly. Kernels without MADV_COLLAPSE (e.g. 5.15) reject the collapse
        # pass, leaving the whole arena on 4K pages, so they default to eager. Shared (memfd)
        # chunks need eager mode on every kernel: fallocate commits their pages without a VMA, which
        # shmem_enabled=advise never backs with hugepages
        mode = os.environ.get("EXL3_MOE_ARENA_HUGEPAGE", "1").strip().lower()
        self.arena_hugepage_eager = mode == "eager" or (mode != "collapse" and not madv_collapse_supported())
        # Packed ("swizzled") expert trellis layout: repacked at arena rehome so the banded
        # CPU kernels' tile groups stream sequentially from DRAM. The group is per matrix and
        # ISA-dependent -- exl3_moe_cpu_swizzle_group is the single rule (band-8 on the
        # AVX-512 tiers, band-2 + planar dwords on AVX2, native on scalar).
        # EXL3_MOE_CPU_SWIZZLE=0 restores the native layout everywhere.
        self.swizzle = os.environ.get("EXL3_MOE_CPU_SWIZZLE", "1") != "0"
        # Experts read per deferred-load pass when the worker loads a layer. Each pass is read
        # into loader tensors and then copied into the arena, so this bounds the transient host
        # memory on top of the arena to a slice of a layer instead of the whole layer (~1.2 GiB
        # per layer on a 512-expert model). 0 loads the whole layer in one pass
        self.load_batch_experts = int(os.environ.get("EXL3_MOE_CPU_LOAD_BATCH", 32))

        # --- GPU-streaming prefill ---
        self.stream_t_explicit = "EXL3_MOE_STREAM_T" in os.environ
        self.stream_t = int(os.environ.get("EXL3_MOE_STREAM_T", 8))
        self.stream_fused_t = int(os.environ.get("EXL3_MOE_STREAM_FUSED_T", 256))
        self.stream_min_rows = int(os.environ.get("EXL3_MOE_STREAM_MIN_ROWS", 32))
        self.batch_experts = max(1, min(
            int(os.environ.get("EXL3_MOE_STREAM_BATCH_EXPERTS", 24)), MOE_JOB_MAX_EXPERTS))

        # --- pinned expert arena (experimental, opt-in) ---
        # EXL3_MOE_PINNED_ARENA=1: the worker's expert arena is memfd-backed and shared with the
        # parent, which maps and page-locks it, so streamed prefill DMAs each expert's block
        # straight out of the arena on the copy stream instead of having the worker's stager
        # thread memcpy it into the pinned handoff ring first (the stager is the prefill
        # bottleneck on fully offloaded models). Costs: the arena is registered with CUDA
        # (~0.2 s per GiB at load), and shmem pages only get transparent hugepages when
        # /sys/kernel/mm/transparent_hugepage/shmem_enabled allows it (advise/always/
        # within_size), so on a default (never) system the CPU kernels run on 4K pages;
        # EXL3_MOE_ARENA_HUGE=2m|1g backs the memfd with hugetlbfs pages instead (requires
        # vm.nr_hugepages / hugepages-1048576kB reservations); Windows uses named sections, no
        # hugepage variant.
        self.pinned_arena = os.environ.get("EXL3_MOE_PINNED_ARENA", "0") != "0"
        self.arena_huge = os.environ.get("EXL3_MOE_ARENA_HUGE", "").strip().lower()
        assert self.arena_huge in ("", "2m", "1g"), "EXL3_MOE_ARENA_HUGE must be 2m or 1g"
        if self.arena_huge and os.name == "nt":
            raise RuntimeError("EXL3_MOE_ARENA_HUGE is Linux-only (hugetlbfs memfd); unset it on Windows")
        # Batched reconstruct tier for the streamed heavy experts (see moe_batch_recon.py):
        # experts too hot for the fused kernel are dequantized in groups with one launch per
        # projection and run through padded bmm. EXL3_MOE_STREAM_BATCH_RECON=0 restores the
        # per-expert loop
        self.stream_batch_recon = os.environ.get("EXL3_MOE_STREAM_BATCH_RECON", "1") != "0"
        # Fused-tier row tiles (32 / 64-row kernel instances per expert range), as EXL3_MOE_MTILE
        # on the GPU side
        self.mtile = os.environ.get("EXL3_MOE_MTILE", "1") != "0"

        # --- debug / kill switches ---
        self.stream_debug = bool(os.environ.get("EXL3_MOE_STREAM_DEBUG"))
        self.cpu_prof = bool(os.environ.get("EXL3_MOE_CPU_PROF"))
        self.memops = os.environ.get("EXL3_MOE_MEMOPS", "1") != "0"


TUNING = MoeCpuTuning()
ext.exl3_moe_cpu_set_memops(TUNING.memops)

# Host placement is per process: planned for the first started host's worker count and kept
# for later hosts (MTP head, draft model, reload). Workers spawned after it restore the
# pre-pin mask (_HOST_ORIG_CPUS) so their pool can pin outside the host's LPs
_HOST_AFFINITY_THREADS: int | None = None
_HOST_ORIG_CPUS: list[int] | None = None


def _apply_host_affinity(threads: int, host_cores: int):
    """Confine the host process to the CPUs the worker pool leaves free. Never raises: an OS
    failure leaves the host unpinned with a notice."""
    global _HOST_AFFINITY_THREADS, _HOST_ORIG_CPUS
    if _HOST_AFFINITY_THREADS is not None:
        if threads > _HOST_AFFINITY_THREADS:
            print(f" -- CPU MoE host affinity: placement planned for {_HOST_AFFINITY_THREADS} workers, "
                  f"kept for a {threads}-thread worker")
        return
    _HOST_AFFINITY_THREADS = threads
    order, n_phys = ext.exl3_moe_cpu_core_order()
    cpus = plan_host_cpus(order, n_phys, threads, host_cores)
    if cpus is None:
        if host_cores > 0 and n_phys:
            print(" -- CPU MoE host affinity: no LP free of workers in one processor group; "
                  "host threads left unpinned")
        return
    try:
        orig = process_cpus()
    except OSError as e:
        print(f" !! CPU MoE host affinity: {e}; host threads left unpinned")
        return
    err = apply_process_affinity(cpus)
    if err is not None:
        print(f" !! CPU MoE host affinity: {err}; host threads left unpinned")
        return
    _HOST_ORIG_CPUS = orig
    print(f" -- CPU MoE host affinity: host on LPs {[enc & 0xFFFF for enc in cpus]}")


# memfd_create only ships in CPython when the interpreter was built against glibc >= 2.27; conda
# and manylinux-built interpreters lack it (and the os.MFD_* constants) even on kernels that
# have had the syscall since 4.17 (PR #341 discussion). Resolve it at runtime instead: os first,
# then libc's symbol, then the raw syscall by architecture. The MFD_* values are kernel ABI
MFD_CLOEXEC = 0x0001
MFD_HUGETLB = 0x0004
MFD_HUGE_2MB = 21 << 26
MFD_HUGE_1GB = 30 << 26
_MEMFD_SYSCALL = {"x86_64": 319, "aarch64": 279, "riscv64": 279, "loongarch64": 279,
                  "ppc64le": 360, "ppc64": 360, "s390x": 350, "i686": 356, "armv7l": 385}


def _memfd_via_libc(name: str, flags: int) -> int:
    """libc's memfd_create symbol (any glibc >= 2.27 / musl >= 1.1.20 at runtime)"""
    import ctypes
    libc = ctypes.CDLL(None, use_errno = True)
    fn = libc.memfd_create   # AttributeError when the runtime libc predates it
    fn.restype = ctypes.c_int
    fn.argtypes = [ctypes.c_char_p, ctypes.c_uint]
    fd = fn(name.encode(), flags)
    if fd < 0:
        err = ctypes.get_errno()
        raise OSError(err, os.strerror(err))
    return fd


def _memfd_via_syscall(name: str, flags: int) -> int:
    """Raw memfd_create syscall, for a runtime libc without the wrapper"""
    import ctypes
    nr = _MEMFD_SYSCALL.get(os.uname().machine)
    if nr is None:
        raise RuntimeError(f"no memfd_create syscall number known for {os.uname().machine}")
    libc = ctypes.CDLL(None, use_errno = True)
    libc.syscall.restype = ctypes.c_long
    fd = libc.syscall(ctypes.c_long(nr), ctypes.c_char_p(name.encode()), ctypes.c_uint(flags))
    if fd < 0:
        err = ctypes.get_errno()
        raise OSError(err, os.strerror(err))
    return int(fd)


def _memfd_create(name: str, flags: int = 0) -> int:
    if hasattr(os, "memfd_create"):
        return os.memfd_create(name, flags)
    try:
        return _memfd_via_libc(name, flags)
    except AttributeError:
        pass
    try:
        return _memfd_via_syscall(name, flags)
    except RuntimeError as e:
        raise RuntimeError(
            f"CPU MoE pinned arena: this Python build has no os.memfd_create and no fallback "
            f"applies ({e}). Unset EXL3_MOE_PINNED_ARENA to use the staged path.") from e


# Windows large pages: there is no madvise/promotion path, so MEM_LARGE_PAGES must be requested
# at VirtualAlloc time. That requires SeLockMemoryPrivilege enabled on this process's token
# (the privilege is granted-but-disabled by default for accounts that have it), and the size
# must be a multiple of GetLargePageMinimum() (2 MiB on x64). Large pages are committed and
# non-pageable, so allocation can fail on a fragmented or busy system -- callers fall back to
# a plain anonymous mapping per chunk. A failed request is expensive to repeat (the kernel
# searches for contiguous memory each time) and contiguity does not come back during a load,
# so the arena remembers how far down the size ladder it had to go (see _HugeArena._new_chunk).

_WIN32_LARGE_PAGE_SUPPORT = None


def _win32_enable_lock_memory_privilege() -> bool:
    """Best-effort SeLockMemoryPrivilege enable on the current process token. Returns True
    only when the privilege ends up enabled."""
    import ctypes
    from ctypes import wintypes

    TOKEN_ADJUST_PRIVILEGES = 0x0020
    SE_PRIVILEGE_ENABLED = 0x00000002

    class LUID(ctypes.Structure):
        _fields_ = [("LowPart", wintypes.DWORD), ("HighPart", wintypes.LONG)]

    class LUID_AND_ATTRIBUTES(ctypes.Structure):
        _fields_ = [("Luid", LUID), ("Attributes", wintypes.DWORD)]

    class TOKEN_PRIVILEGES(ctypes.Structure):
        _fields_ = [("PrivilegeCount", wintypes.DWORD),
                    ("Privileges", LUID_AND_ATTRIBUTES * 1)]

    advapi32 = ctypes.WinDLL("advapi32", use_last_error = True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error = True)
    # GetCurrentProcess returns the -1 pseudo-handle; without a pointer-sized restype ctypes
    # truncates it to 32 bits and OpenProcessToken fails with ERROR_INVALID_HANDLE
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    advapi32.OpenProcessToken.argtypes = [
        wintypes.HANDLE, wintypes.DWORD, ctypes.POINTER(wintypes.HANDLE)]
    advapi32.LookupPrivilegeValueW.argtypes = [
        wintypes.LPCWSTR, wintypes.LPCWSTR, ctypes.POINTER(LUID)]
    advapi32.AdjustTokenPrivileges.argtypes = [
        wintypes.HANDLE, wintypes.BOOL, ctypes.POINTER(TOKEN_PRIVILEGES),
        wintypes.DWORD, ctypes.c_void_p, ctypes.c_void_p]
    token = wintypes.HANDLE()
    if not advapi32.OpenProcessToken(
            kernel32.GetCurrentProcess(),
            TOKEN_ADJUST_PRIVILEGES,
            ctypes.byref(token)):
        return False
    try:
        luid = LUID()
        if not advapi32.LookupPrivilegeValueW(None, "SeLockMemoryPrivilege", ctypes.byref(luid)):
            return False
        tp = TOKEN_PRIVILEGES()
        tp.PrivilegeCount = 1
        tp.Privileges[0] = LUID_AND_ATTRIBUTES(luid, SE_PRIVILEGE_ENABLED)
        advapi32.AdjustTokenPrivileges(token, False, ctypes.byref(tp), 0, None, None)
        # AdjustTokenPrivileges returns success even when it silently dropped privileges it
        # couldn't assign; GetLastError distinguishes full assignment from partial
        return ctypes.get_last_error() == 0
    finally:
        kernel32.CloseHandle(token)


def _win32_large_page_alloc(size: int, min_size: int):
    """Allocate MEM_LARGE_PAGES memory of up to `size` bytes, halving the request on each
    failure down to `min_size`, and return the largest buffer obtained (a ctypes byte array
    bound to the allocation -- buffer-protocol compatible, so it drops into the same
    memoryview / torch.frombuffer consumers as an mmap object), or None when even `min_size`
    cannot be supplied. A large request needs that many physically contiguous 2 MiB regions,
    which a fragmented or busy system often cannot supply even when smaller runs exist, so a
    shrunken chunk is still a win over falling back to 4K pages outright. A weakref.finalize
    on the array issues VirtualFree(MEM_RELEASE) when the last reference dies, giving the
    chunk the same free-on-GC lifetime semantics as an mmap object."""
    import ctypes
    import weakref

    global _WIN32_LARGE_PAGE_SUPPORT
    kernel32 = ctypes.WinDLL("kernel32", use_last_error = True)
    kernel32.GetLargePageMinimum.restype = ctypes.c_size_t

    if _WIN32_LARGE_PAGE_SUPPORT is None:
        large_page_min = kernel32.GetLargePageMinimum()
        _WIN32_LARGE_PAGE_SUPPORT = bool(large_page_min) and _win32_enable_lock_memory_privilege()
        if not _WIN32_LARGE_PAGE_SUPPORT and os.environ.get("EXL3_MOE_ARENA_DEBUG"):
            print(" -- arena: MEM_LARGE_PAGES unavailable "
                  f"(GetLargePageMinimum={large_page_min}, SeLockMemoryPrivilege "
                  "not enabled); arena chunks will use regular pages", flush = True)
    if not _WIN32_LARGE_PAGE_SUPPORT:
        return None

    granularity = kernel32.GetLargePageMinimum()
    MEM_RESERVE = 0x2000
    MEM_COMMIT = 0x1000
    MEM_LARGE_PAGES = 0x20000000
    PAGE_READWRITE = 0x04
    kernel32.VirtualAlloc.restype = ctypes.c_void_p
    kernel32.VirtualAlloc.argtypes = [
        ctypes.c_void_p, ctypes.c_size_t, ctypes.c_ulong, ctypes.c_ulong]
    kernel32.VirtualFree.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_ulong]

    want = (size + granularity - 1) // granularity * granularity
    floor = (min_size + granularity - 1) // granularity * granularity
    addr = None
    while True:
        addr = kernel32.VirtualAlloc(
            None, want, MEM_RESERVE | MEM_COMMIT | MEM_LARGE_PAGES, PAGE_READWRITE)
        if addr or want <= floor:
            break
        want = max(want >> 1, floor)
    if not addr:
        return None
    buf = (ctypes.c_uint8 * want).from_address(addr)
    weakref.finalize(buf, kernel32.VirtualFree, ctypes.c_void_p(addr), 0, 0x8000)  # MEM_RELEASE
    if want < size and os.environ.get("EXL3_MOE_ARENA_DEBUG"):
        print(f" -- arena: MEM_LARGE_PAGES negotiated down to {want >> 20} MiB "
              f"(asked {size >> 20} MiB)", flush = True)
    return buf


def _copy_repacked(dst, tensor, group, planar):
    """Copy a [k/16, n/16, 16K] trellis tensor into a same-shape view, applying the packed
    CPU layout (exl3_moe_cpu_swizzle_group / exl3_moe_cpu_planar_layout): tile (kt, nt) moves
    to (nt/group, kt, nt%group), and with planar each tile's dwords are transposed -- dword w
    to 8*(w%bits) + w/bits, i.e. the trailing (8, bits) dword dims swap (integer rates only,
    where group is nonzero). One strided copy_; group 0 is a plain copy."""
    tk, tn, ps = tensor.shape
    if not group:
        dst.copy_(tensor)
    elif planar:
        wd = ps // 2
        dst.view(torch.int32).view(tn // group, tk, group, wd // 8, 8).copy_(
            tensor.view(torch.int32).view(tk, tn // group, group, 8, wd // 8).permute(1, 0, 2, 4, 3))
    else:
        dst.view(tn // group, tk, group, ps).copy_(
            tensor.view(tk, tn // group, group, ps).permute(1, 0, 2, 3))


def _mmap_addr(m) -> int:
    """Base address of an mmap object (the temporary buffer export is released again, so the
    mapping can still be closed later)"""
    import ctypes
    buf = (ctypes.c_char * 1).from_buffer(m)
    try:
        return ctypes.addressof(buf)
    finally:
        del buf


class _HugeArena:
    """
    Growable pool of large (default 1 GiB) anonymous mmap chunks that expert weights are copied
    into, so hugepage promotion (see promote_hugepages) has few, large regions to work with
    instead of thousands of separate small (sub-2MB) loader allocations
    """
    CHUNK_BYTES = 1 << 30   # 1 GiB
    WIN32_LARGE_FLOOR = 64 << 20   # smallest MEM_LARGE_PAGES chunk worth having (Windows)
    CHECK_STEP = 256 << 20   # host-memory guard granularity for lazily committed chunks

    def __init__(self, shared = False, huge = "", conn = None, node = None, kind = "chunk"):
        """shared: back each chunk with shared memory and publish it over `conn` as
        (kind, index, size, name) so the parent can map the same pages and page-lock them
        for DMA: Linux sends a memfd descriptor after the message (SCM_RIGHTS, name = None),
        Windows a named pagefile section. huge: "2m"/"1g" requests hugetlbfs memfds (Linux).
        node: NUMA node the chunks' pages are bound to (Linux; None = default policy). kind:
        message tag, "chunk" for the worker's expert arena, "home_chunk" for the native-layout
        home copy the GPU reads (hybrid MoE)."""
        self.shared = shared
        self.huge = huge
        self.conn = conn
        self.node = node
        self.kind = kind
        self.chunks = []
        self.cur = None
        self.cur_off = 0
        # A private anonymous chunk (Linux, not shared) only takes RAM for the pages rehome()
        # writes, and the last chunk of a load is usually mostly unused, so the host-memory
        # guard runs on bytes written (in CHECK_STEP slices) rather than on whole chunks.
        # Shared, hugetlb and Windows chunks are committed up front and keep the per-chunk check
        self.lazy = not shared and os.name != "nt"
        self.written = 0
        self.checked = 0
        self.win32_large_bytes = 0   # bytes of chunks backed by MEM_LARGE_PAGES (Windows)
        # Largest MEM_LARGE_PAGES request still worth making (Windows), None until the first
        # attempt: the size the last chunk was served at, or below the smallest size that failed
        self.win32_large_ceiling = None

    def _new_chunk(self, min_bytes):
        import mmap, os
        size = max(self.CHUNK_BYTES, (min_bytes + (2 << 20) - 1) & ~((2 << 20) - 1))
        if not self.lazy:
            numa_mem.check_node_memory(
                size, self.node, f"CPU MoE expert {'home' if self.kind != 'chunk' else 'arena'} chunk "
                                 f"{len(self.chunks)} ({(sum(len(c) for c in self.chunks) + size) >> 20} MiB in total)")
        if self.shared and os.name == "nt":
            # Named pagefile-backed section as a plain mmap (a SharedMemory owner's finalizer
            # trips on the layer tensors' exports at worker exit). It charges commit and gets
            # page-locked by the parent, so both free RAM and commit must cover the chunk
            index = len(self.chunks)
            avail_phys, avail_commit = windows_memory_status()
            if size > min(avail_phys, avail_commit):
                raise RuntimeError(
                    f"CPU MoE pinned arena: chunk {index} needs {size >> 20} MiB, but only "
                    f"{avail_phys >> 20} MiB of physical RAM and {avail_commit >> 20} MiB of commit "
                    f"are available. Free RAM, offload fewer experts, or unset EXL3_MOE_PINNED_ARENA.")
            # Unique per arena kind: CreateFileMapping opens an existing section of the same
            # name instead of failing, so a shared name would alias the home copy and the
            # worker arena of one process
            name = f"exl3_moe_{self.kind}_{os.getpid()}_{index}"
            try:
                m = mmap.mmap(-1, size, tagname = name)
            except OSError as e:
                raise RuntimeError(
                    f"CPU MoE pinned arena: cannot create the {size >> 20} MiB section for chunk "
                    f"{index} ({e.strerror}). Free RAM or commit, or unset EXL3_MOE_PINNED_ARENA.") from e
            if self.conn is not None:
                self.conn.send((self.kind, index, size, name))
        elif self.shared:
            flags = 0
            if self.huge == "1g":
                size = (size + (1 << 30) - 1) & ~((1 << 30) - 1)
                flags = MFD_HUGETLB | MFD_HUGE_1GB
            elif self.huge == "2m":
                flags = MFD_HUGETLB | MFD_HUGE_2MB
            fd = _memfd_create(f"exl3_moe_{self.kind}_{len(self.chunks)}", flags)
            eager = not self.huge and TUNING.arena_hugepage and TUNING.arena_hugepage_eager
            try:
                if eager:
                    # Commit through the mapping instead (MADV_POPULATE_WRITE also reports
                    # ENOMEM up front): faults on a MADV_HUGEPAGE VMA take 2 MiB shmem pages
                    # under shmem_enabled=advise, and follow the node binding of the range
                    os.ftruncate(fd, size)
                    m = mmap.mmap(fd, size, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)
                    if self.node is not None:
                        numa_mem.bind_range_or_warn(_mmap_addr(m), size, self.node, f"CPU MoE {self.kind}")
                    try:
                        m.madvise(mmap.MADV_HUGEPAGE)
                    except Exception:
                        pass
                    try:
                        m.madvise(MADV_POPULATE_WRITE)
                    except OSError as e:
                        if e.errno != errno.EINVAL:
                            m.close()
                            raise
                        with numa_mem.policy_scope(self.node):   # kernel < 5.14
                            os.posix_fallocate(fd, 0, size)
                else:
                    # Preallocate: a hugetlb memfd without enough reserved pages fails here with
                    # ENOMEM instead of SIGBUS on first touch. fallocate allocates the shmem pages
                    # itself (not through a mapping), so the node binding is this thread's policy
                    # for the duration of the call; the mapping is bound as well for later faults
                    with numa_mem.policy_scope(self.node):
                        os.posix_fallocate(fd, 0, size)
                    m = mmap.mmap(fd, size, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)
                    if self.node is not None:
                        numa_mem.bind_range_or_warn(_mmap_addr(m), size, self.node, f"CPU MoE {self.kind}")
            except OSError as e:
                os.close(fd)
                raise RuntimeError(
                    f"CPU MoE pinned arena: cannot allocate a {size >> 20} MiB "
                    f"{'hugetlb ' if self.huge else ''}memfd chunk ({e.strerror}). "
                    + ("Reserve hugepages (vm.nr_hugepages / hugepages-1048576kB) or unset "
                       "EXL3_MOE_ARENA_HUGE." if self.huge else
                       "Check the memory cgroup limit, or unset EXL3_MOE_PINNED_ARENA.")) from e
            if not self.huge and TUNING.arena_hugepage and not eager:
                # Honoured only where shmem_enabled permits (advise/always/within_size), and
                # only for pages faulted from now on, not the ones fallocate committed
                try:
                    m.madvise(mmap.MADV_HUGEPAGE)
                except Exception:
                    pass
            if self.conn is not None:
                # SCM_RIGHTS over the pipe's socketpair. socket.send_fds rather than
                # multiprocessing.reduction.send_handle: the latter blocks for an
                # acknowledgement byte, which would stall the worker's loading until the
                # parent next pumps the pipe
                import socket
                self.conn.send((self.kind, len(self.chunks), size, None))
                with socket.socket(fileno = os.dup(self.conn.fileno())) as sock:
                    socket.send_fds(sock, [b"F"], [fd])
            os.close(fd)   # the mapping keeps the pages alive
        elif os.name == "nt":
            # mmap.MAP_PRIVATE / mmap.PROT_* don't exist on Windows; an anonymous mapping is
            # writable by default there. Windows has no post-hoc hugepage promotion, so the
            # EXL3_MOE_ARENA_HUGEPAGE knob is honoured here instead: each chunk first tries a
            # MEM_LARGE_PAGES VirtualAlloc (needs SeLockMemoryPrivilege and physically
            # contiguous 2 MiB regions), halving the request down to 64 MiB -- and no lower
            # than what the placement needs -- before falling back to a plain mapping, still
            # at full chunk size since regular pages have no contiguity constraint. Later
            # chunks start at the size the previous one was served at instead of walking the
            # ladder from the top again, and stop asking once the smallest size has failed
            m = None
            if TUNING.arena_hugepage:
                floor = max(self.WIN32_LARGE_FLOOR, min_bytes)
                ceiling = self.win32_large_ceiling
                if ceiling is None or ceiling >= floor:
                    m = _win32_large_page_alloc(size if ceiling is None else min(size, ceiling), floor)
                    self.win32_large_ceiling = len(m) if m is not None else floor // 2
            if m is not None:
                self.win32_large_bytes += len(m)
            else:
                m = mmap.mmap(-1, size)
        else:
            m = mmap.mmap(-1, size, mmap.MAP_PRIVATE, mmap.PROT_READ | mmap.PROT_WRITE)
            if self.node is not None:
                # Private anonymous pages are placed at first touch (rehome's copy); bind first
                numa_mem.bind_range_or_warn(_mmap_addr(m), size, self.node, f"CPU MoE {self.kind}")
            if TUNING.arena_hugepage and TUNING.arena_hugepage_eager:
                # Before the first touch, so rehome's copy faults 2 MiB pages directly
                try:
                    m.madvise(mmap.MADV_HUGEPAGE)
                except Exception:
                    pass
        self.chunks.append(m)
        self.cur = m
        self.cur_off = 0
        if os.environ.get("EXL3_MOE_ARENA_DEBUG"):
            total = sum(len(c) for c in self.chunks)
            print(f" -- arena: new chunk {len(m)/1e6:.1f} MB, {len(self.chunks)} chunks, "
                  f"{total/1e9:.3f} GB total", flush = True)

    def reserve(self, nbytes):
        """Make sure the next `nbytes` of rehomes land contiguously in the current chunk;
        returns the (chunk index, byte offset) they will start at"""
        aligned = (nbytes + 63) & ~63
        if self.cur is None or self.cur_off + aligned > len(self.cur):
            self._new_chunk(aligned)
        return len(self.chunks) - 1, self.cur_off

    def promote_hugepages(self):
        """One-shot MADV_COLLAPSE (Linux 6.1+) over each chunk, meant to run once after all
        expert weights are loaded. Deliberately NOT done via a live MADV_HUGEPAGE hint during the
        per-tensor writes in rehome(): with the common defrag=madvise policy, that hint makes the
        kernel do SYNCHRONOUS compaction on every first touch of a hinted region once easily-
        compactable free memory runs low, which turned into multi-second stalls per offloaded
        layer partway through a large model's load. A single explicit collapse pass after
        loading gets the same steady-state throughput benefit without blocking incremental
        per-layer progress. Best-effort: silently leaves chunks at 4K pages if collapse fails or
        the kernel doesn't support it."""
        import mmap, os, time
        if not TUNING.arena_hugepage:
            return
        if os.name == "nt":
            # MEM_LARGE_PAGES is decided at VirtualAlloc time (see _new_chunk); there is no
            # promotion step to run here, only coverage to report. Large pages are locked in
            # RAM, which the user did not ask for explicitly, so say so whenever they are in
            # use; an account without the privilege gets regular pages and no message
            total = sum(len(c) for c in self.chunks)
            if self.win32_large_bytes:
                print(f" -- CPU MoE arena: {self.win32_large_bytes/1e9:.2f} GB of {total/1e9:.2f} GB "
                      f"on large pages (locked in RAM, never paged out; set "
                      f"EXL3_MOE_ARENA_HUGEPAGE=0 to use regular pages)", flush = True)
            elif _WIN32_LARGE_PAGE_SUPPORT and total:
                print(" -- CPU MoE arena: no large pages could be allocated, using regular pages "
                      "(physical memory is too fragmented; large pages are usually available "
                      "again after a reboot)", flush = True)
            return
        if TUNING.arena_hugepage_eager:
            return   # chunks were hinted before their first touch (see _new_chunk)
        collapse = getattr(mmap, "MADV_COLLAPSE", 25)
        t0 = time.perf_counter()
        for c in self.chunks:
            try:
                c.madvise(collapse)
            except Exception:
                pass
        if os.environ.get("EXL3_MOE_ARENA_DEBUG"):
            print(f" -- arena: MADV_COLLAPSE issued on {len(self.chunks)} chunks "
                  f"in {time.perf_counter() - t0:.1f} s", flush = True)

    def rehome(self, tensor, band_swizzle = 0, planar = False):
        """Copy `tensor` into the arena and return a same-dtype/shape view over the copy. The
        arena outlives every tensor it hands out (held for the process lifetime), so the
        returned view stays valid.

        band_swizzle/planar: repack a [k/16, n/16, 16K] trellis tensor with the given packed
        layout during the copy (see _copy_repacked). The returned view keeps the original
        logical shape; only the byte order differs (consumed by the swz-aware kernels in
        moe_mul1.cpp)."""
        import torch
        if tensor is None or tensor.numel() == 0:
            return tensor
        nbytes = tensor.numel() * tensor.element_size()
        aligned = (nbytes + 63) & ~63
        if self.lazy:
            if self.written + aligned > self.checked:
                step = max(aligned, self.CHECK_STEP)
                # Lazy chunks are still mbind-ed to the worker's node (MPOL_BIND, strict), so
                # the node itself must hold them, not just the system
                numa_mem.check_node_memory(
                    step, self.node, f"CPU MoE expert arena ({(self.written + step) >> 20} MiB in total)")
                self.checked = self.written + step
            self.written += aligned
        if self.cur is None or self.cur_off + aligned > len(self.cur):
            self._new_chunk(aligned)
        off = self.cur_off
        self.cur_off += aligned
        buf = memoryview(self.cur)[off : off + nbytes]
        dst = torch.frombuffer(buf, dtype = torch.uint8)
        if band_swizzle or planar:
            _copy_repacked(dst.view(tensor.dtype).view(tensor.shape), tensor, band_swizzle, planar)
        else:
            dst.copy_(tensor.contiguous().view(torch.uint8).reshape(-1))
        return dst.view(tensor.dtype).view(tensor.shape)


def _moe_cpu_child_main(conn, model_dir, threads, stage_threads, pinned = False, huge = "", cpus = None,
                        mem_node = None, home = False, home_node = None):
    """
    Child entry point: receives ("layer", spec) messages, loading each layer's expert tensors
    (deferred, multithreaded) and acking, until ("start", shm_name, layout) switches it into the
    worker loop. Errors are reported over the pipe before exiting.

    NUMA (hybrid MoE / EXL3_NUMA_ENABLE): cpus confines the whole worker to one node's CPUs (the
    pool's core order follows the process mask), mem_node binds its expert arena to that node
    and makes the node this process's preferred node for everything else. home: also keep a
    native-layout copy of every expert's trellis block in shared chunks bound to home_node (the
    GPU's node when known), published as ("home_chunk", ...) for the parent to map and register
    for zero-copy GPU reads; the copy is written once at load and never changed.
    """
    import ctypes
    import signal
    import traceback
    import torch  # noqa: F401
    from ..ext import exllamav3_ext as cext
    from ..loader.safetensors import SafetensorsCollection
    from ..util.misc import install_parent_death_signal as ipds

    # Terminal Ctrl-C is delivered to the whole foreground process group; shutdown is
    # orchestrated by the parent (quit flag) or the kernel (PDEATHSIG), never by SIGINT
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    ipds()

    if cpus is not None:
        # Spawned after the host was confined: drop the inherited host mask, or the pool's
        # worker pins fail (Windows refuses a thread mask outside the process mask)
        err = apply_process_affinity(cpus)
        if err is not None:
            print(f" !! CPU MoE worker affinity: {err}; worker confined to the host's LPs")

    if mem_node is not None and numa_mem.available():
        r = numa_mem.set_mempolicy(numa_mem.MPOL_PREFERRED, [mem_node])
        if r:
            print(f" !! CPU MoE worker: preferred memory node {mem_node} not set ({os.strerror(r)})")

    if os.name == "nt":
        # Hold 1 ms timer resolution for this process (per-process since Windows 10 2004): the
        # worker's poll loops back off to 50 us sleeps, which otherwise round up to the default
        # 15.6 ms quantum -- one quantum per job-ring poll miss lands directly on token latency
        try:
            ctypes.WinDLL("winmm").timeBeginPeriod(1)
        except Exception:
            pass

    shm = None
    try:
        stc = SafetensorsCollection(model_dir)
        cpu = torch.device("cpu")
        # Pinned mode: memfd-backed chunks, published to the parent as they are created
        arena = _HugeArena(shared = pinned, huge = huge, conn = conn if pinned else None, node = mem_node)
        # Native-layout home copy for the GPU (hybrid MoE), always shared with the parent
        home_arena = _HugeArena(shared = True, huge = huge, conn = conn, node = home_node,
                                kind = "home_chunk") if home else None

        def fetch(keys):
            out = []
            for k in keys:
                trellis = stc.get_tensor(k + ".trellis", cpu)
                suh = stc.get_tensor(k + ".suh", cpu, float2half = True)
                svh = stc.get_tensor(k + ".svh", cpu, float2half = True)
                bias = stc.get_tensor(k + ".bias", cpu, optional = True, float2half = True)
                out.append((trellis, suh, svh, bias))
            return out

        # Repack the trellis copies into the packed layout the kernel tier wants for each
        # matrix's bitrate (exl3_moe_cpu_swizzle_group / exl3_moe_cpu_planar_layout; group 0
        # keeps native order: scalar tier, half rates on AVX2, K8 on the AVX-512 tiers)
        swz = TUNING.swizzle

        def rehome_trellis(t):
            grp = cext.exl3_moe_cpu_swizzle_group(t.shape[2] / 16.0) if swz else 0
            pln = grp and cext.exl3_moe_cpu_planar_layout(t.shape[2] / 16.0)
            return arena.rehome(t, band_swizzle = grp, planar = pln)

        def nbytes(t):
            return t.numel() * t.element_size()

        def rehome_experts(g, u, d, home_mask = None):
            """Per expert: the gate/up/down trellis tensors back to back (one contiguous
            expert block in (g, u, d) order, the layout the stager produces and the DMA unit
            of the pinned-arena streamed path), then the small aux tensors. Returns the
            per-projection lists, the per-expert (chunk, byte offset) of each block and, with a
            home arena, the (chunk, byte offset) of each expert's native-layout home block (None
            for experts home_mask leaves out)."""
            gs, us, ds, blocks, homes = [], [], [], [], []
            for e in range(len(u)):
                projs = ([g[e]] if g else []) + [u[e], d[e]]
                total = sum(nbytes(t[0]) for t in projs)
                assert all(nbytes(t[0]) % 64 == 0 for t in projs), "trellis size not 64-byte aligned"
                if home_arena is not None:
                    if home_mask is None or home_mask[e]:
                        # Same bytes as the checkpoint (no CPU repack): the GPU kernels read this
                        homes.append(home_arena.reserve(total))
                        for t in projs:
                            home_arena.rehome(t[0])
                    else:
                        homes.append(None)
                blocks.append(arena.reserve(total))
                trellis = [rehome_trellis(t[0]) for t in projs]
                aux = [(arena.rehome(t[1]), arena.rehome(t[2]), arena.rehome(t[3])) for t in projs]
                if g:
                    gs.append((trellis[0],) + aux[0])
                us.append((trellis[-2],) + aux[-2])
                ds.append((trellis[-1],) + aux[-1])
            return gs, us, ds, blocks, homes

        def biases(ts):
            return [t[3] for t in ts] if ts and ts[0][3] is not None else []

        # Per-layer, per-expert arena views retained for runtime expert installs (dynamic
        # placement): the arena memory is what the compute kernels read, so an in-place copy
        # into these views (with the same swizzle transform rehome applied) replaces an
        # expert's weights. The parent quiesces (full sync, ring drained) before sending an
        # install, so the workers never observe a torn write
        layer_views = []

        while True:
            msg = conn.recv()
            if msg[0] == "layer":
                spec = msg[1]
                num_experts = len(spec["up_keys"])
                batch = TUNING.load_batch_experts or num_experts
                g, u, d, blocks, hblocks = [], [], [], [], []
                for e0 in range(0, num_experts, batch):
                    sl = slice(e0, e0 + batch)
                    stc.begin_deferred_load()
                    bg = fetch(spec["gate_keys"][sl])
                    bu = fetch(spec["up_keys"][sl])
                    bd = fetch(spec["down_keys"][sl])
                    stc.end_deferred_load()
                    # Copy into the hugepage-backed arena now that the deferred reads have
                    # actually populated these tensors; the loader tensors die with this batch
                    hm = spec.get("home_mask")
                    bg, bu, bd, bb, bh = rehome_experts(bg, bu, bd, hm[sl] if hm is not None else None)
                    g += bg
                    u += bu
                    d += bd
                    blocks += bb
                    hblocks += bh
                cext.exl3_moe_cpu_make_layer(
                    [t[0] for t in g], [t[1] for t in g], [t[2] for t in g],
                    [t[0] for t in u], [t[1] for t in u], [t[2] for t in u],
                    [t[0] for t in d], [t[1] for t in d], [t[2] for t in d],
                    biases(g), biases(u), biases(d),
                    spec["activation"], spec["act_limit"],
                    1 if swz else 0,
                )
                layer_views.append((g, u, d))
                # Reclaim this layer's now-discarded loader tensors immediately (rehome_experts
                # copied everything into the arena)
                try:
                    ctypes.CDLL(None).malloc_trim(0)
                except Exception:
                    pass
                # The ack carries the expert block locations (only meaningful to a parent that
                # maps the arena)
                conn.send(("ok", blocks if pinned else None, hblocks if home else None))
            elif msg[0] == "start":
                shm_name, layout = msg[1], msg[2]
                break
            elif msg[0] == "quit":
                return

        stc.close()
        shm = shared_memory.SharedMemory(name = shm_name)
        base = np.frombuffer(shm.buf, dtype = np.uint8).ctypes.data
        cext.exl3_moe_cpu_set_prof(layout.get("cpu_prof", False))

        def install(li, ei, keys):
            """Replace expert ei of layer li with the checkpoint tensors at `keys` (one key
            per projection, gate omitted for gateless layers), in place in the arena."""
            views = layer_views[li]
            projs = views if len(keys) == 3 else views[1:]
            for key, plist in zip(keys, projs):
                v_tr, v_suh, v_svh, v_bias = plist[ei]
                new = stc.get_tensor(key + ".trellis", cpu)
                assert new.shape == v_tr.shape, f"install shape mismatch: {key}"
                grp = cext.exl3_moe_cpu_swizzle_group(new.shape[2] / 16.0) if swz else 0
                if grp:
                    _copy_repacked(v_tr, new, grp, cext.exl3_moe_cpu_planar_layout(new.shape[2] / 16.0))
                else:
                    v_tr.copy_(new)
                v_suh.copy_(stc.get_tensor(key + ".suh", cpu, float2half = True))
                v_svh.copy_(stc.get_tensor(key + ".svh", cpu, float2half = True))
                if v_bias is not None:
                    v_bias.copy_(stc.get_tensor(key + ".bias", cpu, float2half = True))

        # The compute loop runs on its own thread (worker_run releases the GIL); the main
        # thread keeps serving the pipe for runtime installs
        import threading
        worker = threading.Thread(
            target = cext.exl3_moe_cpu_worker_run,
            args = (
                base,
                layout["num_slots"], layout["slot_size"], layout["cap_rows"],
                layout["max_hi"], layout["max_ho"], layout["max_topk"],
                layout["wstage_off"], layout["num_wslots"], layout["wslot_size"],
                threads, stage_threads, layout.get("telemetry_offset", -1),
            ),
            daemon = True,
        )
        worker.start()

        # Hugepage promotion runs off the startup path: MADV_COLLAPSE is synchronous and copies
        # the whole arena into 2 MiB pages (tens of seconds for a 50+ GiB arena, minutes when
        # free memory is fragmented and the kernel has to compact first), and the parent's
        # startup wait must not depend on it. Page migration is transparent to the compute
        # threads, so the worker serves requests on 4K pages until each chunk lands
        threading.Thread(target = arena.promote_hugepages, daemon = True).start()
        if home_arena is not None:
            threading.Thread(target = home_arena.promote_hugepages, daemon = True).start()

        while True:
            try:
                msg = conn.recv()
            except EOFError:
                break
            if msg[0] == "install":
                try:
                    install(msg[1], msg[2], msg[3])
                    conn.send(("ok",))
                except Exception:
                    conn.send(("err", traceback.format_exc()))
            elif msg[0] == "quit":
                break
        worker.join(timeout = 2.0)
    except Exception:
        try:
            conn.send(("err", traceback.format_exc()))
        except Exception:
            pass
        raise
    finally:
        if shm is not None:
            shm.close()


def probe_bandwidth(timed_copy, floor_s = 0.5, cap_s = 2.0, asleep_gbs = 5.0):
    """Sustained pinned->device rate in GB/s from repeated `timed_copy()` calls. An idle link
    sits at Gen1 (the Windows driver drops it after a few idle seconds) and retrains only after
    0.2-0.3 s of sustained traffic, so: copy for at least floor_s, then until the last 8 copies
    are within 5% of the best, up to cap_s while the rate still looks asleep. Median of the
    last 8, so one copy straddling the retrain step cannot win."""
    import time
    t0 = time.perf_counter()
    best, trace = 0.0, []
    while True:
        rate = timed_copy()
        trace.append(rate)
        best = max(best, rate)
        t = time.perf_counter() - t0
        tail = trace[-8:]
        steady = len(tail) == 8 and min(tail) >= 0.95 * best
        if t >= cap_s or (t >= floor_s and steady and best >= asleep_gbs):
            return sorted(tail)[len(tail) // 2]


def run_staged_experts(y, out_ext, items, token_sorted, weight_sorted, base, stride, proj_off, gated,
                       scales, Ks, activation, act_limit, cb, fused_t, fbufs, mtile, recon, ctx):
    """GPU tiers over one batch of experts whose weights sit in a VRAM staging buffer, all on the
    current stream (the caller has made it wait for the buffer's DMA). Shared by the streamed
    prefill of -mcl / -mcs (_submit_prefill_streamed) and the hybrid prefill stream lane
    (moe_hybrid_dispatch.StreamWork).

    items: per expert (buffer position, expert id, segment start in token_sorted /
    weight_sorted, rows). Position j's native-layout block starts at base + j * stride, its
    projections at proj_off (g, u, d) within it. scales(e): the expert's (suh_g, svh_g, suh_u,
    svh_u, suh_d, svh_d) addresses (gate entries ignored when gateless). Ks: (Kg, Ku, Kd); cb:
    (mcg, mul1) per projection, flattened. Experts with up to fused_t rows run through one
    fused exl3_moe launch per row tile (fbufs: its temp buffers, None = no fused tier); the
    rest through batched reconstruct groups (recon, None = no batched tier) up to recon.max_rows
    rows. ctx: per-call dict caching the batched tier's padded inputs across batches.
    Accumulates weighted into out_ext (rows + 1, the last row the batched tier's padding sink).
    Returns the (position, expert) pairs neither tier took, for the caller's own fallback"""
    from ..modules.moe_batch_recon import plan_groups
    rows = out_ext.shape[0] - 1
    out = out_ext[:rows]
    g_off, u_off, d_off = proj_off

    # Mid tier: one fused kernel over the batch's cooler experts. Heavy experts stay in the
    # descriptor (the kernel skips counts above the temp-row capacity) so the token segments
    # line up with expert_count
    fc = [c for _, _, _, c in items if c <= fused_t] if fbufs is not None and fused_t else []
    if fc:
        tbl = [[] for _ in range(9)]
        for j, e, _, _ in items:
            bb = base + j * stride
            sg, vg, su, vu, sd, vd = scales(e)
            if gated:
                tbl[0].append(bb + g_off)
                tbl[1].append(sg)
                tbl[2].append(vg)
            tbl[3].append(bb + u_off)
            tbl[4].append(su)
            tbl[5].append(vu)
            tbl[6].append(bb + d_off)
            tbl[7].append(sd)
            tbl[8].append(vd)
        if not gated:
            # Placeholder gate tables, never dereferenced (gate GEMM is skipped)
            for i in (0, 1, 2):
                tbl[i] = tbl[i + 3]
        tblt = host_to_device(torch.tensor(tbl, dtype = torch.int64), y.device)
        ec = host_to_device(torch.tensor([c for _, _, _, c in items] + [0], dtype = torch.long), y.device)
        tok = torch.cat([token_sorted[s : s + c] for _, _, s, c in items])
        wts = torch.cat([weight_sorted[s : s + c] for _, _, s, c in items]).half()
        Kg, Ku, Kd = Ks
        # Row-tile tiers as on the GPU side (block_sparse_mlp): one launch per tile over its
        # expert range
        t1 = sum(1 for c in fc if 16 < c <= 32)
        t2 = sum(1 for c in fc if c > 32)
        tiers = [(t2, 33, fused_t, 64), (t1, 17, 32, 32), (len(fc) - t1 - t2, 1, 16, 16)] \
            if mtile and (t1 or t2) else [(len(fc), 1, fused_t, 16)]
        for n_act, lo, hi, mt in tiers:
            if not n_act:
                continue
            ext.exl3_moe(
                y, out, ec, tok, wts,
                fbufs[0], fbufs[1], fbufs[2], fbufs[3],
                activation, Kg, Ku, Kd,
                tblt[0], tblt[1], tblt[2], tblt[3], tblt[4], tblt[5],
                tblt[6], tblt[7], tblt[8],
                *cb,
                act_limit, n_act, None, None, lo, hi, mt
            )

    # Heavy tier: batched reconstruct (groups of experts, a handful of launches per group; see
    # moe_batch_recon.py). Experts above the batched tier's row cap are left to the caller
    # (large slabs pad and stream more than they save in launches)
    heavy = [(j, e, s, c) for j, e, s, c in items if not (fc and c <= fused_t)]
    if recon is None:
        return [(j, e) for j, e, _, _ in heavy]
    single = [(j, e) for j, e, _, c in heavy if c > recon.max_rows]
    heavy = [h for h in heavy if h[3] <= recon.max_rows]
    if heavy:
        y_ext, tok_ext, w_ext = staged_recon_inputs(y, token_sorted, weight_sorted, recon.dims_u[0], ctx)
        at = {e: (j, s, c) for j, e, s, c in heavy}
        for grp in plan_groups([e for _, e, _, _ in heavy], lambda e: at[e][2], recon.cap):
            # Trellis addresses inside the staging buffer, per projection
            bb = [base + at[e][0] * stride for e in grp]
            recon.run_group(
                y_ext, out_ext, tok_ext, w_ext,
                grp, [at[e][1] for e in grp], [at[e][2] for e in grp],
                ptrs = ([b + g_off for b in bb] if gated else None,
                        [b + u_off for b in bb], [b + d_off for b in bb]))
    return single


def staged_recon_inputs(y, token_sorted, weight_sorted, hi, ctx):
    """Padding sources / sinks of the batched tier, built once per call (cached in ctx): the
    input zero-padded to the quantized width with a zero row appended, and the sorted assignment
    lists with a sentinel entry (token = rows, the output's sink row; weight 0)"""
    r = ctx.get("recon_ext")
    if r is None:
        rows = y.shape[0]
        y_in = y if y.shape[1] == hi else torch.nn.functional.pad(y, (0, hi - y.shape[1]))
        r = ctx["recon_ext"] = (
            torch.cat([y_in, torch.zeros((1, hi), dtype = y.dtype, device = y.device)]),
            torch.cat([token_sorted, torch.full((1,), rows, dtype = token_sorted.dtype, device = y.device)]),
            torch.cat([weight_sorted.half(), torch.zeros((1,), dtype = torch.half, device = y.device)]),
        )
    return r


# Page-locking budget and failure tracking for mapped expert homes (hybrid MoE), shared by every
# host in the process: registering 100+ GiB is not guaranteed to succeed (RLIMIT_MEMLOCK, driver
# limits, fragmentation), so each 1 GiB chunk is registered on its own as it appears, failures
# leave that chunk pageable (its experts then take CPU lanes, see moe_hybrid_dispatch), and after
# two consecutive failures no further chunk is attempted
_HOME_PIN = {"registered": 0, "fail_run": 0, "disabled": False, "logged": 0, "notices": set()}
_HOME_PIN_LIMIT = int(float(os.environ.get("EXL3_EXPERT_HOME_PIN_LIMIT_GB", "0") or 0) * (1 << 30))


def _home_notice(key, msg):
    if key not in _HOME_PIN["notices"]:
        _HOME_PIN["notices"].add(key)
        print(msg, flush = True)


def register_mapped_home(ptr: int, size: int, what: str) -> int:
    """Page-lock and map [ptr, ptr + size) for zero-copy GPU reads. Returns the device alias of
    ptr (cudaHostGetDevicePointer; never assumed equal to the host address), or 0 when the
    region stays pageable: budget exhausted, registration refused, or registration disabled
    after repeated failures. Never raises"""
    st = _HOME_PIN
    if st["disabled"]:
        return 0
    if _HOME_PIN_LIMIT and st["registered"] + size > _HOME_PIN_LIMIT:
        _home_notice("limit", f" -- {what}: EXL3_EXPERT_HOME_PIN_LIMIT_GB reached "
                              f"({st['registered'] >> 30} GiB registered); further homes stay pageable "
                              f"(their experts run on CPU lanes)")
        return 0
    try:
        cuda_host_register(ptr, size, flags = CUDA_HOST_REGISTER_PORTABLE | CUDA_HOST_REGISTER_MAPPED)
    except Exception as e:
        st["fail_run"] += 1
        print(f" !! {what}: cudaHostRegister of {size >> 20} MiB failed ({str(e).splitlines()[0][:160]}); "
              f"chunk stays pageable, its experts run on CPU lanes", flush = True)
        if st["fail_run"] >= 2:
            st["disabled"] = True
            print(f" !! {what}: two consecutive registration failures, no further home chunk will be "
                  f"registered (raise RLIMIT_MEMLOCK / set EXL3_EXPERT_HOME_PIN_LIMIT_GB)", flush = True)
        return 0
    try:
        dev = cuda_host_get_device_pointer(ptr)
    except Exception as e:
        cuda_host_unregister(ptr)
        print(f" !! {what}: no device alias for a registered chunk ({e}); chunk stays pageable", flush = True)
        return 0
    st["fail_run"] = 0
    st["registered"] += size
    if st["registered"] - st["logged"] >= (8 << 30):
        st["logged"] = st["registered"]
        print(f" -- expert homes: {st['registered'] >> 30} GiB page-locked and mapped", flush = True)
    return dev


def unregister_mapped_home(ptr: int, size: int):
    try:
        cuda_host_unregister(ptr)
        _HOME_PIN["registered"] = max(0, _HOME_PIN["registered"] - size)
    except Exception:
        pass


class MoeCpuHost:

    def __init__(self, config, numa: dict | None = None):
        """numa (hybrid MoE / EXL3_NUMA_ENABLE, see moe_numa_hosts.py): this host's worker node --
        dict(index, node, cpus, mem_node, threads, home, home_node, map_arena, telemetry,
        handoff_node). None: the classic single worker, unchanged"""
        self.config = config
        self.numa = numa or {}
        self.model_dir = config.directory
        self.specs = []
        self.by_key = {}
        self.live_layers = 0
        self.acked = 0
        self.started = False
        self.shm = None
        self.proc = None
        self.conn = None
        self.v_abort = None
        self.seq = 0
        self.next_slot = 0
        self.slot_last_seq = [0] * MOE_MAX_SLOTS
        self.num_slots = TUNING.num_slots
        self.cap_rows = TUNING.cap_rows
        # Per-component thread override: config.infer_params.moe_cpu_threads for the main model,
        # draft_moe_cpu_threads for anything else (MTP head / draft model); falls back to the
        # tuning default (EXL3_MOE_CPU_THREADS env, else physical cores minus EXL3_MOE_HOST_CORES)
        comp = getattr(config.infer_params, "moe_cpu_component", "text")
        cfg_threads = getattr(config.infer_params,
            "moe_cpu_threads" if comp == "text" else "draft_moe_cpu_threads", None)
        self.threads = self.numa.get("threads") or cfg_threads or TUNING.threads
        self.stage_threads = TUNING.stage_threads
        # GPU-streaming prefill: experts with at least stream_t assigned tokens are streamed to
        # the GPU (weights DMA'd through a pinned staging ring) while the tail stays on the CPU
        self.num_wslots = TUNING.num_wslots
        self.wslot_size = TUNING.wslot_size
        self.stream_t = TUNING.stream_t
        self.stream_min_rows = TUNING.stream_min_rows
        self.batch_experts = TUNING.batch_experts
        self.wseq = 0
        self.next_wslot = 0
        self.wslot_prev_seq = [0] * MOE_MAX_WSLOTS
        self.aux = {}
        self._dev_bufs = {}      # per device: persistent streamed-prefill buffers (see _device_buffers)
        # Pinned arena (EXL3_MOE_PINNED_ARENA): parent-side mappings of the worker's memfd
        # chunks (mmap, int16 view) indexed like the worker's chunk list, and per layer the
        # per-expert (chunk, byte offset) of its contiguous [gate | up | down] trellis block
        # Worker-arena homes (hybrid MoE without a separate native copy) need the arena shared
        # with the parent and registered mapped
        self.map_arena = bool(self.numa.get("map_arena"))
        self.pinned = TUNING.pinned_arena or self.map_arena
        self.arena_maps = []
        self.arena_views = []
        self.arena_dev = []          # per arena chunk: device alias (mapped) or 0
        self.layer_blocks = []
        # Native-layout home copy (hybrid MoE): parent-side mappings, device aliases (0 =
        # pageable) and per layer the per-expert (chunk, offset) of each home block
        self.home = bool(self.numa.get("home"))
        self.home_maps = []
        self.home_views = []
        self.home_dev = []
        self.layer_home_blocks = []
        self.telemetry = bool(self.numa.get("telemetry"))
        self.batch_recon = TUNING.stream_batch_recon

    def _spawn(self):
        if self.proc is not None:
            return
        ctx = multiprocessing.get_context("spawn")
        self.conn, child_conn = ctx.Pipe(duplex = True)
        cpus = self.numa.get("cpus") or _HOST_ORIG_CPUS
        self.proc = ctx.Process(
            target = _moe_cpu_child_main,
            args = (child_conn, self.model_dir, self.threads, self.stage_threads,
                    self.pinned, TUNING.arena_huge if (self.pinned or self.home) else "", cpus,
                    self.numa.get("mem_node"), self.home, self.numa.get("home_node")),
            daemon = True,
        )
        # The MTP head / draft model's pool runs between the main model's passes on the same
        # node: pinned to the same cores, each idle pool's spin-wait steals the active pool's
        # cores. EXL3_MOE_DRAFT_CPU_PIN (default 0) sets EXL3_MOE_CPU_PIN for those workers only;
        # the child reads it from the environment it is spawned with
        comp = getattr(getattr(getattr(self, "config", None), "infer_params", None), "moe_cpu_component", "text")
        pin_env = None
        if comp != "text":
            pin_env = os.environ.get("EXL3_MOE_CPU_PIN")
            os.environ["EXL3_MOE_CPU_PIN"] = os.environ.get("EXL3_MOE_DRAFT_CPU_PIN", "0")
        try:
            self.proc.start()
        finally:
            if comp != "text":
                if pin_env is None:
                    os.environ.pop("EXL3_MOE_CPU_PIN", None)
                else:
                    os.environ["EXL3_MOE_CPU_PIN"] = pin_env
        child_conn.close()
        # Cleanupper fires at the end of the __main__ scope, before interpreter teardown breaks
        # the shm views and pipe machinery that shutdown() needs; PDEATHSIG in the child covers
        # the paths where no Python hook runs at all
        cleanupper.register_atexit(self.shutdown)

    def _pump(self, timeout):
        """Receive one message from the child, surfacing errors and death"""
        if self.conn.poll(timeout):
            msg = self.conn.recv()
            if msg[0] == "err":
                raise RuntimeError(f"CPU MoE worker failed:\n{msg[1]}")
            if msg[0] == "ok":
                # Layer acks arrive in registration order: entry i describes specs[i]
                self.layer_blocks.append(msg[1] if len(msg) > 1 else None)
                self.layer_home_blocks.append(msg[2] if len(msg) > 2 else None)
                self.acked += 1
            elif msg[0] == "chunk":
                self._attach_chunk(msg[1], msg[2], msg[3])
            elif msg[0] == "home_chunk":
                self._attach_home_chunk(msg[1], msg[2], msg[3])
            return True
        if not self.proc.is_alive():
            raise RuntimeError("CPU MoE worker process died")
        return False

    def _attach_chunk(self, index, size, name):
        """Pinned arena: map arena chunk `index` (Linux: descriptor sent right after the
        ("chunk", ...) message; Windows: section `name`) and page-lock it for DMA. Registration
        is done here, per chunk as it appears during loading, so its cost overlaps the rest of
        the load instead of stacking up at startup. Any failure closes the mapping before the
        error leaves this frame."""
        import mmap
        import socket
        assert index == len(self.arena_maps), "arena chunk published out of order"
        if name is not None:
            # Opening by name fails loudly if the worker already dropped the section
            m = shared_memory.SharedMemory(name = name)
        else:
            with socket.socket(fileno = os.dup(self.conn.fileno())) as sock:
                _, fds, _, _ = socket.recv_fds(sock, 1, 1)
            assert len(fds) == 1, "arena chunk descriptor missing"
            fd = fds[0]
            try:
                m = mmap.mmap(fd, size, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)
            finally:
                os.close(fd)
        view = None
        if self.map_arena:
            # Worker-arena homes (hybrid MoE): mapped registration with the non-fatal fallback of
            # register_mapped_home (an unregistered chunk's experts take CPU lanes)
            view = torch.frombuffer(m.buf if name is not None else m, dtype = torch.int16, count = size // 2)
            dev = register_mapped_home(view.data_ptr(), size, f"CPU MoE arena chunk {index}")
            self.arena_maps.append(m)
            self.arena_views.append(view)
            self.arena_dev.append(dev)
            return
        try:
            # count = size // 2: a section opened by name reports its page-rounded size, and
            # frombuffer rejects a buffer shorter than the advertised chunk
            view = torch.frombuffer(m.buf if name is not None else m, dtype = torch.int16,
                                    count = size // 2)
            cuda_host_register(view.data_ptr(), size, flags = CUDA_HOST_REGISTER_PORTABLE)
        except Exception as e:
            view = None   # no tensor over an unmapped region may survive in the traceback's frame
            m.close()
            raise RuntimeError(
                f"CPU MoE pinned arena: chunk {index} ({size >> 20} MiB) could not be attached "
                f"({e}). Unset EXL3_MOE_PINNED_ARENA to use the staged path.") from e
        self.arena_maps.append(m)
        self.arena_views.append(view)
        self.arena_dev.append(0)
        if os.environ.get("EXL3_MOE_ARENA_DEBUG"):
            print(f" -- pinned arena: mapped + registered chunk {index} ({size >> 20} MiB)",
                  flush = True)

    def _attach_home_chunk(self, index, size, name):
        """Native-layout home chunk (hybrid MoE): map it and try the mapped registration. A chunk
        whose registration fails stays mapped and readable by the host (warm installs copy from
        it synchronously); its device alias is 0 so the dispatcher never gives its experts a
        GPU lane that reads it"""
        import mmap
        import socket
        assert index == len(self.home_maps), "home chunk published out of order"
        if name is not None:
            m = shared_memory.SharedMemory(name = name)
        else:
            with socket.socket(fileno = os.dup(self.conn.fileno())) as sock:
                _, fds, _, _ = socket.recv_fds(sock, 1, 1)
            assert len(fds) == 1, "home chunk descriptor missing"
            try:
                m = mmap.mmap(fds[0], size, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)
            finally:
                os.close(fds[0])
        view = torch.frombuffer(m.buf if name is not None else m, dtype = torch.uint8, count = size)
        dev = register_mapped_home(view.data_ptr(), size, f"CPU MoE home chunk {index}")
        self.home_maps.append(m)
        self.home_views.append(view)
        self.home_dev.append(dev)

    # ---- hybrid MoE accessors --------------------------------------------------------------

    def _home_location(self, layer_idx, local_idx):
        """(views, devs, chunk, offset) of an expert's native-layout block: the home copy, or
        the worker arena block when the arena itself is the home (map_arena)"""
        if self.home:
            blocks = self.layer_home_blocks[layer_idx]
            ci, off = blocks[local_idx]
            return self.home_views, self.home_dev, ci, off
        if self.map_arena:
            ci, off = self.layer_blocks[layer_idx][local_idx]
            return self.arena_views, self.arena_dev, ci, off
        return None

    def home_device_ptr(self, layer_idx, local_idx) -> int:
        """Device alias of an expert's native-layout block, 0 when not GPU-readable"""
        loc = self._home_location(layer_idx, local_idx)
        if loc is None:
            return 0
        views, devs, ci, off = loc
        return devs[ci] + off if devs[ci] else 0

    def home_host_view(self, layer_idx, local_idx, nbytes) -> torch.Tensor | None:
        """CPU uint8 view of an expert's native-layout block (host-driven installs)"""
        loc = self._home_location(layer_idx, local_idx)
        if loc is None:
            return None
        views, devs, ci, off = loc
        v = views[ci]
        if v.dtype != torch.uint8:
            v = v.view(torch.uint8)
        return v[off: off + nbytes]

    def register_layer(self, key, gate_keys, up_keys, down_keys, activation, act_limit, hi, ho, topk,
                       proj_dims = None, aux = None, interm_fp32 = False, home_mask = None):
        if key in self.by_key:
            # Autosplit rollback retry: the child keeps its copy, reuse the index, but take
            # the re-fetched aux tensors: the retry runs on a different device, and the stored
            # copies live on the one the layer just rolled back from. Streamed-prefill dequant
            # builds raw pointer tables from these, so stale entries are device-A addresses
            # handed to kernels on device B (illegal memory access on the first  multi-chunk
            # prefill after a cross-device rollback)
            idx = self.by_key[key]
            self.live_layers += 1
            if aux is not None:
                self.aux[idx] = aux
            return idx
        assert not self.started, "cannot register layers after the worker has started"
        self._spawn()
        spec = dict(
            gate_keys = gate_keys, up_keys = up_keys, down_keys = down_keys,
            activation = activation, act_limit = act_limit,
            hi = hi, ho = ho, topk = topk,
            num_experts = len(up_keys),
            proj_dims = proj_dims,
            interm_fp32 = interm_fp32,      # resident experts' gate/up output dtype (BlockSparseMLP interm_dtype)
            home_mask = home_mask,          # per expert: keep a home copy (None = all)
        )
        if proj_dims is not None:
            # Deterministic per-expert byte layout (gate, up, down), mirrored by the worker's
            # stage function
            def tb(d):
                k, n, K = d
                return (k // 16) * (n // 16) * int(16 * K) * 2
            gb = tb(proj_dims["g"]) if proj_dims.get("g") else 0
            ub, db = tb(proj_dims["u"]), tb(proj_dims["d"])
            spec["proj_bytes"] = (gb, ub, db)
            spec["expert_bytes"] = gb + ub + db
        self.specs.append(spec)
        self.live_layers += 1
        idx = len(self.specs) - 1
        self.by_key[key] = idx
        if aux is not None:
            self.aux[idx] = aux
        self.conn.send(("layer", {k: v for k, v in spec.items() if k != "proj_dims"}))
        return idx

    def commit_module(self, module_key):
        """
        Called by the loader after a top-level module has committed on the GPU side: block until
        the child has loaded every layer registered under that module, keeping load progress
        honest and the child at most one layer behind.
        """
        if self.proc is None:
            return
        idxs = [i for k, i in self.by_key.items()
                if k == module_key or k.startswith(module_key + ".")]
        if not idxs:
            return
        need = max(idxs) + 1
        while self.acked < need:
            self._pump(1.0)

    def ensure_started(self):
        if self.started or not self.specs:
            return
        while self.acked < len(self.specs):
            self._pump(1.0)

        max_hi = max(s["hi"] for s in self.specs)
        max_ho = max(s["ho"] for s in self.specs)
        max_topk = max(s["topk"] for s in self.specs)
        # All staged sections must be contiguous 2D blocks: torch implements strided GPU<->CPU
        # copies with a host-side staging pass at enqueue time, which breaks stream ordering (the
        # readback would ship the slot's *previous* contents). Uniform dims keep every copy a
        # full-width contiguous slice
        assert all(s["hi"] == max_hi and s["ho"] == max_ho and s["topk"] == max_topk
                   for s in self.specs), "CPU MoE offload requires uniform expert dims and top-k"

        off_x = 0
        off_sel = _align64(off_x + self.cap_rows * max_hi * 2)
        off_w = _align64(off_sel + self.cap_rows * max_topk * 4)
        off_out = _align64(off_w + self.cap_rows * max_topk * 2)
        slot_size = _align64(off_out + self.cap_rows * max_ho * 4)
        # Optional per-job telemetry record (MoeJobTelemetry, 64 bytes) at the end of each slot
        off_tel = slot_size if self.telemetry else -1
        if self.telemetry:
            slot_size += 64
        self.layout = dict(
            num_slots = self.num_slots, slot_size = slot_size, cap_rows = self.cap_rows,
            max_hi = max_hi, max_ho = max_ho, max_topk = max_topk, telemetry_offset = off_tel,
        )

        self.layout["wstage_off"] = MOE_CTRL_SIZE + self.num_slots * slot_size
        self.layout["num_wslots"] = self.num_wslots
        self.layout["wslot_size"] = self.wslot_size
        self.layout["cpu_prof"] = TUNING.cpu_prof
        size = MOE_CTRL_SIZE + self.num_slots * slot_size + self.num_wslots * self.wslot_size
        check_shm_capacity(size, "The CPU MoE offload handoff segment")
        self.shm = shared_memory.SharedMemory(create = True, size = size)
        buf = np.frombuffer(self.shm.buf, dtype = np.uint8)
        buf[:MOE_CTRL_SIZE] = 0
        self.base_ptr = buf.ctypes.data
        # NUMA: the GPU writes slot inputs and flags over PCIe and the worker reads them; the
        # segment goes to the node given for it (the GPU's node when known), faulted in by the
        # registration below under that policy. None: default placement
        hnode = self.numa.get("handoff_node")
        with numa_mem.policy_scope(hnode):
            if hnode is not None:
                buf[MOE_CTRL_SIZE:] = 0
            cuda_host_register(self.base_ptr, size,
                               flags = CUDA_HOST_REGISTER_PORTABLE | CUDA_HOST_REGISTER_MAPPED)
        # GPU-visible alias of the registered buffer: equal to base_ptr under UVA (Linux
        # desktop), but a distinct address under WDDM (Windows), where the host VA is not a
        # valid device pointer
        self.gpu_base_ptr = cuda_host_get_device_pointer(self.base_ptr)

        u32 = np.frombuffer(self.shm.buf, dtype = np.uint32)
        self.v_quit = u32[0:1]
        self.v_pass_wake = u32[16:17]
        self.v_abort = u32[32:33]
        self.v_ready = u32[48:49]
        self.v_thread_caps = u32[49:51]     # runtime caps: multi-row jobs, single-row jobs
        self.v_thread_caps[:] = getattr(self, "_thread_caps", (0, 0))
        self.v_jobs_tail = u32[64:65]
        self.v_jobs_head = u32[80:81]
        self.v_jobs = np.frombuffer(
            self.shm.buf, dtype = np.uint32,
            offset = MOE_CTRL_JOBS_OFFSET, count = MOE_JOB_RING * (MOE_JOB_BYTES // 4)).reshape(MOE_JOB_RING, MOE_JOB_BYTES // 4)

        self.slots = []
        for s in range(self.num_slots):
            sbase = MOE_CTRL_SIZE + s * slot_size
            def view(off, count, dtype):
                return torch.frombuffer(self.shm.buf, dtype = dtype, count = count,
                                        offset = sbase + off)
            # Device-visible aliases of the slot's data sections, for the fused issue/collect
            # kernels' zero-copy accesses (same mapped registration as the flag words)
            gpu_data = self.gpu_base_ptr + sbase
            self.slots.append(dict(
                x = view(off_x, self.cap_rows * max_hi, torch.half).view(self.cap_rows, max_hi),
                sel = view(off_sel, self.cap_rows * max_topk, torch.int32).view(self.cap_rows, max_topk),
                w = view(off_w, self.cap_rows * max_topk, torch.half).view(self.cap_rows, max_topk),
                out = view(off_out, self.cap_rows * max_ho, torch.float).view(self.cap_rows, max_ho),
                x_dev = gpu_data + off_x,
                sel_dev = gpu_data + off_sel,
                w_dev = gpu_data + off_w,
                out_dev = gpu_data + off_out,
                x_cap = self.cap_rows * max_hi,         # halves (hybrid issue capacity check)
                sel_cap = self.cap_rows * max_topk,     # selection / weight entries
                data_ready = self.gpu_base_ptr + MOE_SLOT_FLAGS_OFFSET + s * 64,
                done = self.gpu_base_ptr + MOE_SLOT_FLAGS_OFFSET + 64 * MOE_MAX_SLOTS + s * 64,
                consumed = self.gpu_base_ptr + MOE_SLOT_FLAGS_OFFSET + 2 * 64 * MOE_MAX_SLOTS + s * 64,
                tel = np.frombuffer(self.shm.buf, dtype = np.uint32, count = 16, offset = sbase + off_tel)
                    if off_tel >= 0 else None,
            ))
        # Per-device empty-job gates for the fused kernels (issue writes, collect reads)
        self.dev_count = {}

        # Weight-staging views, flags and streams for GPU-streamed prefill
        self.wviews = []
        for s in range(self.num_wslots):
            self.wviews.append(torch.frombuffer(
                self.shm.buf, dtype = torch.int16, count = self.wslot_size // 2,
                offset = self.layout["wstage_off"] + s * self.wslot_size))
        # Wslot flag banks sit after data_ready/done/consumed (keep in sync with
        # moe_handoff.h's layout comment and the watchdog unblock above)
        fl = self.gpu_base_ptr + MOE_SLOT_FLAGS_OFFSET + 3 * 64 * MOE_MAX_SLOTS
        self.stage_done_addr = [fl + s * 64 for s in range(MOE_MAX_WSLOTS)]
        self.pinned_free_addr = [fl + 64 * MOE_MAX_WSLOTS + s * 64 for s in range(MOE_MAX_WSLOTS)]
        self.v_stage_tail = u32[MOE_STAGE_TAIL_OFFSET // 4 : MOE_STAGE_TAIL_OFFSET // 4 + 1]
        self.v_stage_head = u32[MOE_STAGE_HEAD_OFFSET // 4 : MOE_STAGE_HEAD_OFFSET // 4 + 1]
        self.v_stage_jobs = np.frombuffer(
            self.shm.buf, dtype = np.uint32,
            offset = MOE_STAGE_JOBS_OFFSET, count = MOE_STAGE_RING * (MOE_JOB_BYTES // 4)).reshape(MOE_STAGE_RING, MOE_JOB_BYTES // 4)
        # Per-device CUDA state for the streamed-prefill path (copy stream, VRAM ring, events)
        self.sstate = {}

        self.conn.send(("start", self.shm.name, self.layout))

        import time
        t0 = time.time()
        # Startup is now just the shared-memory attach, layer registration and thread spawn
        # (hugepage promotion runs in the worker's background); the limit stays generous for
        # slow hosts and is overridable
        timeout = float(os.environ.get("EXL3_MOE_CPU_START_TIMEOUT", "60"))
        while not self.v_ready[0]:
            if not self.proc.is_alive():
                raise RuntimeError("CPU MoE worker process died during startup")
            if time.time() - t0 > timeout:
                raise RuntimeError(
                    f"CPU MoE worker startup timeout ({timeout:.0f} s; EXL3_MOE_CPU_START_TIMEOUT overrides)")
            time.sleep(0.005)
        self.started = True
        self._flags_u32 = u32
        self._start_watchdog()
        kern = "avx512-vbmi" if ext.exl3_moe_cpu_has_avx512_vbmi() else \
               ("avx512-vnni" if ext.exl3_moe_cpu_has_avx512_vnni() else \
               ("avx512-bw" if ext.exl3_moe_cpu_has_avx512_bw() else \
               ("avx2" if ext.exl3_moe_cpu_has_avx2() else "scalar")))
        if ext.exl3_moe_cpu_has_amx():
            kern += " + amx"
        where = f", node {self.numa['node']}" if self.numa.get("node") is not None else ""
        print(f" -- CPU MoE worker started: {len(self.specs)} layers, {kern}, {self.threads} threads{where}")
        # The worker was spawned before this and pins its own threads; keep the host off them.
        # Per-node hosts (moe_numa_hosts) plan the host placement for the whole group instead
        if not self.numa or self.numa.get("node") is None:
            # Classic single worker, or a group whose plan has one unconfined worker (single
            # node, NUMA off): the classic host placement applies unchanged
            _apply_host_affinity(self.threads, TUNING.host_cores)

    def _start_watchdog(self):
        """
        GPU-side waits go through stream memops (no timeout, unlike the fallback kernel's 30s
        abort), so a dead worker would otherwise hang the stream forever. Detect it host-side
        and unblock every pending GEQ wait by writing satisfying values into the flags, then set
        the abort flag so the next begin_pass raises.
        """
        import threading, time

        def wd():
            while True:
                if not self.started or self.proc is None:
                    return
                if not self.proc.is_alive():
                    try:
                        # Abort first, then release the waits: any forward that gets past a
                        # released wait (stale CPU partials) is guaranteed to see the abort flag
                        # once the host has synchronized with it (check_workers)
                        self.v_abort[0] = 1
                        u32 = self._flags_u32
                        seq, wseq = self.seq + 1, self.wseq + 1
                        done0 = MOE_SLOT_FLAGS_OFFSET + 64 * MOE_MAX_SLOTS
                        cons0 = MOE_SLOT_FLAGS_OFFSET + 2 * 64 * MOE_MAX_SLOTS
                        for s in range(MOE_MAX_SLOTS):
                            _write_flag(u32, done0 + s * 64, seq)
                            _write_flag(u32, cons0 + s * 64, seq)
                        fl = MOE_SLOT_FLAGS_OFFSET + 3 * 64 * MOE_MAX_SLOTS
                        for s in range(MOE_MAX_WSLOTS):
                            _write_flag(u32, fl + s * 64, wseq)                         # stage_done
                            _write_flag(u32, fl + 64 * MOE_MAX_WSLOTS + s * 64, wseq)   # pinned_free
                        self.v_abort[0] = 1
                    except Exception:
                        pass
                    return
                time.sleep(0.5)

        threading.Thread(target = wd, daemon = True).start()

    def aborted(self) -> bool:
        return bool(self.started and self.v_abort is not None and self.v_abort[0])

    def set_thread_caps(self, multi_row: int = 0, single_row: int = 0) -> bool:
        """Cap the worker's pool participants per job from the next job on, without a restart:
        multi_row for prefill-shaped jobs, single_row for one-token (decode) jobs; 0 lifts a cap.
        Caps never exceed the pool's thread count. Before the worker starts they are kept and
        applied at start; returns whether they took effect now"""
        self._thread_caps = (max(0, int(multi_row)), max(0, int(single_row)))
        v = getattr(self, "v_thread_caps", None)
        if v is None:
            return False
        v[:] = self._thread_caps
        return True

    def begin_pass(self):
        if not self.started:
            self.ensure_started()
        if not self.started:
            return
        if self.v_abort[0]:
            raise RuntimeError("CPU MoE worker timed out (abort flag set)")
        self.v_pass_wake[0] += 1

    def submit(self, layer_idx, y, selected_experts, routing_weights):
        """
        Enqueue the routed-expert computation for one offloaded layer onto the current CUDA
        stream; returns the (asynchronously filled) float output tensor. Never synchronizes the
        host with the stream.
        """
        # Device guard: the flag kernels launch on the *current* device's current stream, which
        # need not match the layer's device (e.g. a model loaded entirely on cuda:1)
        with torch.cuda.device(y.device):
            spec = self.specs[layer_idx]
            h = y.shape[1]
            out = torch.empty((y.shape[0], h), dtype = torch.float, device = y.device)
            if os.environ.get("EXL3_MOE_SUBMIT_PROF"):
                if not hasattr(self, "_prof_ev"):
                    self._prof_ev = []
                ev0 = torch.cuda.Event(enable_timing = True)
                ev1 = torch.cuda.Event(enable_timing = True)
                ev0.record()
                jobs, rtmp = self._issue_compute(layer_idx, y, selected_experts, routing_weights, spec, out, h)
                self._collect_compute(jobs, out, rtmp, h)
                ev1.record()
                self._prof_ev.append((ev0, ev1))
                if len(self._prof_ev) >= 64:
                    done_ms = [a.elapsed_time(b) for a, b in self._prof_ev[:-1] if b.query()]
                    if done_ms:
                        done_ms.sort()
                        print(f" -- submit prof ({len(done_ms)} brackets, stream ms): "
                              f"med {done_ms[len(done_ms) // 2]:.3f} "
                              f"p90 {done_ms[int(len(done_ms) * 0.9)]:.3f} "
                              f"max {done_ms[-1]:.3f}", flush = True)
                    self._prof_ev = self._prof_ev[-1:]
            else:
                jobs, rtmp = self._issue_compute(layer_idx, y, selected_experts, routing_weights, spec, out, h)
                self._collect_compute(jobs, out, rtmp, h)
        return out

    def submit_issue(self, layer_idx, y, selected_experts, routing_weights):
        """
        Two-phase submit for the per-layer expert split: stage this layer's inputs and publish
        the compute job(s) nwo, defer the stream-side waits and readbacks to submit_collect so
        the caller can enqueue its own GPU expert work in between. That work then executes
        concurrently with the worker instead of behind the flag wait.
        """
        with torch.cuda.device(y.device):
            spec = self.specs[layer_idx]
            h = y.shape[1]
            out = torch.empty((y.shape[0], h), dtype = torch.float, device = y.device)
            jobs, rtmp = self._issue_compute(layer_idx, y, selected_experts, routing_weights, spec, out, h)
        return (jobs, rtmp, out, h, y.device)

    def submit_collect(self, handle):
        """Enqueue the deferred flag waits and readbacks; returns the (asynchronously
        filled) output tensor. Never synchronizes the host."""
        jobs, rtmp, out, h, dev = handle
        with torch.cuda.device(dev):
            self._collect_compute(jobs, out, rtmp, h)
        return out

    def submit_issue_fused(self, layer_idx, y, selected_experts, routing_weights,
                           split_map, split_hist, first_cpu):
        """
        Kernel-fused variant of submit_issue for decode-size jobs: ONE kernel launch stages
        sel/x/w straight into the pinned slot with zero-copy stores (replacing the int32
        cast, the zero-pad and three cudaMemcpyAsync launches), doubling as moe_split_map in
        dynamic-placement mode (in-place sel translate + hit histogram). When no selected
        expert is CPU-resident, the activation/weight payload is skipped entirely, the
        worker skips the compute, and the fused collect skips the readback: an inactive
        layer costs two flag memops and two near-empty kernels, and no PCIe payload.

        Returns None if the job shape does not fit the single-slot fast path (caller falls
        back to the copy path). selected_experts must hold RAW router ids here; translation
        (dynamic map or static tail offset) happens inside the kernel.
        """
        rows = y.shape[0]
        spec = self.specs[layer_idx]
        if rows > self.cap_rows or not (y.is_contiguous() and routing_weights.is_contiguous()):
            return None
        h_ = y.shape[1]
        hi = spec["hi"]
        dev = y.device
        with torch.cuda.device(dev):
            counts = self.dev_count.get(dev)
            if counts is None:
                counts = self.dev_count[dev] = \
                    torch.zeros((self.num_slots,), dtype = torch.int32, device = dev)
            slot_idx = self.next_slot
            self.next_slot = (self.next_slot + 1) % self.num_slots
            self.seq += 1
            seq = self.seq
            slot = self.slots[slot_idx]

            tail = int(self.v_jobs_tail[0])
            if tail - int(self.v_jobs_head[0]) >= MOE_JOB_RING - 4:
                import time
                while tail - int(self.v_jobs_head[0]) >= MOE_JOB_RING - 4:
                    if self.v_abort[0] or not self.proc.is_alive():
                        raise RuntimeError("CPU MoE worker failed (ring stall)")
                    time.sleep(0.0002)
            job = self.v_jobs[tail % MOE_JOB_RING]
            _set_job_seq(job, seq)
            job[1] = layer_idx
            job[2] = rows
            job[3] = spec["topk"]
            job[4] = slot_idx
            job[5] = 2    # MOE_JOB_KIND_COMPUTE_GATED
            self.v_jobs_tail[0] = tail + 1

            if self.slot_last_seq[slot_idx]:
                ext.exl3_moe_flag_wait(slot["consumed"], self.slot_last_seq[slot_idx],
                                       self.gpu_base_ptr + 128)
            ext.moe_split_issue(
                selected_experts.view(-1), split_map, split_hist,
                y, routing_weights,
                slot["sel_dev"], slot["x_dev"], slot["w_dev"],
                counts, slot_idx, hi, first_cpu,
            )
            ext.exl3_moe_flag_write(slot["data_ready"], seq)
            self.slot_last_seq[slot_idx] = seq
        return (seq, slot_idx, rows, h_, spec["ho"], dev, counts)

    def fused_reserve(self, layer_idx, rows, device):
        """Hybrid dispatcher, decode-size job: claim the next slot, push a COMPUTE_GATED job
        descriptor and order the slot's reuse after its previous tenant's readback (consumed),
        on the current stream. The caller's issue kernel then fills sel/x/w and the slot gate,
        fused_publish releases the job, fused_collect folds the result in. Same protocol as
        submit_issue_fused, split so one kernel can serve several hosts (NUMA nodes)"""
        spec = self.specs[layer_idx]
        if rows > self.cap_rows:
            raise ValueError(f"CPU MoE job of {rows} rows exceeds the slot capacity ({self.cap_rows})")
        if self.v_abort[0]:
            raise RuntimeError("CPU MoE worker died or timed out (abort flag set)")
        with torch.cuda.device(device):
            counts = self.dev_count.get(device)
            if counts is None:
                counts = self.dev_count[device] = \
                    torch.zeros((self.num_slots,), dtype = torch.int32, device = device)
            slot_idx = self.next_slot
            self.next_slot = (self.next_slot + 1) % self.num_slots
            self.seq += 1
            seq = self.seq
            slot = self.slots[slot_idx]
            tail = int(self.v_jobs_tail[0])
            if tail - int(self.v_jobs_head[0]) >= MOE_JOB_RING - 4:
                import time
                while tail - int(self.v_jobs_head[0]) >= MOE_JOB_RING - 4:
                    if self.v_abort[0] or not self.proc.is_alive():
                        raise RuntimeError("CPU MoE worker failed (ring stall)")
                    time.sleep(0.0002)
            job = self.v_jobs[tail % MOE_JOB_RING]
            _set_job_seq(job, seq)
            job[1] = layer_idx
            job[2] = rows
            job[3] = spec["topk"]
            job[4] = slot_idx
            job[5] = 2    # MOE_JOB_KIND_COMPUTE_GATED
            self.v_jobs_tail[0] = tail + 1
            if self.slot_last_seq[slot_idx]:
                ext.exl3_moe_flag_wait(slot["consumed"], self.slot_last_seq[slot_idx], self.gpu_base_ptr + 128)
            self.slot_last_seq[slot_idx] = seq
        return dict(seq = seq, slot_idx = slot_idx, slot = slot, counts = counts, ho = spec["ho"],
                    dev = device, rows = rows)

    def fused_publish(self, r):
        with torch.cuda.device(r["dev"]):
            ext.exl3_moe_flag_write(r["slot"]["data_ready"], r["seq"])

    def fused_collect(self, r, final_2d):
        """Wait for the worker, add its partial into final_2d (rows, h) from the pinned slot (a
        no-op read for a job whose gate the issue kernel cleared), release the slot"""
        slot = r["slot"]
        with torch.cuda.device(r["dev"]):
            ext.exl3_moe_flag_wait(slot["done"], r["seq"], self.gpu_base_ptr + 128)
            ext.moe_split_collect_add(final_2d, slot["out_dev"], r["counts"], r["slot_idx"], r["ho"])
            ext.exl3_moe_flag_write(slot["consumed"], r["seq"])

    def read_telemetry(self) -> list[tuple[int, int, int, int, int]]:
        """Latest (seq, compute_us, assignments, experts, rows) per slot (telemetry on), as
        published with each job's done flag. Host read of pinned memory, no stream interaction"""
        out = []
        if not self.started or not self.telemetry:
            return out
        for slot in self.slots:
            t = slot.get("tel")
            if t is not None:
                out.append((int(t[0]), int(t[1]), int(t[2]), int(t[4]), int(t[3])))
        return out

    def submit_collect_fused(self, handle, final_2d):
        """Fold the worker's partial into final_2d (rows, h) in place, straight from the
        pinned slot, or, for a job the issue kernel recorded as empty, do nothing (no
        PCIe reads). Never synchronizes the host."""
        seq, slot_idx, rows, h_, ho, dev, counts = handle
        slot = self.slots[slot_idx]
        with torch.cuda.device(dev):
            ext.exl3_moe_flag_wait(slot["done"], seq, self.gpu_base_ptr + 128)
            ext.moe_split_collect_add(final_2d, slot["out_dev"], counts, slot_idx, ho)
            ext.exl3_moe_flag_write(slot["consumed"], seq)

    def _issue_compute(self, layer_idx, y, selected_experts, routing_weights, spec, out, h):
        """
        Stage inputs and publish one compute job per cap_rows chunk (descriptors, D2H copies
        and data_ready flags only). Waits and readbacks are deferred to _collect_compute so
        other GPU work can be enqueued in between. BUT! Only up to num_slots jobs deep: a
        deeper batch inline-collects the (i - num_slots)-th job before reusing its slot.
        Without the window, slot reuse inside one batch waits on a consumed flag whose write
        (in the deferred collect) sits BEHIND the wait on the same stream (an in-stream
        deadlock (second-prompt freeze)) and even under the old done-flag gating the
        worker's next compute could overwrite a slot output the deferred readback had not
        fetched yet. Caller holds device guard.
        """
        rows = y.shape[0]
        h_ = y.shape[1]
        hi = spec["hi"]
        ho = spec["ho"]
        sel32 = selected_experts.to(torch.int32)
        # Zero-pad up to the quantized input width on the GPU so every D2H below is a contiguous
        # full-width block (see the assert in ensure_started for why this matters)
        y_pad = torch.nn.functional.pad(y, (0, hi - h_)) if hi != h_ else y
        rtmp = torch.empty((min(self.cap_rows, rows), ho), dtype = torch.float, device = y.device) \
            if ho != h_ else None
        jobs = []
        for a in range(0, rows, self.cap_rows):
            if len(jobs) >= self.num_slots:
                self._collect_one(jobs.pop(0), out, rtmp, h)
            b = min(a + self.cap_rows, rows)
            n = b - a
            slot_idx = self.next_slot
            self.next_slot = (self.next_slot + 1) % self.num_slots
            self.seq += 1
            seq = self.seq
            slot = self.slots[slot_idx]

            # Descriptor first (host-visible before the GPU can publish the data flag). Throttle
            # against ring overflow: a long prefill enqueues every job of the pass with no host
            # sync, and overwriting unconsumed descriptors corrupts the whole stream
            tail = int(self.v_jobs_tail[0])
            if tail - int(self.v_jobs_head[0]) >= MOE_JOB_RING - 4:
                import time
                while tail - int(self.v_jobs_head[0]) >= MOE_JOB_RING - 4:
                    if self.v_abort[0] or not self.proc.is_alive():
                        raise RuntimeError("CPU MoE worker failed (ring stall)")
                    time.sleep(0.0002)
            job = self.v_jobs[tail % MOE_JOB_RING]
            _set_job_seq(job, seq)
            job[1] = layer_idx
            job[2] = n
            job[3] = spec["topk"]
            job[4] = slot_idx
            job[5] = 0    # MOE_JOB_KIND_COMPUTE
            self.v_jobs_tail[0] = tail + 1

            # Serialize on the previous tenant's output having been read back (consumed flag,
            # written by the collecting stream after its D2H), not merely computed (done flag):
            # with offloaded layers spread over multiple devices, the previous tenant's collect
            # may sit queued on a different stream than this issue
            if self.slot_last_seq[slot_idx]:
                ext.exl3_moe_flag_wait(slot["consumed"], self.slot_last_seq[slot_idx], self.gpu_base_ptr + 128)
            slot["x"][:n].copy_(y_pad[a:b], non_blocking = True)
            slot["sel"][:n].copy_(sel32[a:b], non_blocking = True)
            slot["w"][:n].copy_(routing_weights[a:b], non_blocking = True)
            ext.exl3_moe_flag_write(slot["data_ready"], seq)
            self.slot_last_seq[slot_idx] = seq
            jobs.append((seq, slot_idx, a, b, n))
        return jobs, rtmp

    def _collect_one(self, job, out, rtmp, h):
        seq, slot_idx, a, b, n = job
        slot = self.slots[slot_idx]
        ext.exl3_moe_flag_wait(slot["done"], seq, self.gpu_base_ptr + 128)
        if rtmp is None:
            out[a:b].copy_(slot["out"][:n], non_blocking = True)
        else:
            rtmp[:n].copy_(slot["out"][:n], non_blocking = True)
            out[a:b] = rtmp[:n, :h]
        # Publish consumption AFTER the readback on this same stream: the slot's next tenant
        # (possibly issuing from another device) gates on this
        ext.exl3_moe_flag_write(slot["consumed"], seq)

    def _collect_compute(self, jobs, out, rtmp, h):
        """Wait for each still-pending job and read its output back; the padded width is
        trimmed on the GPU. Jobs beyond the slot count were already collected inline by the
        issue loop's sliding window. Caller holds the device guard."""
        for job in jobs:
            self._collect_one(job, out, rtmp, h)

    def _max_proj_numel(self):
        mx = 0
        for s in self.specs:
            pd = s.get("proj_dims")
            if pd:
                for k in ("g", "u", "d"):
                    if pd.get(k):
                        mx = max(mx, pd[k][0] * pd[k][1])
        return mx

    def _stream_layout(self, spec):
        """Packed trellis layout of a layer's streamed projections as ((group, planar) for g, u,
        d), plus whether any of them needs the GPU restore pass. The arena holds the bytes in
        the layout the CPU kernel tier wants per bitrate (exl3_moe_cpu_swizzle_group /
        exl3_moe_cpu_planar_layout, the same rule the child loader and expert replacement
        apply), and the GPU kernels only read native order, so every packed projection is
        restored after its DMA. Rates are fixed per layer, so the policy is cached on the spec;
        a layer with nothing packed reads vram_slots directly (no ring, no copy pass)"""
        lay = spec.get("stream_layout")
        if lay is None:
            pd = spec["proj_dims"]
            lay = []
            for nm in ("g", "u", "d"):
                if pd.get(nm) and TUNING.swizzle:
                    K = pd[nm][2]
                    lay.append((int(ext.exl3_moe_cpu_swizzle_group(K)),
                                int(ext.exl3_moe_cpu_planar_layout(K))))
                else:
                    lay.append((0, 0))
            lay = spec["stream_layout"] = (tuple(lay), any(g or p for g, p in lay))
        return lay

    def _device_buffers(self, device, spec = None):
        """The persistent device-side buffers of the streamed-prefill path (VRAM weight ring,
        native-order ring for restored experts, reconstruct scratch, fused-tier buffers, batched
        tier statics), created on first request per device. The autosplit loader requests them
        during a layer's measured load (prefill_worst_case_parts), before the worker starts, so
        they count as allocated when the split is planned instead of appearing on the first real
        prefill; _ensure_stream_state adopts them. The native-order ring exists only once a
        layer whose experts arrive packed asks for it (spec given), so hosts whose tier packs
        nothing never allocate it; once allocated it stays for later layers to account for"""
        key = torch.device(device).index or 0
        d = self._dev_bufs.get(key)
        mx = self._max_proj_numel()
        if d is None:
            d = dict(
                vram_slots = [torch.empty(self.wslot_size // 2, dtype = torch.int16, device = device)
                              for _ in range(self.num_wslots)],
                native_slots = None,
                w_scratch = None,
                fused_bufs = {},
                recon = {},
            )
            self._dev_bufs[key] = d
        if spec is not None and d["native_slots"] is None and self._stream_layout(spec)[1]:
            d["native_slots"] = [torch.empty(self.wslot_size // 2, dtype = torch.int16, device = device)
                                 for _ in range(self.num_wslots)]
        # Reconstruct scratch sized for the largest projection registered so far; grows if a
        # later layer is larger
        if mx and (d["w_scratch"] is None or d["w_scratch"].numel() < mx):
            d["w_scratch"] = torch.empty(mx, dtype = torch.half, device = device)
        return d

    def _ensure_stream_state(self, device):
        key = torch.device(device).index or 0
        st = self.sstate.get(key)
        if st is not None:
            return st
        bufs = self._device_buffers(device)
        st = dict(
            copy_stream = torch.cuda.Stream(device = device),
            vram_slots = bufs["vram_slots"],
            native_slots = bufs["native_slots"],
            wready_ev = [torch.cuda.Event() for _ in range(self.num_wslots)],
            wconsumed_ev = [torch.cuda.Event() for _ in range(self.num_wslots)],
            wslot_used = [False] * self.num_wslots,
            w_scratch = bufs["w_scratch"],
            # Fused-tier (exl3_moe) temp buffers per (hidden, intermediate) shape and the
            # batched reconstruct tier state per layer, shared with the preallocated buffers
            fused_t = TUNING.stream_fused_t,
            fused_bufs = bufs["fused_bufs"],
            # Batched reconstruct tier state per layer (moe_batch_recon.BatchReconLayer)
            recon = bufs["recon"],
        )

        # Probe pinned->device bandwidth once: the break-even assignment count for streaming an
        # expert scales inversely with the link's bandwidth, so a chipset-attached x4 card needs
        # a much hotter expert to justify the weight DMA than a CPU-direct x16 one. An explicit
        # EXL3_MOE_STREAM_T overrides the scaling.
        probe = min(self.wslot_size, 16 << 20)
        ev0, ev1 = torch.cuda.Event(enable_timing = True), torch.cuda.Event(enable_timing = True)
        def timed_copy():
            ev0.record(st["copy_stream"])
            st["vram_slots"][0][:probe // 2].copy_(self.wviews[0][:probe // 2], non_blocking = True)
            ev1.record(st["copy_stream"])
            ev1.synchronize()
            return probe / (ev0.elapsed_time(ev1) * 1e-3) / 1e9   # GB/s
        with torch.cuda.stream(st["copy_stream"]):
            # streaming keeps the link awake, so the sustained awake rate is the one to use
            bw = probe_bandwidth(timed_copy)
        st["bw"] = bw
        if st["native_slots"] is not None:
            # The ring was carved on the compute stream during the measured load; the copy
            # stream writes into it, so its first write is ordered after that stream (the
            # allocator may have recycled a block whose last user is still queued there)
            st["copy_stream"].wait_stream(torch.cuda.current_stream())
        if TUNING.stream_t_explicit:
            st["stream_t"] = self.stream_t
        else:
            # Calibrated on Qwen3.8-Flash-Next (512 experts, 410 on the CPU, 12 threads) over
            # gen5 x16 (57 GB/s), gen5 x8 (29 GB/s) and gen4 x4 (6.7 GB/s) links: 8 was best
            # on both gen5 links (4 and 16 both slower), 16 on gen4 x4 (32 no better). The
            # break-even count grows with the square root of the bandwidth deficit, not
            # linearly: the tail's CPU cost falls with the same rows the streaming gains
            st["stream_t"] = max(self.stream_t, int(round(self.stream_t * (25.0 / max(bw, 0.5)) ** 0.5)))
        if TUNING.stream_debug:
            print(f" -- stream state cuda:{key}: pinned->device {bw:.1f} GB/s, "
                  f"stream_t {st['stream_t']}")
        self.sstate[key] = st
        return st

    def _dq_linear(self, x, trellis_view, dims, suh, svh, bias, w_scratch, out_dtype = torch.half):
        """reconstruct-path linear: had_in(x * suh) @ W -> had_out * svh (+ bias). out_dtype
        follows the resident experts (fp32 for the down projection, the model's interm_dtype
        for gate/up): on models with massive activations the output-side Hadamard concentrates
        a 128-block past the fp16 range, and an in-place fp16 transform overflows to inf"""
        k, n, K = dims
        xh = torch.empty_like(x)
        ext.had_r_128(x, xh, suh, None, 1.0)
        w = w_scratch[:k * n].view(k, n)
        ext.reconstruct(w, trellis_view, K, False, True)
        y = torch.empty((x.shape[0], n), dtype = out_dtype, device = x.device)
        ext.hgemm(xh, w, y)
        ext.had_r_128(y, y, None, svh, 1.0)
        if bias is not None:
            y += bias
        return y

    def _act(self, spec, g, u):
        act = spec["activation"]
        if act in (0, 1):
            # Nonzero act_limit clamps up symmetrically and the activated gate from above,
            # before the multiply (mirrors the act_mul kernels; DS4 ships swiglu_limit = 10
            # with plain silu)
            fn = torch.nn.functional.silu if act == 0 else torch.nn.functional.gelu
            av, uf = fn(g.float()), u.float()
            lim = spec["act_limit"]
            if lim:
                av = av.clamp(max = lim)
                uf = uf.clamp(-lim, lim)
            return (av * uf).half()
        if act == 3:
            lim = spec["act_limit"]
            gf = g.float().clamp(max = lim)
            uf = u.float().clamp(-lim, lim)
            return ((uf + 1.0) * gf * torch.sigmoid(1.702 * gf)).half()
        uf = torch.nn.functional.relu(u.float())
        return (uf * uf).half()

    def install_expert(self, layer_idx, local_idx, keys):
        """Dynamic placement: replace worker expert `local_idx` of `layer_idx` with the
        checkpoint tensors at `keys` (per-projection prefixes, gate first when gated). The
        caller must have quiesced (full stream sync => job ring drained, worker idle) before
        calling; blocks until the child acks the in-place arena copy."""
        self.conn.send(("install", layer_idx, local_idx, keys))
        msg = self.conn.recv()
        if msg[0] != "ok":
            raise RuntimeError(f"CPU MoE worker expert install failed: {msg[1] if len(msg) > 1 else msg}")

    def submit_prefill(self, layer_idx, y, selected_experts, routing_weights):
        """
        Split the routed-expert workload by per-expert token count: hot experts (count >=
        stream_t) have their weights staged by the worker, DMA'd through the pinned ring to a
        small VRAM ring on a copy stream, and computed on the GPU via the reconstruct path,
        while the cold tail runs on the CPU, compressed to the rows that still have at least one
        unmasked assignment. Tail jobs are issued before the streamed batches and collected
        after them, so the CPU works the tail while the GPU streams. Falls back to the plain CPU
        path when nothing qualifies.
        """
        spec = self.specs[layer_idx]
        rows = y.shape[0]
        if (rows < self.stream_min_rows or spec.get("expert_bytes") is None
                or spec["expert_bytes"] > self.wslot_size or layer_idx not in self.aux):
            return self.submit(layer_idx, y, selected_experts, routing_weights)

        with torch.cuda.device(y.device):
            st = self._ensure_stream_state(y.device)
            E = spec["num_experts"]
            flat = selected_experts.reshape(-1)
            # Shifted histogram so any -1 sentinels land in bin 0 instead of polluting expert 0.
            # scatter_add, not torch.bincount: bincount hides two blocking min/max reductions
            # (negative-input validation and output sizing), leaving the tolist below as the only
            # sync before the tail-row compression
            shifted = flat + 1
            counts1 = torch.zeros(E + 1, dtype = torch.long, device = flat.device)
            counts1.scatter_add_(0, shifted, torch.ones_like(shifted))
            counts1_h = counts1.tolist()
            neg, counts_h = counts1_h[0], counts1_h[1:]
            streamed = [e for e in range(E) if counts_h[e] >= st["stream_t"]]
            if TUNING.stream_debug:
                n_str = sum(counts_h[e] for e in streamed)
                print(f" -- stream L{layer_idx}: rows {rows}, streamed experts "
                      f"{len(streamed)}/{E}, assignments {n_str}/{sum(counts_h)}")
            if not streamed:
                return self.submit(layer_idx, y, selected_experts, routing_weights)
            return self._submit_prefill_streamed(
                layer_idx, y, selected_experts, routing_weights, spec, streamed, st,
                counts_h, flat, shifted, neg)

    def _stream_fused_t(self, spec, aux, h):
        """Fused-kernel row capacity for a streamed layer, 0 when the layer isn't eligible
        (same rule as support_fused on the GPU side: silu/gelu gated or relu2 gateless, no
        per-expert biases, no padded dims)"""
        return TUNING.stream_fused_t if (
            spec["activation"] in (0, 1, 2) and spec["hi"] == h and spec["ho"] == h
            and not any(aux.get(b) is not None for b in ("bias_g", "bias_u", "bias_d"))
        ) else 0

    def _stream_recon_layer(self, st, layer_idx, spec, aux, device):
        """Batched reconstruct tier state for a streamed layer, built once per (device,
        layer); None when disabled or the experts carry biases (a batched add would be needed)"""
        if not self.batch_recon or any(aux.get(b) is not None for b in ("bias_g", "bias_u", "bias_d")):
            return None
        recon = st["recon"].get(layer_idx)
        if recon is None:
            from ..modules.moe_batch_recon import BatchReconLayer
            pd = spec["proj_dims"]
            gated = pd.get("g") is not None
            scales = {p: (aux["suh_" + p], aux["svh_" + p])
                      for p in (("g", "u", "d") if gated else ("u", "d"))}
            recon = BatchReconLayer(
                pd.get("g"), pd["u"], pd["d"], (False, True), (False, True), (False, True),
                spec["activation"], spec["act_limit"], device, scales)
            st["recon"][layer_idx] = recon
        return recon

    def _stream_fused_bufs(self, st, spec, device):
        """Fused-tier temp buffers for a streamed layer's (hidden, intermediate) shape, kept
        per device (the kernel reads both dims from the buffers, so they must match the layer)"""
        key = (spec["hi"], spec["proj_dims"]["u"][1])
        fbufs = st["fused_bufs"].get(key)
        if fbufs is None:
            conc = ext.exl3_moe_max_concurrency(torch.device(device).index or 0)
            fbufs = tuple(
                torch.empty((conc, TUNING.stream_fused_t, dim), dtype = torch.half, device = device)
                for dim in (key[0], key[0], key[1], key[1]))
            st["fused_bufs"][key] = fbufs
        return fbufs

    def prefill_worst_case_parts(self, layer_idx, rows, device, assignments):
        """Autosplit worst case for one layer's prefill on `device` with `assignments` routed
        rows on CPU-resident experts, as (fixed, variable) bytes: the outputs and lists that
        live for the whole call, and the larger of one batched-reconstruct group and the
        per-expert path (freed before the GPU-side tiers allocate theirs). The persistent
        device buffers are allocated for real here (_device_buffers) so the load accounts for
        them"""
        spec = self.specs[layer_idx]
        h, hi = spec["ho"], spec["hi"]
        A = assignments
        # Plain CPU path: the fp32 output and the readback staging
        fixed = rows * h * 4 + min(self.cap_rows, rows) * h * 4
        if (rows < self.stream_min_rows or spec.get("expert_bytes") is None
                or spec["expert_bytes"] > self.wslot_size or layer_idx not in self.aux):
            return fixed, 0
        aux = self.aux[layer_idx]
        pd = spec["proj_dims"]
        gated = pd.get("g") is not None
        n = pd["u"][1]
        # Streamed path: padded fp32 output, the tail's compressed output, the input copy with
        # its zero row, and the sorted assignment lists
        fixed += (rows + 1) * h * 4 + rows * h * 4 + (rows + 1) * hi * 2 + A * 32
        with torch.cuda.device(device):
            bufs = self._device_buffers(device, spec)
            if self._stream_fused_t(spec, aux, h):
                self._stream_fused_bufs(bufs, spec, device)
            recon = self._stream_recon_layer(bufs, layer_idx, spec, aux, device)
        r = min(rows, A)
        per_expert = r * (hi * 2 + n * 2 * (2 if gated else 1) + h * 4)
        batched = recon.worst_case_bytes(A, slot_mode = False) if recon is not None else 0
        return fixed, max(per_expert, batched)

    def _submit_prefill_streamed(self, layer_idx, y, selected_experts, routing_weights, spec,
                                 streamed, st, counts_h, flat, shifted, neg):
        rows = y.shape[0]
        h = y.shape[1]
        E = spec["num_experts"]
        topk = selected_experts.shape[1]
        # One spare row: the batched reconstruct tier's padding sink (never read back)
        out_ext = torch.zeros((rows + 1, h), dtype = torch.float, device = y.device)
        out = out_ext[:rows]

        # Group assignments by expert once: every expert's token segment is then a slice at
        # host-known prefix offsets. Anything per-expert/per-batch from here on is sync-free —
        # per-expert nonzero() would pin the host to the stream position and collapse the
        # copy-stream lookahead into lockstep with compute
        order = torch.argsort(flat)
        token_sorted = torch.div(order, topk, rounding_mode = "floor")
        weight_sorted = routing_weights.reshape(-1).index_select(0, order)
        offs = [neg]
        for c in counts_h:
            offs.append(offs[-1] + c)

        # CPU tail: mask streamed assignments, then compress to rows that still carry work (a
        # nearly-fully-streamed layer otherwise pays a full cap_rows-chunked pass of no-ops).
        # Issue only; the waits and readbacks come after the streamed batches are enqueued.
        # The streamed table is built host-side and indexed with the shifted ids (entry 0 is the
        # -1 sentinel, always False), replacing the index_put/clamp/compare/and kernel chain
        table = np.zeros(E + 1, dtype = np.bool_)
        for e in streamed:
            table[e + 1] = True
        smask1 = host_to_device(torch.from_numpy(table), y.device)
        is_streamed = smask1.index_select(0, shifted)
        sel_tail = flat.masked_fill(is_streamed, -1).view(rows, topk)
        tidx = (sel_tail >= 0).any(dim = 1).nonzero(as_tuple = True)[0]
        n_tail = tidx.shape[0]
        tail_jobs = None
        if n_tail:
            out_t = torch.empty((n_tail, h), dtype = torch.float, device = y.device)
            tail_jobs, rtmp = self._issue_compute(
                layer_idx, y.index_select(0, tidx), sel_tail.index_select(0, tidx),
                routing_weights.index_select(0, tidx), spec, out_t, h)

        aux = self.aux[layer_idx]
        pd = spec["proj_dims"]
        gb, ub, db = spec["proj_bytes"]
        exp_b = spec["expert_bytes"]
        per_slot = min(self.wslot_size // exp_b, self.batch_experts)
        gated = pd.get("g") is not None
        abort = self.gpu_base_ptr + 128
        copy_stream = st["copy_stream"]
        # Pinned arena: per-expert (chunk, offset) of the DMA source; None = staged path
        blocks = self.layer_blocks[layer_idx] if self.pinned else None

        # Packed-layout policy of this layer's projections (see _stream_layout): packed bytes
        # are restored to native order after the DMA, so the kernels below always see native
        layout, restore = self._stream_layout(spec)

        # Mid-tier experts (count <= fused_t) run through the fused MoE kernel per staged batch;
        # experts too hot for the temp buffers take the per-expert reconstruct path. Same
        # eligibility as support_fused on the GPU side: mul1 (given), silu/gelu gated or relu2
        # gateless, no per-expert biases, no padded dims
        fused_t = self._stream_fused_t(spec, aux, h)
        recon = self._stream_recon_layer(st, layer_idx, spec, aux, y.device)
        recon_ctx = {}
        fbufs = None
        if fused_t and any(counts_h[e] <= fused_t for e in streamed):
            fbufs = self._stream_fused_bufs(st, spec, y.device)

        for i0 in range(0, len(streamed), per_slot):
            batch = streamed[i0:i0 + per_slot]
            ws = self.next_wslot
            self.next_wslot = (self.next_wslot + 1) % self.num_wslots
            self.wseq += 1
            seq = self.wseq

            if blocks is not None:
                # Pinned arena: DMA each expert's contiguous block straight out of the
                # registered arena mapping on the copy stream (no worker involvement, the
                # stager thread stays idle)
                with torch.cuda.stream(copy_stream):
                    if st["wslot_used"][ws]:
                        copy_stream.wait_event(st["wconsumed_ev"][ws])
                    raw = st["vram_slots"][ws]
                    for bi, e in enumerate(batch):
                        ci, off = blocks[e]
                        raw[bi * exp_b // 2 : (bi + 1) * exp_b // 2].copy_(
                            self.arena_views[ci][off // 2 : (off + exp_b) // 2],
                            non_blocking = True)
            else:
                # Stage job: its own ring, consumed by the worker's dedicated stager thread, so
                # the weight memcpys overlap the compute pool's work on the tail
                stail = int(self.v_stage_tail[0])
                while stail - int(self.v_stage_head[0]) >= MOE_STAGE_RING - 2:
                    import time
                    if self.v_abort[0] or not self.proc.is_alive():
                        raise RuntimeError("CPU MoE worker failed (stage ring stall)")
                    time.sleep(0.0002)
                job = self.v_stage_jobs[stail % MOE_STAGE_RING]
                _set_job_seq(job, seq)
                job[1] = layer_idx
                job[2] = len(batch)
                job[3] = 0
                job[4] = ws
                job[5] = 1    # MOE_JOB_KIND_STAGE
                job[6] = self.wslot_prev_seq[ws] & 0xFFFFFFFF   # compared cyclically by the worker
                for bi, e in enumerate(batch):
                    job[7 + bi] = e
                self.v_stage_tail[0] = stail + 1
                self.wslot_prev_seq[ws] = seq

                used = (len(batch) * exp_b) // 2
                with torch.cuda.stream(copy_stream):
                    if st["wslot_used"][ws]:
                        copy_stream.wait_event(st["wconsumed_ev"][ws])
                    ext.exl3_moe_flag_wait(self.stage_done_addr[ws], seq, abort)
                    st["vram_slots"][ws][:used].copy_(self.wviews[ws][:used], non_blocking = True)
                    ext.exl3_moe_flag_write(self.pinned_free_addr[ws], seq)

            if restore and st["native_slots"] is None:
                # First batch on this device that needs native-order bytes: adopt the ring the
                # measured load allocated, or allocate it now (manual loads) on the compute
                # stream. The caching allocator ties the block to the stream it was carved on,
                # so the copy stream's first write into it is ordered after that stream
                st["native_slots"] = self._device_buffers(y.device, spec)["native_slots"]
                copy_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(copy_stream):
                if restore:
                    # Restore the native tile order on the copy stream, one launch per
                    # projection over the whole batch (group-0 projections in an otherwise
                    # packed layer take the plain-copy path in the same kernel)
                    for pi, (name, off) in enumerate((("g", 0), ("u", gb), ("d", gb + ub))):
                        if not pd.get(name):
                            continue
                        k, n, K = pd[name]
                        ext.moe_unswizzle_trellis(
                            st["vram_slots"][ws], st["native_slots"][ws], len(batch), exp_b, off,
                            k // 16, n // 16, K, layout[pi][0], layout[pi][1])
                st["wready_ev"][ws].record(copy_stream)
            st["wslot_used"][ws] = True

            # Compute the batch on the current stream once the DMA lands
            torch.cuda.current_stream().wait_event(st["wready_ev"][ws])
            vslot = st["native_slots"][ws] if restore else st["vram_slots"][ws]
            per_e = [(bi, e, token_sorted[offs[e] : offs[e] + counts_h[e]],
                      weight_sorted[offs[e] : offs[e] + counts_h[e]])
                     for bi, e in enumerate(batch)]
            if TUNING.stream_debug:
                n_fused = sum(1 for e in batch if counts_h[e] <= fused_t) if fused_t and fbufs is not None else 0
                print(f" --   batch L{layer_idx} ws{ws}: {len(batch)} experts, fused_t {fused_t}, "
                      f"n_fused {n_fused}, counts {[counts_h[e] for e in batch]}")

            # Fused and batched tiers (run_staged_experts); what neither takes (experts above
            # the batched tier's row cap, or every heavy expert without a batched tier) runs on
            # the per-expert loop below. Streamed experts are mul1 by construction
            single = run_staged_experts(
                y, out_ext, [(bi, e, offs[e], counts_h[e]) for bi, e in enumerate(batch)],
                token_sorted, weight_sorted, vslot.data_ptr(), exp_b, (0, gb, gb + ub), gated,
                lambda e: (aux["suh_g"][e].data_ptr() if gated else 0, aux["svh_g"][e].data_ptr() if gated else 0,
                           aux["suh_u"][e].data_ptr(), aux["svh_u"][e].data_ptr(),
                           aux["suh_d"][e].data_ptr(), aux["svh_d"][e].data_ptr()),
                (pd["g"][2] if gated else pd["u"][2], pd["u"][2], pd["d"][2]),
                spec["activation"], float(spec["act_limit"] or 0.0), (False, True, False, True, False, True),
                fused_t, fbufs, TUNING.mtile, recon, recon_ctx)
            single_ids = {e for _, e in single}
            for bi, e, idx, wseg in per_e:
                if e not in single_ids:
                    continue
                boff = (bi * exp_b) // 2
                xg = y.index_select(0, idx)
                # Zero-pad to the quantized input width (the had transform requires it)
                hi = spec["hi"]
                if xg.shape[1] != hi:
                    xg = torch.nn.functional.pad(xg, (0, hi - xg.shape[1]))
                we = wseg.float().unsqueeze(1)
                def tview(off_b, dims):
                    k, n, K = dims
                    numel = (k // 16) * (n // 16) * int(16 * K)
                    return vslot[boff + off_b // 2 : boff + off_b // 2 + numel] \
                        .view(k // 16, n // 16, int(16 * K))
                # Same output dtypes as the resident experts: the intermediate as the model's
                # interm_dtype, the down projection fp32 (the activation casts to half for the
                # down GEMM either way)
                idt = torch.float if spec.get("interm_fp32") else torch.half
                if gated:
                    gy = self._dq_linear(xg, tview(0, pd["g"]), pd["g"],
                                         aux["suh_g"][e], aux["svh_g"][e],
                                         aux["bias_g"][e] if aux.get("bias_g") else None,
                                         st["w_scratch"], out_dtype = idt)
                uy = self._dq_linear(xg, tview(gb, pd["u"]), pd["u"],
                                     aux["suh_u"][e], aux["svh_u"][e],
                                     aux["bias_u"][e] if aux.get("bias_u") else None,
                                     st["w_scratch"], out_dtype = idt)
                a = self._act(spec, gy if gated else None, uy) if gated else self._act(spec, None, uy)
                dy = self._dq_linear(a, tview(gb + ub, pd["d"]), pd["d"],
                                     aux["suh_d"][e], aux["svh_d"][e],
                                     aux["bias_d"][e] if aux.get("bias_d") else None,
                                     st["w_scratch"], out_dtype = torch.float)
                out.index_add_(0, idx, dy[:, :h] * we)
            st["wconsumed_ev"][ws].record(torch.cuda.current_stream())

        # Collect the CPU tail (by now usually complete) and merge
        if tail_jobs:
            self._collect_compute(tail_jobs, out_t, rtmp, h)
            out.index_add_(0, tidx, out_t)
        return out

    def unregister(self):
        # Called per offloaded layer on unload; shut down when the model releases the last
        # one. A LIVE COUNT, not specs.pop(): register_layer hands out stable indices into
        # specs (and returns cached indices on autosplit rollback retries), so removing
        # entries desyncs every later layer's index and the ack bookkeeping — under tight
        # autosplit budgets with split layers on every device this hung the load
        if self.live_layers == 0:
            return
        self.live_layers -= 1
        if self.live_layers == 0:
            self.shutdown()

    def shutdown(self):
        if self.proc is not None:
            try:
                if self.started and self.shm is not None:
                    self.v_quit[0] = 1
                    # The child's main thread serves the pipe at runtime (expert installs);
                    # the flag stops its compute thread, the message unblocks the recv loop
                    if self.conn is not None:
                        self.conn.send(("quit",))
                elif self.conn is not None:
                    self.conn.send(("quit",))
                self.proc.join(timeout = 5)
                if self.proc.is_alive():
                    self.proc.terminate()
                    self.proc.join(timeout = 2)
                if self.proc.is_alive():
                    self.proc.kill()
            except Exception:
                pass
            self.proc = None
        # Full reset: a later re-registration must not resolve stale indices against a child
        # that no longer exists
        self.specs = []
        self.by_key = {}
        self.aux = {}
        self.acked = 0
        self.live_layers = 0
        self.seq = self.wseq = 0
        self.next_slot = self.next_wslot = 0
        self.slot_last_seq = [0] * MOE_MAX_SLOTS
        self.wslot_prev_seq = [0] * MOE_MAX_WSLOTS
        cleanupper.unregister_atexit(self.shutdown)
        if self.conn is not None:
            try:
                self.conn.close()
            except Exception:
                pass
            self.conn = None
        if self.shm is not None:
            try:
                cuda_host_unregister(self.base_ptr)
            except Exception:
                pass
            # Drop every view over the buffer before closing, or mmap refuses to unmap
            self.slots = None
            self.wviews = None
            self.sstate = None
            self._dev_bufs = {}
            self.v_quit = self.v_pass_wake = self.v_abort = self.v_ready = None
            self.v_thread_caps = None
            self.v_jobs_tail = self.v_jobs_head = self.v_jobs = None
            self.v_stage_tail = self.v_stage_head = self.v_stage_jobs = None
            self._flags_u32 = None
            import gc
            gc.collect()
        # Pinned arena mappings: unpin, drop the views, unmap. The pages themselves die with
        # the worker (memfd, no name to unlink; a Windows section goes with its last handle)
        for i, view in enumerate(self.arena_views):
            if i < len(self.arena_dev) and self.arena_dev[i]:
                unregister_mapped_home(view.data_ptr(), view.numel() * view.element_size())
                continue
            try:
                cuda_host_unregister(view.data_ptr())
            except Exception:
                pass
        self.arena_views = []
        self.arena_dev = []
        self.layer_blocks = []
        # Home chunks: unregister (registered ones only), drop the views, unmap
        for view, dev in zip(self.home_views, self.home_dev):
            if dev:
                unregister_mapped_home(view.data_ptr(), view.numel())
        self.home_views = []
        self.home_dev = []
        self.layer_home_blocks = []
        maps, self.arena_maps = self.arena_maps + self.home_maps, []
        self.home_maps = []
        if maps:
            import gc
            gc.collect()
            for m in maps:
                try:
                    m.close()
                except Exception:
                    pass
        if self.shm is not None:
            try:
                self.shm.close()
                self.shm.unlink()
            except Exception:
                pass
            self.shm = None
        self.started = False
        self.by_key = {}


def check_workers(*configs):
    """Host-side check after the host has synchronized with a forward pass (the generator calls
    it once per iteration, after sampling): a worker that died while jobs were in flight had its
    flag waits released by the watchdog, so that forward folded in stale CPU partials. The
    watchdog sets the abort flag before releasing anything, hence this check sees it for every
    such forward, and the iteration's results are discarded with an error instead of returned"""
    for config in configs:
        if config is None:
            continue
        for name, h in getattr(config, "moe_cpu_hosts", {}).items():
            if h.aborted():
                raise RuntimeError(
                    f"CPU MoE worker {name} died or timed out during the last forward pass; its "
                    f"CPU expert outputs are invalid and the iteration's results were discarded")

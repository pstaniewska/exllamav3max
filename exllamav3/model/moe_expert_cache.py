"""
Shared GPU expert cache for hybrid MoE execution: one owner per CUDA device.

The owner holds the only VRAM budget for cached experts on its device. Every MoE layer that runs
in hybrid mode (main model, MTP head, draft model alike) registers with the owner of its device
and hands it the per-expert trellis rows of its pointer tables. From then on the owner decides
what those rows point at:

- a VRAM slot holding the expert (resident; the layer's kernels read VRAM),
- the expert's pinned + mapped host home (zero-copy; the kernels read host memory over PCIe),
- the layer's bounce block (valid VRAM placeholder) when the home is not GPU-readable; such an
  expert is never computed by the GPU unless resident (the dispatcher gives it a CPU lane).

Admission and eviction during inference happen inside moe_hybrid_issue (CLOCK, on the compute
stream). This module does the rest on the host: registration, slot classes, budget sizing,
shrinking ahead of memory pressure, regrowth when idle, warm start, unregistration. Rules that keep
it safe under asynchronous execution (see also moe_hybrid.cu):

1. All kernels that read slots or tables of a device run on the stream(s) noted in begin_pass, in
   order with the issue kernel that last changed them.
2. Host-side edits that only enable or redirect (enabling new slots, installing warm experts,
   resetting a layer's rows to its homes) are enqueued on the current stream, so they are ordered
   with every later kernel and need no synchronization.
3. Host-side edits that free memory (disabling slots before a chunk is released, unregistering
   a layer whose tables are about to be freed) first quiesce: an event is recorded on every noted
   stream and waited for, then the edit runs and is itself waited for, and only then is the
   memory released. A chunk tensor is never dropped while a kernel could still read it.
4. Homes are immutable for the lifetime of the registration; an eviction only repoints rows.

Budget: the cache is sized lazily at the first forward pass (after the KV cache, recurrent
state, draft/MTP heads and their statics exist), to the memory actually free on the device minus a
floor: the larger of EXL3_EXPERT_CACHE_RESERVE_MB and the measured worst-case transient of a
forward on the device plus EXL3_EXPERT_CACHE_MIN_FREE_MB, capped by EXL3_EXPERT_CACHE_LIMIT_GB.
Before every prefill-size pass the owner checks the floor again and gives back whole chunks while
it is short; it regrows one chunk at a time when the generator goes idle. Loading another model
component releases every chunk first (release_all) so the load sees the device as it would
without the cache.

Slot classes: experts of different byte sizes (different models or quantizations) get separate
slot classes, each with its own chunks and CLOCK hand, all drawing from the one device budget in
proportion to the expert bytes registered in each class.
"""

from __future__ import annotations
import os
import json
import threading
import atexit
import time
import collections
import numpy as np
import torch

from . import moe_hybrid_ref as ref
from .moe_hybrid_ref import (
    EMAX, MAX_TABLES, MAX_NODES, MAX_CLASSES, S_COUNT, CTL_SIZE, STATS_SIZE, STAT_NAMES,
    CTL_BG_ADMIT, CTL_GPU_GATE, CTL_HAND, CTL_CLASS_LO, CTL_CLASS_HI, ST_NO_LANE,
    OP_EVICT, OP_DISABLE, OP_ENABLE, OP_INSTALL, OP_RESET_LAYER,
)


def _env_int(name, default):
    v = os.environ.get(name, "")
    return int(v) if v.strip() else default


class CacheTuning:
    """Expert cache knobs, read once at import (mutate the module-level TUNING for sweeps)"""

    def __init__(self):
        lim = os.environ.get("EXL3_EXPERT_CACHE_LIMIT_GB", "").strip()
        self.limit_bytes = int(float(lim) * (1 << 30)) if lim else None
        self.reserve_bytes = _env_int("EXL3_EXPERT_CACHE_RESERVE_MB", 1536) << 20
        self.min_free_bytes = _env_int("EXL3_EXPERT_CACHE_MIN_FREE_MB", 1024) << 20
        self.admit_max = _env_int("EXL3_EXPERT_CACHE_ADMIT_MAX", 64)
        # Frequency-gated admissions of CPU-lane misses per decode layer call (0 = only GPU-lane
        # misses are admitted, the old behaviour)
        self.bg_admit = max(0, _env_int("EXL3_EXPERT_CACHE_BG_ADMIT", 1))
        # GPU-lane misses evict only a slot whose expert was picked at most as often (else zero-copy);
        # 0 = admit on every GPU-lane miss, the old behaviour
        self.gpu_gate = 1 if _env_int("EXL3_EXPERT_CACHE_GPU_GATE", 1) else 0
        self.warm = os.environ.get("EXL3_EXPERT_CACHE_WARM") or None
        self.save_stats = os.environ.get("EXL3_EXPERT_CACHE_SAVE_STATS") or None
        self.gmax = max(1, _env_int("EXL3_EXPERT_CACHE_GATHER_MAX", 64))
        self.gather_blocks = max(1, _env_int("EXL3_EXPERT_CACHE_GATHER_BLOCKS", 16))
        self.chunk_slots = max(1, _env_int("EXL3_EXPERT_CACHE_CHUNK_SLOTS", 32))
        self.lcap = max(1, _env_int("EXL3_EXPERT_CACHE_MAX_LAYERS", 256))
        self.grow_hysteresis = _env_int("EXL3_EXPERT_CACHE_GROW_HYST_MB", 256) << 20
        self.decay = float(os.environ.get("EXL3_EXPERT_CACHE_DECAY", 0.5))
        self.policy = os.environ.get("EXL3_EXPERT_CACHE_POLICY", "clock").strip().lower()
        assert self.policy in ("clock", "none"), "EXL3_EXPERT_CACHE_POLICY must be clock or none"
        self.debug = os.environ.get("EXL3_EXPERT_CACHE_DEBUG", "0") not in ("", "0")
        # Short-lived counter log: one JSON line per interval (counters only, never request
        # content), the file holding only the newest stats_log_max lines
        self.stats_log = os.environ.get("EXL3_EXPERT_CACHE_STATS_LOG") or None
        self.stats_log_s = max(1, _env_int("EXL3_EXPERT_CACHE_STATS_LOG_S", 60))
        self.stats_log_max = max(1, _env_int("EXL3_EXPERT_CACHE_STATS_LOG_MAX", 1440))


TUNING = CacheTuning()

_OOM = getattr(torch, "OutOfMemoryError", None) or torch.cuda.OutOfMemoryError

_NO_COST = torch.zeros((4,), dtype = torch.float)

_CACHES: dict = {}
_CACHES_LOCK = threading.Lock()

# Worst-case forward transient per device index, noted by every component's autosplit load
# (max-merged, so a later component with a smaller transient never lowers it); read by the
# device's owner whenever it computes its floor, whether the owner exists yet or not
_DEVICE_TRANSIENT: dict = {}


def note_device_transient(device_index: int, nbytes: int, device_type: str = "cuda"):
    key = (device_type, device_index)
    with _CACHES_LOCK:
        _DEVICE_TRANSIENT[key] = max(_DEVICE_TRANSIENT.get(key, 0), int(nbytes))


def _round_up(x, m):
    return (x + m - 1) // m * m


class SizeClass:
    def __init__(self, cid: int, slot_bytes: int, bounce: torch.Tensor):
        self.id = cid
        self.slot_bytes = slot_bytes
        self.bounce = bounce
        self.layers: set[int] = set()
        self.weight = 0                 # registered expert bytes (share of the budget)
        self.lo = 0
        self.cap = 0                    # slot capacity (multiple of chunk_slots)
        self.chunks: list = []          # per chunk index: tensor or None

    def allocated_bytes(self) -> int:
        return sum(c.numel() for c in self.chunks if c is not None)


class CacheLayer:
    """Registration record of one hybrid MoE layer on a device"""

    def __init__(self, uid, key, L, num_experts, expert_bytes, proj_off, cls, tables):
        self.uid = uid
        self.key = key
        self.L = L
        self.num_experts = num_experts
        self.expert_bytes = expert_bytes
        self.proj_off = proj_off
        self.cls = cls
        self.tables = tables            # [(tensor, elem_offset, stride, proj)]; refs keep them alive
        self.homes = np.zeros(num_experts, dtype = np.int64)
        self.host_view = None           # e -> CPU uint8 tensor of the expert block (warm installs)
        self.cpuloc = np.full((MAX_NODES, num_experts), -1, dtype = np.int32)


class ExpertCache:

    @staticmethod
    def for_device(device, backend: str | None = None, mem_info = None) -> "ExpertCache":
        """The single owner for `device` (created on first request)"""
        device = torch.device(device)
        key = (device.type, device.index or 0)
        with _CACHES_LOCK:
            c = _CACHES.get(key)
            if c is None:
                c = ExpertCache(device, backend or ("ref" if device.type == "cpu" else "cuda"), mem_info)
                _CACHES[key] = c
            return c

    @staticmethod
    def all() -> list["ExpertCache"]:
        with _CACHES_LOCK:
            return list(_CACHES.values())

    @staticmethod
    def _drop_registry():
        """Tests: forget every owner and every noted transient"""
        with _CACHES_LOCK:
            _CACHES.clear()
            _DEVICE_TRANSIENT.clear()

    def __init__(self, device, backend: str = "cuda", mem_info = None, tuning: CacheTuning | None = None):
        self.device = torch.device(device)
        self.backend = backend
        assert backend in ("cuda", "ref")
        self.T = tuning or TUNING
        self._mem_info = mem_info
        self.lock = threading.RLock()
        self.layers: dict[int, CacheLayer] = {}
        self.by_uid: dict[str, CacheLayer] = {}
        self.free_L = list(range(self.T.lcap - 1, -1, -1))
        self.classes: list[SizeClass] = []
        self.sized = False
        self.grow_pending = False
        self.streams: dict[int, object] = {}
        self.transient_hint = 0
        self.extra_seen = 0                 # largest prefill extra requested via ensure_headroom
        self.warmed = False
        self.scap = 0
        self.notices = set()

        lcap, dev = self.T.lcap, self.device
        def full(n, v, dt):
            return torch.full((n,), v, dtype = dt, device = dev)
        self.arrays = dict(
            slotof = full(lcap * EMAX, -1, torch.int32),
            ctl = full(CTL_SIZE, 0, torch.int32),
            stats = full(STATS_SIZE, 0, torch.int64),
            gjobs = full(self.T.gmax * 3, 0, torch.int64),
            hist = full(lcap * EMAX, 0, torch.float),
            reg_tables = full(lcap * MAX_TABLES, 0, torch.int64),
            reg_tstride = full(lcap * MAX_TABLES, 0, torch.int32),
            reg_tproj = full(lcap * MAX_TABLES, 0, torch.int32),
            reg_projoff = full(lcap * 3, 0, torch.int64),
            reg_home = full(lcap * EMAX, 0, torch.int64),
            reg_bounce = full(lcap, 0, torch.int64),
            reg_bytes = full(lcap, 0, torch.int64),
            reg_class = full(lcap, -1, torch.int32),
            reg_cpuloc = full(lcap * MAX_NODES * EMAX, -1, torch.int32),
        )
        self._alloc_slot_arrays(0)
        if self.T.save_stats:
            atexit.register(self._save_stats_atexit)

    # ---- state descriptor -------------------------------------------------------------------

    def _alloc_slot_arrays(self, scap: int):
        n = max(1, scap)
        dev = self.device
        self.arrays["owner"] = torch.full((n,), -1, dtype = torch.int32, device = dev)
        self.arrays["refb"] = torch.zeros((n,), dtype = torch.int32, device = dev)
        self.arrays["pin"] = torch.full((n,), -1, dtype = torch.int32, device = dev)
        self.arrays["slot_base"] = torch.zeros((n,), dtype = torch.int64, device = dev)
        self.scap = scap
        self._rebuild_state()

    def _rebuild_state(self):
        a = self.arrays
        d = [0] * S_COUNT
        order = ["slotof", "owner", "refb", "pin", "slot_base", "ctl", "stats", "gjobs", "hist",
                 "reg_tables", "reg_tstride", "reg_tproj", "reg_projoff", "reg_home", "reg_bounce",
                 "reg_bytes", "reg_class", "reg_cpuloc"]
        for i, k in enumerate(order):
            d[i] = a[k].data_ptr()
        d[ref.S_SCAP] = self.scap
        d[ref.S_LCAP] = self.T.lcap
        d[ref.S_GMAX] = self.T.gmax
        self.state = torch.tensor(d, dtype = torch.long)
        self.ref_state = ref.RefState(a, self.scap, self.T.lcap, self.T.gmax) if self.backend == "ref" else None

    # ---- kernel entry points (CUDA kernels, or the reference on CPU) ------------------------

    def run_host_ops(self, ops: list[tuple[int, int, int, int]]):
        if not ops:
            return
        t = torch.tensor(ops, dtype = torch.long).view(-1)
        if self.backend == "ref":
            ref.host_ops(self.ref_state, t.numpy())
            return
        from ..ext import exllamav3_ext as ext
        with torch.cuda.device(self.device):
            td = t.to(self.device)
            ext.moe_hybrid_host_ops(self.state, td, len(ops))

    def issue(self, layer: CacheLayer, sel, y, w, sel_gpu, lane_out, cpu_sel, nodes, node_cost, hi,
              prefill, admit, gpu_max, cpu_frac, prefill_gpu_min, node_policy, pref_node, lane_cost = None,
              gather = True):
        """One layer call's lane assignment (see moe_hybrid_issue). nodes: int64 CPU (ND_ROWS, n),
        node_cost: float CPU (n), lane_cost: float CPU (4) or None (cost model off). gather = False:
        the caller runs the decode admission copy itself (gather) once it has published the CPU jobs,
        so the copy stays off the CPU lanes' critical path. Returns whether a gather is owed"""
        if self.policy_none():
            admit = 0
        if lane_cost is None:
            lane_cost = _NO_COST
        with self.lock:
            if self.backend == "ref":
                nd = dict(n = nodes.shape[1], h_sel = nodes[ref.ND_SEL].tolist(), h_x = nodes[ref.ND_X].tolist(),
                          h_w = nodes[ref.ND_W].tolist(), dev_count = nodes[ref.ND_COUNT].tolist(),
                          slot = nodes[ref.ND_SLOT].tolist(), xcap = nodes[ref.ND_XCAP].tolist(),
                          selcap = nodes[ref.ND_SELCAP].tolist(), cost = node_cost.tolist())
                ref.issue(
                    self.ref_state, nd, layer.L, layer.num_experts, sel.view(-1).numpy(),
                    y.numpy() if y is not None else None, hi, w.view(-1).numpy(), sel_gpu.view(-1).numpy(),
                    lane_out.numpy() if lane_out is not None else None,
                    cpu_sel.view(-1).numpy() if cpu_sel is not None else None,
                    prefill, admit, gpu_max, cpu_frac, prefill_gpu_min, node_policy, pref_node,
                    lane_cost.tolist())
                if admit and not prefill:
                    if not gather:
                        return True
                    ref.gather(self.ref_state)
                return False
            from ..ext import exllamav3_ext as ext
            ext.moe_hybrid_issue(
                self.state, layer.L, layer.num_experts, sel.view(-1), y, w.view(-1), sel_gpu,
                lane_out, cpu_sel, nodes, node_cost, hi, prefill, admit, gpu_max, cpu_frac, lane_cost,
                prefill_gpu_min, node_policy, pref_node)
            if admit and not prefill:
                if not gather:
                    return True
                ext.moe_hybrid_gather(self.state, sel_gpu, self.T.gather_blocks)
            return False

    def gather(self, sel_gpu):
        """Decode admission copy owed by issue(..., gather = False), stream-ordered after it"""
        with self.lock:
            if self.backend == "ref":
                ref.gather(self.ref_state)
                return
            from ..ext import exllamav3_ext as ext
            ext.moe_hybrid_gather(self.state, sel_gpu, self.T.gather_blocks)

    def policy_none(self) -> bool:
        return self.T.policy == "none"

    # ---- registration -------------------------------------------------------------------------

    def bounce_for(self, expert_bytes: int) -> torch.Tensor:
        """VRAM placeholder block for experts of this size (the class bounce): valid memory for the
        layer's Linear views and for rows of experts whose home is not GPU-readable"""
        with self.lock:
            return self._class_for(expert_bytes).bounce

    def _class_for(self, expert_bytes: int) -> SizeClass:
        slot_bytes = _round_up(expert_bytes, 256)
        for c in self.classes:
            if c.slot_bytes == slot_bytes:
                return c
        if len(self.classes) >= MAX_CLASSES:
            # Reuse the id of a class no layer uses any more (its chunks were dropped when its
            # last layer left); the slot layout changes either way, handled below
            dead = next((c for c in self.classes if not c.layers), None)
            if dead is None:
                raise RuntimeError(f"Expert cache on {self.device}: more than {MAX_CLASSES} distinct expert sizes "
                                   f"in use at once")
            cid = dead.id
            bounce = torch.zeros((slot_bytes,), dtype = torch.uint8, device = self.device)
            c = SizeClass(cid, slot_bytes, bounce)
            if self.sized:
                self.release_all()
                self.sized = False
            self.classes[cid] = c
            return c
        bounce = torch.zeros((slot_bytes,), dtype = torch.uint8, device = self.device)
        c = SizeClass(len(self.classes), slot_bytes, bounce)
        self.classes.append(c)
        if self.sized:
            # A new class changes the slot layout: drop every chunk now (rows go back to homes),
            # re-layout at the next pass
            self.release_all()
            self.sized = False
        return c

    def register_layer(self, uid: str, key: str, num_experts: int, expert_bytes: int,
                       proj_off: tuple[int, int, int], tables: list) -> CacheLayer:
        """tables: [(tensor (int64, device), elem_offset, stride, proj)] -- every table holding a
        per-expert trellis address of this layer. Rows start at the bounce block"""
        assert 0 < num_experts <= EMAX, f"{key}: {num_experts} experts exceed the cache's {EMAX}"
        assert len(tables) <= MAX_TABLES
        assert expert_bytes % 16 == 0, "expert blocks must be 16-byte multiples"
        with self.lock:
            if uid in self.by_uid:
                raise RuntimeError(f"Expert cache: layer {uid} is already registered on {self.device}")
            if not self.free_L:
                raise RuntimeError(f"Expert cache: more than {self.T.lcap} hybrid layers on {self.device} "
                                   f"(EXL3_EXPERT_CACHE_MAX_LAYERS)")
            cls = self._class_for(expert_bytes)
            L = self.free_L.pop()
            layer = CacheLayer(uid, key, L, num_experts, expert_bytes, proj_off, cls, tables)
            a = self.arrays
            tv = [0] * MAX_TABLES
            ts = [0] * MAX_TABLES
            tp = [0] * MAX_TABLES
            for i, (tensor, off, stride, proj) in enumerate(tables):
                assert tensor.dtype == torch.long and tensor.device == self.device and tensor.is_contiguous()
                assert tensor.numel() >= off + (num_experts - 1) * stride + 1
                tv[i] = tensor.data_ptr() + 8 * off
                ts[i] = stride
                tp[i] = proj
            self._write(a["reg_tables"], L * MAX_TABLES, tv)
            self._write(a["reg_tstride"], L * MAX_TABLES, ts)
            self._write(a["reg_tproj"], L * MAX_TABLES, tp)
            self._write(a["reg_projoff"], L * 3, list(proj_off))
            self._write(a["reg_bounce"], L, [cls.bounce.data_ptr()])
            self._write(a["reg_bytes"], L, [expert_bytes])
            self._write(a["reg_class"], L, [cls.id])
            a["reg_home"][L * EMAX: (L + 1) * EMAX].zero_()
            a["reg_cpuloc"][L * MAX_NODES * EMAX: (L + 1) * MAX_NODES * EMAX].fill_(-1)
            a["slotof"][L * EMAX: (L + 1) * EMAX].fill_(-1)
            a["hist"][L * EMAX: (L + 1) * EMAX].zero_()
            self.run_host_ops([(OP_RESET_LAYER, L, num_experts, 0)])
            cls.layers.add(L)
            cls.weight += expert_bytes * num_experts
            self.layers[L] = layer
            self.by_uid[uid] = layer
            if self.sized:
                # The new layer changes the class shares; existing chunks stay, growth rebalances
                self.grow_pending = True
            return layer

    def set_layer_homes(self, layer: CacheLayer, homes: np.ndarray, host_view = None):
        """homes: device addresses of each expert's native-layout host block (0 = not GPU-readable).
        Rows of non-resident experts are repointed accordingly (stream-ordered)"""
        assert homes.shape == (layer.num_experts,)
        with self.lock:
            layer.homes = homes.astype(np.int64)
            layer.host_view = host_view
            row = np.zeros(EMAX, dtype = np.int64)
            row[:layer.num_experts] = layer.homes
            self._write(self.arrays["reg_home"], layer.L * EMAX, row.tolist())
            self.run_host_ops([(OP_RESET_LAYER, layer.L, layer.num_experts, 0)])

    def set_layer_cpuloc(self, layer: CacheLayer, cpuloc: np.ndarray):
        """cpuloc: (num_nodes, E) worker-local expert index per worker node, -1 = not held"""
        with self.lock:
            full = np.full((MAX_NODES, EMAX), -1, dtype = np.int32)
            full[:cpuloc.shape[0], :layer.num_experts] = cpuloc
            layer.cpuloc = full[:, :layer.num_experts].copy()
            self._write(self.arrays["reg_cpuloc"], layer.L * MAX_NODES * EMAX, full.reshape(-1).tolist())

    def unregister_layer(self, layer: CacheLayer):
        """Before the layer's tables are freed: evict its slots and clear its registry rows so no
        later eviction writes into freed memory (quiesced; see rule 3)"""
        with self.lock:
            if self.by_uid.get(layer.uid) is not layer:
                return
            # Stats: snapshot (merged into the file) when the last layer of a component leaves
            tag = layer.uid.split(":", 1)[0]
            if self.T.save_stats and not any(l is not layer and l.uid.split(":", 1)[0] == tag
                                             for l in self.layers.values()):
                self.save_stats(self.T.save_stats)
            self.quiesce()
            ops = []
            if self.scap:
                row = self.arrays["slotof"][layer.L * EMAX: layer.L * EMAX + layer.num_experts].cpu()
                ops = [(OP_EVICT, 0, 0, int(s)) for s in row.tolist() if s >= 0]
            self.run_host_ops(ops)
            a = self.arrays
            L = layer.L
            a["reg_tables"][L * MAX_TABLES: (L + 1) * MAX_TABLES].zero_()
            a["reg_class"][L] = -1
            a["reg_home"][L * EMAX: (L + 1) * EMAX].zero_()
            a["reg_cpuloc"][L * MAX_NODES * EMAX: (L + 1) * MAX_NODES * EMAX].fill_(-1)
            a["hist"][L * EMAX: (L + 1) * EMAX].zero_()
            self._sync_current()
            layer.cls.layers.discard(L)
            layer.cls.weight -= layer.expert_bytes * layer.num_experts
            del self.layers[L]
            del self.by_uid[layer.uid]
            self.free_L.append(L)
            if not self.layers:
                self.release_all()
                self.sized = False
                self.grow_pending = False
                # Nothing references the bounce blocks any more (the layers' Linears are gone).
                # Per-load hints go too: the next model is measured again when it loads
                self.classes = []
                self.transient_hint = 0
                self.extra_seen = 0
                self.warmed = False
                with _CACHES_LOCK:
                    _DEVICE_TRANSIENT.pop((self.device.type, self.device.index or 0), None)
            elif not layer.cls.layers:
                # The class is dead: its chunks would stay allocated and count against the limit
                # while no layer can use them
                self._drop_class_chunks(layer.cls)
                self.grow_pending = True

    # ---- memory and budget ---------------------------------------------------------------------

    def _write(self, tensor: torch.Tensor, offset: int, values: list):
        src = torch.tensor(values, dtype = tensor.dtype)
        tensor[offset: offset + len(values)].copy_(src)

    def mem_info(self) -> tuple[int, int]:
        """(available, total) bytes: free device memory plus the caching allocator's reserved but
        unallocated pool (both usable for the next allocation)"""
        if self._mem_info is not None:
            return self._mem_info()
        free, total = torch.cuda.mem_get_info(self.device)
        pool = torch.cuda.memory_reserved(self.device) - torch.cuda.memory_allocated(self.device)
        return free + max(0, pool), total

    def floor_bytes(self) -> int:
        """Free VRAM the cache always leaves: the reserve, or the largest known transient (this
        device's autosplit measurements of every component, the hints noted here, the largest
        prefill extra seen) plus EXL3_EXPERT_CACHE_MIN_FREE_MB, whichever is larger"""
        dev_t = _DEVICE_TRANSIENT.get((self.device.type, self.device.index or 0), 0)
        t = max(self.transient_hint, dev_t, self.extra_seen)
        return max(self.T.reserve_bytes, t + self.T.min_free_bytes)

    def note_transient(self, nbytes: int):
        """Worst-case transient of a forward on this device (autosplit measurement, MoE prefill
        bound at the load's chunk size): the floor keeps at least this much free"""
        with self.lock:
            self.transient_hint = max(self.transient_hint, int(nbytes))

    def cache_bytes(self) -> int:
        return sum(c.allocated_bytes() for c in self.classes)

    def chunk_bytes(self, c: SizeClass) -> int:
        return c.slot_bytes * self.T.chunk_slots

    def note_stream(self, stream):
        if stream is not None:
            self.streams[int(stream.cuda_stream) if hasattr(stream, "cuda_stream") else id(stream)] = stream

    def quiesce(self):
        """Wait for every kernel that could read slots or tables of this device (rule 3)"""
        if self.backend == "ref":
            return
        with torch.cuda.device(self.device):
            for s in list(self.streams.values()) + [torch.cuda.current_stream(self.device)]:
                ev = torch.cuda.Event()
                ev.record(s)
                ev.synchronize()

    def _sync_current(self):
        if self.backend == "ref":
            return
        with torch.cuda.device(self.device):
            ev = torch.cuda.Event()
            ev.record(torch.cuda.current_stream(self.device))
            ev.synchronize()

    def _layout(self):
        """Slot index layout per class, sized for the whole device (metadata only: a few bytes
        per potential slot), so chunks can come and go without moving any slot index. The CLOCK
        range of a class (CTL_CLASS_HI) covers only its enabled prefix (see _sync_range).
        Replaces the slot arrays: in-flight kernels may still read the old ones, so quiesce"""
        self.quiesce()
        _, total = self.mem_info()
        cap_total = total if self.T.limit_bytes is None else min(total, self.T.limit_bytes)
        wsum = sum(c.weight for c in self.classes) or 1
        lo = 0
        for c in self.classes:
            share = cap_total * c.weight // wsum
            nchunks = max(0, share // self.chunk_bytes(c)) if c.weight else 0
            if c.weight and nchunks == 0 and self.chunk_bytes(c) <= cap_total:
                # A small class's share rounds down to nothing: give it one chunk (the limit
                # still caps what grow allocates)
                nchunks = 1
            c.lo = lo
            c.cap = nchunks * self.T.chunk_slots
            c.chunks = [None] * nchunks
            lo += c.cap
        self._alloc_slot_arrays(lo)
        ctl = [0] * CTL_SIZE
        for c in self.classes:
            ctl[CTL_CLASS_LO + 2 * c.id] = c.lo
            ctl[CTL_CLASS_HI + 2 * c.id] = c.lo     # nothing enabled yet
            ctl[CTL_HAND + c.id] = 0
        ctl[CTL_BG_ADMIT] = self.T.bg_admit
        ctl[CTL_GPU_GATE] = self.T.gpu_gate
        epoch = int(self.arrays["ctl"][0].item())
        ctl[0] = epoch
        self._write(self.arrays["ctl"], 0, ctl)

    def enabled_chunks(self, c: SizeClass) -> int:
        """Enabled chunks of a class form a prefix: grow takes the lowest free index, shrink
        gives back the highest enabled one"""
        n = 0
        while n < len(c.chunks) and c.chunks[n] is not None:
            n += 1
        assert all(x is None for x in c.chunks[n:]), "expert cache: enabled chunks are not a prefix"
        return n

    def _sync_range(self, c: SizeClass):
        """Bound the class's CLOCK scan to its enabled slots (stream-ordered with the kernels):
        a scan over a long disabled tail would end in no_victim for every admission while the
        hand crosses it. The kernel resets a hand that falls outside the range"""
        hi = c.lo + self.enabled_chunks(c) * self.T.chunk_slots
        self._write(self.arrays["ctl"], CTL_CLASS_HI + 2 * c.id, [hi])

    def ensure_sized(self):
        """First sizing (lazy, at the first pass) or regrowth after release_all / new layers"""
        with self.lock:
            if self.T.policy == "none" or not self.layers:
                return
            if not self.sized:
                self._layout()
                self.sized = True
                self.grow(regrow = False)
                if not self.warmed:
                    # Once per owner generation: a later re-layout (new size class) keeps the
                    # live pick history instead of reloading the file's counts
                    self.warm_start()
                    self.warmed = True
                self.grow_pending = False
                if self.T.limit_bytes is not None:
                    for c in self.classes:
                        if c.weight and not any(x is not None for x in c.chunks):
                            self._notice(f"nochunk{c.slot_bytes}",
                                         f" !! expert cache {self.device}: no slot chunk of {self.chunk_bytes(c) >> 20} MiB "
                                         f"({self.T.chunk_slots} x {c.slot_bytes >> 10} KiB experts) fits "
                                         f"EXL3_EXPERT_CACHE_LIMIT_GB; this expert size is not cached (lower "
                                         f"EXL3_EXPERT_CACHE_CHUNK_SLOTS or raise the limit)")
            elif self.grow_pending:
                self.grow(regrow = False)
                self.grow_pending = False

    def grow(self, regrow: bool = True) -> int:
        """Allocate chunks while the floor holds (and the limit allows); returns bytes added"""
        added = 0
        with self.lock:
            if not self.sized:
                return 0
            avail, _ = self.mem_info()
            room = avail - self.floor_bytes() - (self.T.grow_hysteresis if regrow else 0)
            limit = self.T.limit_bytes
            cur = self.cache_bytes()
            skip = set()
            while True:
                cands = [c for c in self.classes if c.weight and None in c.chunks and c.id not in skip]
                if not cands:
                    break
                c = min(cands, key = lambda c: c.allocated_bytes() / c.weight)
                b = self.chunk_bytes(c)
                if b > room or (limit is not None and cur + b > limit):
                    # This class's chunk does not fit; a class with smaller chunks still may
                    skip.add(c.id)
                    continue
                ci = c.chunks.index(None)
                try:
                    t = torch.empty((b,), dtype = torch.uint8, device = self.device)
                except _OOM:
                    self._notice("oom", f" -- expert cache {self.device}: chunk allocation failed, growth stopped")
                    break
                c.chunks[ci] = t
                s0 = c.lo + ci * self.T.chunk_slots
                bases = [t.data_ptr() + i * c.slot_bytes for i in range(self.T.chunk_slots)]
                self._write(self.arrays["slot_base"], s0, bases)
                self.run_host_ops([(OP_ENABLE, 0, 0, s0 + i) for i in range(self.T.chunk_slots)])
                self._sync_range(c)
                room -= b
                cur += b
                added += b
            if added and self.T.debug:
                print(f" -- expert cache {self.device}: +{added >> 20} MiB, {cur >> 20} MiB in "
                      f"{sum(1 for c in self.classes for x in c.chunks if x is not None)} chunks", flush = True)
        return added

    def shrink(self, nbytes: int) -> int:
        """Give back at least `nbytes` (whole chunks, largest share first); returns bytes freed"""
        freed = 0
        with self.lock:
            if not self.sized:
                return 0
            victims = []
            plan = {c.id: [i for i, x in enumerate(c.chunks) if x is not None] for c in self.classes}
            while freed < nbytes:
                cands = [c for c in self.classes if plan[c.id]]
                if not cands:
                    break
                c = max(cands, key = lambda c: len(plan[c.id]) * self.chunk_bytes(c) / max(c.weight, 1))
                ci = plan[c.id].pop()
                victims.append((c, ci))
                freed += self.chunk_bytes(c)
            if not victims:
                return 0
            self.quiesce()
            ops = []
            for c, ci in victims:
                s0 = c.lo + ci * self.T.chunk_slots
                ops += [(OP_DISABLE, 0, 0, s0 + i) for i in range(self.T.chunk_slots)]
            hold = []
            for c, ci in victims:
                hold.append(c.chunks[ci])
                c.chunks[ci] = None
            # Range first (stream-ordered before the disable ops), so no later scan visits a
            # disabled slot of a freed chunk; the ops then evict and disable
            for c in {c.id: c for c, _ in victims}.values():
                self._sync_range(c)
            self.run_host_ops(ops)
            self._sync_current()
            del hold
            if self.backend == "cuda":
                torch.cuda.empty_cache()
            if self.T.debug:
                print(f" -- expert cache {self.device}: -{freed >> 20} MiB, {self.cache_bytes() >> 20} MiB left",
                      flush = True)
        return freed

    def _drop_class_chunks(self, c: SizeClass):
        idx = [i for i, x in enumerate(c.chunks) if x is not None]
        if not idx:
            return
        self.quiesce()
        hold = [c.chunks[i] for i in idx]
        for i in idx:
            c.chunks[i] = None
        self._sync_range(c)
        ops = [(OP_DISABLE, 0, 0, c.lo + i * self.T.chunk_slots + k) for i in idx for k in range(self.T.chunk_slots)]
        self.run_host_ops(ops)
        self._sync_current()
        del hold
        if self.backend == "cuda":
            torch.cuda.empty_cache()

    def ensure_headroom(self, extra: int = 0) -> int:
        """Shrink until the floor is free again; `extra` is the MoE transient of the coming pass.
        The floor already holds the device's measured forward transient (which includes the
        hybrid layers' bound at the load's chunk size), so `extra` raises the requirement only
        where it exceeds that transient, and it is remembered so idle regrowth stops short of
        it (no shrink / regrow churn per request). Returns bytes freed"""
        with self.lock:
            if not self.sized:
                return 0
            self.extra_seen = max(self.extra_seen, int(extra))
            avail, _ = self.mem_info()
            deficit = self.floor_bytes() - avail
            return self.shrink(deficit) if deficit > 0 else 0

    def release_all(self):
        """Drop every chunk (rows go back to homes); regrown at the next pass"""
        with self.lock:
            if not self.sized:
                return
            total = self.cache_bytes()
            if total:
                self.shrink(total)
            self.grow_pending = True

    # ---- passes and idle maintenance -------------------------------------------------------------

    def begin_pass(self, rows: int, prefill_rows: int, stream = None):
        with self.lock:
            self.note_stream(stream)
            if not self.sized or self.grow_pending:
                self.ensure_sized()
            elif rows >= prefill_rows:
                self.ensure_headroom()

    def idle_maintenance(self):
        """Generator queue drained: decay the pick history, check invariants, regrow"""
        with self.lock:
            if not self.layers:
                return
            self.arrays["hist"].mul_(self.T.decay)
            st = self.stats()
            if self.T.stats_log:
                self._log_stats(st)
            if st["no_lane"]:
                raise RuntimeError(
                    f"Expert cache {self.device}: {st['no_lane']} expert picks had no execution lane "
                    f"(no GPU-readable home and no CPU worker copy); this is a configuration error")
            if self.sized:
                self.grow(regrow = True)

    _LOG_KEYS = ("hits", "misses", "admits", "bg_admits", "bg_rejects", "gpu_rejects", "evictions", "zero_copy", "cpu",
                 "calls", "admit_bytes")

    def _log_stats(self, st: dict):
        """Append the counter deltas since the previous line to T.stats_log, at most once per
        T.stats_log_s, keeping the newest T.stats_log_max lines (called at idle points only)"""
        now = time.time()
        last = getattr(self, "_log_last", None)
        if last is not None and now - last[0] < self.T.stats_log_s:
            return
        if last is None:
            self._log_lines = collections.deque(maxlen = self.T.stats_log_max)
            prev = {k: 0 for k in self._LOG_KEYS}
        else:
            prev = last[1]
        d = {k: int(st[k]) - int(prev[k]) for k in self._LOG_KEYS}
        cur = {k: int(st[k]) for k in self._LOG_KEYS}
        self._log_last = (now, cur)
        hm = d["hits"] + d["misses"]
        line = dict(t = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now)), device = str(self.device),
                    hit_rate = round(d["hits"] / hm, 4) if hm else None, **d)
        self._log_lines.append(json.dumps(line))
        try:
            tmp = self.T.stats_log + ".tmp"
            with open(tmp, "w", encoding = "utf8") as f:
                f.write("\n".join(self._log_lines) + "\n")
            os.replace(tmp, self.T.stats_log)
        except OSError as e:
            self._notice("stats_log", f" !! expert cache: cannot write EXL3_EXPERT_CACHE_STATS_LOG ({e})")

    def stats(self) -> dict:
        """Device counters (one D2H read: call between passes, never per token)"""
        v = self.arrays["stats"].cpu().tolist()
        out = {k: v[i] for i, k in enumerate(STAT_NAMES)}
        out["cpu_by_node"] = v[ref.ST_CPU_NODE: ref.ST_CPU_NODE + MAX_NODES]
        return out

    def summary(self) -> dict:
        with self.lock:
            s = self.stats()
            occupied = int((self.arrays["owner"] >= 0).sum().item()) if self.scap else 0
            enabled = int((self.arrays["pin"] >= 0).sum().item()) if self.scap else 0
            avail, total = self.mem_info()
            s.update(dict(
                device = str(self.device), layers = len(self.layers), cache_mib = self.cache_bytes() >> 20,
                slots_enabled = enabled, slots_occupied = occupied,
                slot_utilization = round(occupied / enabled, 4) if enabled else 0.0,
                floor_mib = self.floor_bytes() >> 20,
                vram_limit_mib = (self.T.limit_bytes >> 20) if self.T.limit_bytes is not None else None,
                vram_available_mib = avail >> 20, vram_total_mib = total >> 20,
                hit_rate = round(s["hits"] / (s["hits"] + s["misses"]), 4) if s["hits"] + s["misses"] else None,
                classes = [dict(slot_bytes = c.slot_bytes, layout_slots = c.cap,
                                chunks = sum(1 for x in c.chunks if x is not None), layers = len(c.layers))
                           for c in self.classes]))
            return s

    # ---- warm start and stats -------------------------------------------------------------------

    def warm_start(self):
        """Fill free slots with the experts the warm-start stats rank hottest per class (host-driven
        copies from the homes, then OP_INSTALL, all on the current stream)"""
        path = self.T.warm
        if not path:
            return
        try:
            with open(path, encoding = "utf8") as f:
                stats = json.load(f)
        except (OSError, ValueError) as e:
            self._notice("warm", f" !! expert cache: cannot read EXL3_EXPERT_CACHE_WARM ({e}), starting cold")
            return
        installed = 0
        for c in self.classes:
            free = [c.lo + ci * self.T.chunk_slots + i
                    for ci, x in enumerate(c.chunks) if x is not None for i in range(self.T.chunk_slots)]
            ranked = []
            for L in c.layers:
                layer = self.layers[L]
                counts = stats.get(layer.uid, stats.get(layer.key))
                if not counts or len(counts) != layer.num_experts or layer.host_view is None:
                    continue
                hist = self.arrays["hist"]
                self._write(hist, L * EMAX, [float(x) for x in counts])
                # Experts without a home copy (CPU-only experts of a mixed-rate layer) never
                # take a slot
                ranked += [(float(counts[e]), L, e) for e in range(layer.num_experts)
                           if counts[e] > 0 and layer.host_view(e) is not None]
            ranked.sort(key = lambda r: (-r[0], r[1], r[2]))
            ops = []
            for (_, L, e), s in zip(ranked, free):
                layer = self.layers[L]
                dst = self._slot_tensor(c, s)[:layer.expert_bytes]
                src = layer.host_view(e)
                dst.copy_(src, non_blocking = self.backend == "cuda" and src.is_pinned())
                ops.append((OP_INSTALL, L, e, s))
            self.run_host_ops(ops)
            installed += len(ops)
        if self.T.debug and installed:
            print(f" -- expert cache {self.device}: warm start installed {installed} experts", flush = True)

    def _slot_tensor(self, c: SizeClass, s: int) -> torch.Tensor:
        i = s - c.lo
        ci, si = divmod(i, self.T.chunk_slots)
        t = c.chunks[ci]
        return t[si * c.slot_bytes: (si + 1) * c.slot_bytes]

    def hist_by_layer(self) -> dict:
        h = self.arrays["hist"].cpu().numpy()
        return {l.uid: h[l.L * EMAX: l.L * EMAX + l.num_experts].tolist() for l in self.layers.values()}

    def save_stats(self, path: str):
        with self.lock:
            data = self.hist_by_layer()
        try:
            prev = {}
            if os.path.exists(path):
                with open(path, encoding = "utf8") as f:
                    prev = json.load(f)
            prev.update(data)
            with open(path, "w", encoding = "utf8") as f:
                json.dump(prev, f)
        except (OSError, ValueError) as e:
            print(f" !! expert cache: could not save stats to {path} ({e})", flush = True)

    def _save_stats_atexit(self):
        try:
            if self.layers:
                self.save_stats(self.T.save_stats)
        except Exception:
            pass

    def _notice(self, key, msg):
        if key not in self.notices:
            self.notices.add(key)
            print(msg, flush = True)

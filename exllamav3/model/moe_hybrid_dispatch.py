"""
Hybrid CPU/GPU expert dispatcher (EXL3_HYBRID_MOE=1 / -mh).

    BlockSparseMLP routing (unchanged)
      -> HybridDispatcher.submit: one moe_hybrid_issue kernel decides a lane per selected expert
           resident in the device's expert cache   -> GPU, VRAM slot (cache hit)
           non-resident, admitted this call        -> GPU, VRAM slot (gather copy, same stream)
           non-resident, home GPU-readable         -> GPU, zero-copy read of the host home
           non-resident, CPU lane chosen           -> worker of NUMA node n (shard or replica)
      -> the layer's unmodified GPU MoE tiers run over the pointer tables (GPU lanes only)
      -> HybridDispatcher.combine: CPU partials folded in, stream-ordered

One dispatcher per model component (main model, MTP head, draft model), each with its own worker
group (moe_numa_hosts.HostGroup); all components on a device share that device's single
ExpertCache owner and budget (moe_expert_cache). Speculative decoding is invisible here: a verify
forward is just a forward with more rows, routed through the same decode (rows up to the CPU
slot capacity, EXL3_EXPERT_DECODE_MAX_ROWS) or prefill path. Nothing depends on q_len == 1.

Numerics: every GPU lane runs the unmodified ExLlamaV3 kernels on the original EXL3 weights,
whether read from VRAM or over PCIe, so with EXL3_EXPERT_DISPATCH_POLICY=gpu_exact the layer is
identical to the all-VRAM model. CPU lanes use the existing int8-activation CPU kernels (the same
approximation as -mcl / -mcs); the hybrid policy trades that for latency, explicitly.

Lane policy inputs are measured, not assumed: the PCIe rate comes from a timed copy out of a
registered home at finalize, each node's CPU cost from per-job worker telemetry (compute time per
assignment). Until both are measured, EXL3_MOE_ZC_CPU_FRAC_INIT sets the CPU share of misses.

Prefill uses the same measurements. A node's job cost is fitted as a per expert plus b per
assignment (prefill_cost), and each call chooses the row threshold between the zero-copy and CPU
lanes (choose_prefill_gpu_min). Experts with fewer rows go to the CPU, larger ones are read over
PCIe, so the PCIe reads and the CPU nodes finish together. A faster CPU kernel (the AMX tier)
lowers b, and the threshold rises with it. Until the fit exists, the bootstrap threshold
PREFILL_GPU_MIN_BOOT applies; EXL3_EXPERT_PREFILL_GPU_MIN=<n> fixes the threshold.
"""

from __future__ import annotations
import contextlib
import os
import time
import weakref
import numpy as np
import torch

from .moe_expert_cache import ExpertCache, TUNING as CACHE_TUNING
from .moe_hybrid_ref import (
    EMAX, LANE_ZC, MAX_NODES, ND_ROWS, ND_SEL, ND_X, ND_W, ND_COUNT, ND_SLOT, ND_XCAP, ND_SELCAP,
    LC_HIT, LC_ZC, LC_CPU_FIXED, LC_CPU,
)

DISPATCH_POLICIES = ("legacy", "gpu_exact", "hybrid")
NODE_POLICIES = {"numa_local": 0, "numa_sharded": 0, "numa_balanced": 1, "numa_adaptive": 2}
HOME_MODES = ("native_copy", "worker_arena")

# Cache layer ids must be unique per device: a draft model loads with the main model's component
# name ("text") and usually the same module keys. A dispatcher takes its tag when its first layer
# loads and gives it back when its last layer detaches: the plain component name when no live
# dispatcher holds it (stable warm-start / stats keys across reloads), else "<comp>#n"
_LIVE_TAGS: set = set()

# CPU cost samples come only from decode-shaped jobs (rows per job up to this many): prefill
# jobs carry many rows per expert and would overstate the per-expert cost of a decode miss
COST_SAMPLE_MAX_ROWS = 8

# Prefill threshold (rows for a non-resident expert to go zero-copy) until the prefill cost fit
# is measured, and the former fixed default
PREFILL_GPU_MIN_BOOT = 8
# Decay per telemetry sample of the prefill cost fit
PREFILL_FIT_DECAY = 0.98


def acquire_tag(comp: str) -> str:
    tag, n = comp, 0
    while tag in _LIVE_TAGS:
        n += 1
        tag = f"{comp}#{n}"
    _LIVE_TAGS.add(tag)
    return tag


def release_tag(tag: str | None):
    if tag is not None:
        _LIVE_TAGS.discard(tag)


def resolve_node_policy(policy: str) -> str:
    """EXL3_NUMA_POLICY "auto": measured cost split when experts are replicated on every node;
    with shards the policy has no choice to make, so keep the GPU-local rule"""
    if policy != "auto":
        return policy
    from .moe_numa_hosts import memory_policy
    return "numa_adaptive" if memory_policy() == "replicate" else "numa_local"


class HybridTuning:

    def __init__(self):
        self.enabled = os.environ.get("EXL3_HYBRID_MOE", "0") not in ("", "0")
        self.dispatch = os.environ.get("EXL3_EXPERT_DISPATCH_POLICY", "hybrid").strip().lower()
        assert self.dispatch in DISPATCH_POLICIES, f"EXL3_EXPERT_DISPATCH_POLICY must be one of {DISPATCH_POLICIES}"
        self.home = os.environ.get("EXL3_EXPERT_HOME", "native_copy").strip().lower()
        assert self.home in HOME_MODES, f"EXL3_EXPERT_HOME must be one of {HOME_MODES}"
        # Unset or "auto": numa_adaptive with replicated experts (every node holds every expert, so
        # spread the work by measured cost), numa_local with shards (the owner is the only holder)
        self.node_policy = os.environ.get("EXL3_NUMA_POLICY", "auto").strip().lower()
        assert self.node_policy in NODE_POLICIES or self.node_policy in ("legacy", "auto"), \
            f"EXL3_NUMA_POLICY must be auto, legacy or one of {tuple(NODE_POLICIES)}"
        zc = os.environ.get("EXL3_MOE_ZC_MAX", "auto").strip().lower()
        self.zc_max = None if zc == "auto" else int(zc)
        self.cpu_frac_init = float(os.environ.get("EXL3_MOE_ZC_CPU_FRAC_INIT", 0.5))
        self.zc_fused_rows = int(os.environ.get("EXL3_EXPERT_CACHE_ZC_FUSED_ROWS", 32))
        # "auto": chosen per prefill call from measured costs; a number fixes it. Default 1 (every
        # non-resident expert of a prefill call on the GPU): the per-call cost model sent too many
        # experts to the CPU lanes and measured 155-169 tok/s against 252 with 1 (H100 NVL, two
        # Xeon 6760P, GLM 5.3 at 4.91 bpw, 16k prompt); the CPU-only split ran at 88
        pgm = os.environ.get("EXL3_EXPERT_PREFILL_GPU_MIN", "1").strip().lower()
        self.prefill_gpu_min = None if pgm in ("", "auto") else int(pgm)
        # Prefill stream lane (StreamWork): zero-copy experts of wide prefill calls are DMAed
        # into a VRAM ring on a copy stream and computed from there. "auto": calls of at least
        # stream_rows rows; "1": every prefill call; "0": off. CUDA only
        self.stream = os.environ.get("EXL3_EXPERT_PREFILL_STREAM", "auto").strip().lower() or "auto"
        assert self.stream in ("auto", "0", "1"), "EXL3_EXPERT_PREFILL_STREAM must be auto, 0 or 1"
        self.stream_rows = int(os.environ.get("EXL3_EXPERT_PREFILL_STREAM_ROWS", 2048))
        self.stream_min_rows = max(1, int(os.environ.get("EXL3_EXPERT_PREFILL_STREAM_MIN_ROWS", 1)))
        # Ring size: at most stream_slots experts and stream_mb MiB, in stream_bufs buffers
        self.stream_slots = int(os.environ.get("EXL3_EXPERT_PREFILL_STREAM_SLOTS", 32))
        self.stream_mb = int(os.environ.get("EXL3_EXPERT_PREFILL_STREAM_MB", 512))
        self.stream_bufs = max(2, int(os.environ.get("EXL3_EXPERT_PREFILL_STREAM_BUFS", 4)))
        # Widest pass that takes the decode path (cost model, cache admission, no host sync),
        # capped by the CPU slot capacity. Unset: the slot capacity, so speculative verify passes
        # (batch x (draft + 1) rows, hybrid drafts of up to 63 tokens at batch 1) and batched
        # decode stay decode-shaped. Below EXL3_MOE_STREAM_MIN_ROWS restores the former boundary
        dmr = os.environ.get("EXL3_EXPERT_DECODE_MAX_ROWS", "").strip()
        self.decode_max_rows = int(dmr) if dmr else None
        self.ema = float(os.environ.get("EXL3_MOE_ADAPTIVE_EMA", 0.1))
        self.debug = os.environ.get("EXL3_HYBRID_MOE_DEBUG", "0") not in ("", "0")
        # Exposed CPU-lane wait measured with CUDA events (no host sync: completed event pairs
        # are read at the next pass start with Event.query)
        self.profile = os.environ.get("EXL3_HYBRID_MOE_PROFILE", "0") not in ("", "0")
        self.remeasure = int(os.environ.get("EXL3_MOE_COST_REMEASURE", 256))
        # Decode thread cap per CPU worker, chosen from measured token times (DecodeThreadTuner);
        # a fraction of the pool such as 0.75 fixes it, "off" leaves the workers' caps alone
        dt = os.environ.get("EXL3_MOE_DECODE_THREADS", "auto").strip().lower()
        self.decode_threads = None if dt in ("", "auto") else -1.0 if dt in ("off", "0") else float(dt)
        # Expert deferral (DeferredPartial): a decode layer's CPU partial reaches the residual at
        # the end of the next block instead of the layer's own routed sum, so the CPU job overlaps
        # the next block's GPU work. An approximation: that block's attention and router see the
        # residual without it
        self.defer = os.environ.get("EXL3_MOE_DEFER", "0") not in ("", "0")
        # Partial deferral: only the CPU-lane picks with the smallest routing weights are
        # deferred, at most defer_max per token (0: every CPU pick), never one whose weight is
        # >= defer_min_w (None: no bound). The rest is collected inside the layer as usual
        self.defer_max = int(os.environ.get("EXL3_MOE_DEFER_MAX", "0") or 0)
        mw = os.environ.get("EXL3_MOE_DEFER_MIN_W", "").strip()
        self.defer_min_w = float(mw) if mw else None

    def defer_partial(self) -> bool:
        """Deferral splits each node's CPU job into an immediate and a deferred one"""
        return self.defer_max > 0 or self.defer_min_w is not None


HT = HybridTuning()


class DecodeThreadTuner:
    """Picks how many of each CPU worker's pool threads run one-token (decode) jobs, from the
    measured wall time between consecutive decode passes.

    A decode job is a few expert GEMVs split across the whole pool, with a barrier per phase:
    every participant that another process preempts (a VM, kubelet, the host's own threads)
    stalls all of them, every layer, every token. Fewer participants leave cores for the
    neighbours and finish sooner once that wait outweighs the lost bandwidth, which depends on
    the machine and its load, so it is measured: each candidate fraction of the pool runs for a
    block of decode passes, the best median wins, and the search repeats now and then around
    the winner as the load changes. Prefill-shaped jobs keep the full pool."""

    FRACTIONS = (1.0, 0.875, 0.75, 0.625, 0.5, 0.375)
    SETTLE = 2          # passes discarded after a switch
    SAMPLES = 12        # timed passes per candidate
    HOLD = 2048         # decode passes on the winner before re-searching its neighbours
    GAP_S = 1.0         # a longer gap between passes is idle time, not a token
    MARGIN = 0.05       # a smaller share must beat a larger one by this much to win

    def __init__(self, fixed: float | None = None):
        self.fixed = fixed
        self.cands = list(self.FRACTIONS)
        self.idx = 0
        self.best = None
        self.samples: list[float] = []
        self.results: dict[float, float] = {}
        self.skip = self.SETTLE
        self.hold = 0
        self.t_prev = None
        self.applied = None

    def current(self) -> float:
        if self.fixed is not None:
            return self.fixed
        return self.best if self.hold > 0 else self.cands[self.idx]

    def on_pass(self, decode: bool, now: float) -> float | None:
        """Call at every pass start; returns the fraction to apply when it changes"""
        if self.fixed is not None and self.fixed < 0:
            return None
        if self.fixed is not None:
            return None if self.applied == self.fixed else self._apply(self.fixed)
        prev, self.t_prev = self.t_prev, (now if decode else None)
        if not decode or prev is None or now - prev > self.GAP_S:
            return None if self.applied is not None else self._apply(self.current())
        dt = now - prev
        if self.hold > 0:
            self.hold -= 1
            if self.hold == 0:
                self._neighbours()
            return None if self.applied == self.current() else self._apply(self.current())
        if self.skip > 0:
            self.skip -= 1
            return None
        self.samples.append(dt)
        if len(self.samples) < self.SAMPLES:
            return None
        self.results[self.cands[self.idx]] = float(np.median(self.samples))
        self.samples = []
        self.skip = self.SETTLE
        self.idx += 1
        if self.idx >= len(self.cands):
            self.best = self._pick()
            self.hold = self.HOLD
            self.idx = 0
        return self._apply(self.current())

    def _pick(self) -> float:
        """Largest fraction within MARGIN of the fastest: the medians are noisy on a shared host,
        and a smaller pool only pays when it is clearly faster"""
        t_best = min(self.results.values())
        return max(f for f, t in self.results.items() if t <= t_best * (1.0 + self.MARGIN))

    def _neighbours(self):
        """Re-search the winner and its neighbours on the full ladder"""
        f = list(self.FRACTIONS)
        i = f.index(self.best)
        self.cands = f[max(0, i - 1): i + 2]
        self.idx = 0
        self.results = {}
        self.samples = []
        self.skip = self.SETTLE

    def _apply(self, frac: float) -> float:
        self.applied = frac
        return frac


def decode_rows(cap_rows: int) -> int:
    """Rows up to which a pass takes the decode path: EXL3_EXPERT_DECODE_MAX_ROWS, at most the CPU
    slot capacity (the decode path stages every row of the call into one worker slot)"""
    want = cap_rows if HT.decode_max_rows is None else HT.decode_max_rows
    return max(0, min(want, cap_rows))


class PrefillCostFit:
    """One node's CPU job cost as a * experts + b * assignments (microseconds), least squares
    over the node's telemetry records with exponential forgetting. Decode jobs (about one
    assignment per expert) pin a; prefill jobs (many per expert) separate b from it"""

    def __init__(self, decay: float = PREFILL_FIT_DECAY):
        self.decay = decay
        self.s = np.zeros(5)                # sum ee, ea, aa, eu, au

    def add(self, experts: int, assignments: int, us: float):
        e, a = float(experts), float(assignments)
        self.s *= self.decay
        self.s += (e * e, e * a, a * a, e * us, a * us)

    def solve(self) -> tuple[float, float] | None:
        """(a, b), both >= 0, or None while the records cannot separate them"""
        ee, ea, aa, eu, au = self.s
        if ee <= 0 or aa <= 0:
            return None
        det = ee * aa - ea * ea
        if det <= 1e-3 * ee * aa:
            return None
        a = (eu * aa - au * ea) / det
        b = (au * ee - eu * ea) / det
        if b < 0:
            return eu / ee, 0.0
        if a < 0:
            return 0.0, au / aa
        return a, b


class MarginalCostFit:
    """One node's decode-shaped CPU job cost as f + c * experts (microseconds), least squares with
    exponential forgetting. c is the node's marginal cost of one more expert, the quantity a
    node choice changes; us / experts would fold the fixed per-job cost f into it, so a node
    handed fewer experts per job looks dearer per expert and is handed fewer still"""

    def __init__(self, decay: float = PREFILL_FIT_DECAY):
        self.decay = decay
        self.s = np.zeros(5)                # sum 1, e, ee, u, eu

    def add(self, experts: int, us: float):
        e = float(experts)
        self.s *= self.decay
        self.s += (1.0, e, e * e, us, e * us)

    def marginal(self) -> float | None:
        """c > 0, or None while the records cannot separate it from f"""
        n, se, see, su, seu = self.s
        det = n * see - se * se
        if n <= 0 or det <= 1e-3 * n * see:
            return None
        c = (n * seu - se * su) / det
        return c if c > 0 else None


def choose_prefill_gpu_min(counts: np.ndarray, t_gpu0: float, t_zc: float, t_fixed: float,
                           a: float, b: float, nodes: int, cpu0: tuple[int, int] = (0, 0)) -> int:
    """Row threshold for one prefill call: non-resident experts with at least this many rows read
    over PCIe (t_zc each, after t_gpu0 of other GPU work), the rest go to the CPU nodes (t_fixed
    once, then a + b * rows per expert, spread over `nodes`). cpu0 = (experts, rows) the CPU
    takes regardless (no GPU-readable home). The CPU side costs more per row and the PCIe side
    does not, so the smallest experts are the CPU's. Minimizes the later of the two finishing
    times; ties keep the experts on the GPU (exact numerics)"""
    c = np.sort(counts[counts > 0])
    m = c.size
    if not m:
        return PREFILL_GPU_MIN_BOOT
    nodes = max(nodes, 1)
    base = a * cpu0[0] + b * cpu0[1]
    cpu = (base + np.concatenate(([0.0], np.cumsum(a + b * c.astype(np.float64))))) / nodes
    k = np.arange(m + 1)
    cpu = np.where((k > 0) | (base > 0), cpu + t_fixed, 0.0)
    gpu = t_gpu0 + (m - k) * t_zc
    worst = np.maximum(cpu, gpu)
    # A threshold cannot split equal counts: only k at a change of count (or the ends)
    ok = np.ones(m + 1, dtype = bool)
    ok[1:m] = c[1:] != c[:-1]
    worst = np.where(ok, worst, np.inf)
    best = int(np.argmin(worst))
    return int(c[best]) if best < m else int(c[-1]) + 1


def prefill_stream_on(rows: int, cuda: bool) -> bool:
    """Whether a prefill call of `rows` rows uses the stream lane (EXL3_EXPERT_PREFILL_STREAM)"""
    if not cuda or HT.stream == "0":
        return False
    return HT.stream == "1" or rows >= HT.stream_rows


def stream_ring_shape(expert_bytes: int) -> tuple[int, int, int] | None:
    """(buffers, experts per buffer, bytes per expert) of the stream lane's VRAM ring: at most
    EXL3_EXPERT_PREFILL_STREAM_SLOTS experts and _MB MiB, split into _BUFS buffers (fewer when
    the budget holds fewer experts). Experts start 256-byte aligned, as in a cache slot. None
    when the budget holds fewer than two experts (no double buffering)"""
    stride = (expert_bytes + 255) // 256 * 256
    total = min(HT.stream_slots, (HT.stream_mb << 20) // stride)
    if total < 2:
        return None
    nbufs = min(HT.stream_bufs, total)
    return nbufs, total // nbufs, stride


def select_stream_experts(lanes: np.ndarray, counts: np.ndarray, min_rows: int,
                          max_rows: int | None = None) -> np.ndarray:
    """Experts of a prefill call that take the stream lane: zero-copy lane, at least min_rows
    rows (and at most max_rows when the layer has no batched tier for larger ones). Hottest
    first, so the batched tier's groups of one ring buffer have similar row counts"""
    E = counts.shape[0]
    pick = (lanes[:E] == LANE_ZC) & (counts >= min_rows)
    if max_rows is not None:
        pick &= counts <= max_rows
    ids = np.nonzero(pick)[0]
    return ids[np.argsort(-counts[ids], kind = "stable")]


def mask_streamed(sel_gpu: torch.Tensor, ids: np.ndarray, num_experts: int):
    """Stream-lane experts leave the GPU tiers' selection like CPU picks (masked to num_experts,
    which the tiers skip); stream-ordered, no host sync"""
    from ..util.device_copy import host_to_device
    table = np.zeros(num_experts + 1, dtype = np.bool_)
    table[ids] = True
    t = host_to_device(torch.from_numpy(table), sel_gpu.device)
    sel_gpu.masked_fill_(t.index_select(0, sel_gpu.view(-1)).view(sel_gpu.shape), num_experts)


class StreamRing:
    """Per (device, expert stride) state of the prefill stream lane: the copy stream and one
    (ready, consumed) event pair per ring buffer. The ring memory itself is taken per call from
    the caching allocator (StreamWork), so between prefill calls it is free memory the expert
    cache can grow into, and while a call runs it is part of the call's transient
    (HybridDispatcher.worst_case)"""

    def __init__(self, device, nbufs: int, per: int, stride: int):
        self.device = device
        self.nbufs, self.per, self.stride = nbufs, per, stride
        self.copy = torch.cuda.Stream(device = device)
        self.ready = [torch.cuda.Event() for _ in range(nbufs)]
        self.consumed = [torch.cuda.Event() for _ in range(nbufs)]


class StreamWork:
    """One prefill call's stream lane. Zero-copy experts with enough rows are copied from their
    pinned home (the same bytes the zero-copy kernels would read over PCIe) into a VRAM ring on
    the ring's copy stream, and computed from the ring by the layer's own GPU kernels (fused
    exl3_moe per row tile, batched reconstruct groups; moe_cpu_host.run_staged_experts) into an
    fp32 partial that combine folds into the routed sum. Same kernels and weights as the GPU
    tiers, so each expert's output is the all-VRAM one; only the fp32 summation order differs.

    Stream ordering (the -mcl / -mcs streamed prefill's ring protocol):
      submit:   ring memory allocated on the compute stream; copy.wait_stream(compute) so the
                first DMA is ordered after the block's previous users; record_stream(copy) so a
                call that never reaches combine cannot have its block reused under a pending DMA.
                DMAs of the first `nbufs` batches enqueued, each ending in ready[b].record(copy)
      combine:  after the GPU tiers, per batch k (buffer b = k % nbufs): compute.wait_event(
                ready[b]), the kernels, consumed[b].record(compute), then the DMA of batch
                k + nbufs into b, which starts with copy.wait_event(consumed[b]). Every DMA is
                consumed before combine returns, so the block is free on the compute stream after

    The first ring fill overlaps the call's GPU tiers and the CPU lanes; later fills overlap the
    previous buffers' compute.
    TODO: prefetch layer L+1's streamed experts during layer L (residency is fixed within a pass,
    but its selection is only known after its router)"""

    @staticmethod
    def layer_tiers(module, y) -> tuple:
        """(fused row cap, fused temp buffers, batched tier) the stream lane runs a layer's
        experts with: the layer's own (uncapped: the weights are in VRAM), None where it has none"""
        fb = module.fused_mode_buffers
        fbufs = None if fb is None else (fb.temp_state_g, fb.temp_state_u, fb.temp_intermediate_g,
                                         fb.temp_intermediate_u)
        recon = module._batch_recon_layer(y)
        return (module.fused_rows if fb is not None else 0), fbufs, (recon if recon not in (None, False) else None)

    def __init__(self, ring: StreamRing, hl, module, tiers: tuple, y, sel_flat, w, ids: np.ndarray,
                 counts: np.ndarray):
        E = hl.num_experts
        self.ring = ring
        self.hl = hl
        self.module = module
        self.tiers = tiers
        self.y = y
        self.ids = ids
        self.counts = counts
        self.host_view = hl.cache_layer.host_view
        self.nbytes = hl.expert_bytes
        per = ring.per
        self.batches = [ids[i: i + per].tolist() for i in range(0, ids.size, per)]
        # Expert-sorted assignment lists: every expert's segment is a slice at host-known offsets
        # (counts were read back with the lanes), so the batches below need no further sync
        s = torch.where((sel_flat >= 0) & (sel_flat < E), sel_flat, E)
        order = torch.argsort(s)
        self.token_sorted = torch.div(order, hl.topk, rounding_mode = "floor")
        self.weight_sorted = w.reshape(-1).index_select(0, order)
        self.offs = np.concatenate(([0], np.cumsum(counts[:E])))[:E].astype(np.int64)
        dev = y.device
        compute = torch.cuda.current_stream(dev)
        self.buf = torch.empty((ring.nbufs * per * ring.stride,), dtype = torch.uint8, device = dev)
        ring.copy.wait_stream(compute)
        self.buf.record_stream(ring.copy)
        self.used = [False] * ring.nbufs
        for k in range(min(ring.nbufs, len(self.batches))):
            self._dma(k)

    def _dma(self, k: int):
        ring = self.ring
        b = k % ring.nbufs
        base = b * ring.per * ring.stride
        with torch.cuda.stream(ring.copy):
            if self.used[b]:
                ring.copy.wait_event(ring.consumed[b])
            for j, e in enumerate(self.batches[k]):
                o = base + j * ring.stride
                self.buf[o: o + self.nbytes].copy_(self.host_view(e), non_blocking = True)
            ring.ready[b].record(ring.copy)
        self.used[b] = True

    def run(self) -> torch.Tensor:
        """Compute every batch on the current (compute) stream; returns the fp32 partial
        (rows x h)"""
        from .moe_cpu_host import run_staged_experts, staged_recon_inputs
        hl, m, ring, y = self.hl, self.module, self.ring, self.y
        rows = y.shape[0]
        compute = torch.cuda.current_stream(y.device)
        out_ext = torch.zeros((rows + 1, hl.h), dtype = torch.float, device = y.device)
        mg = m.multi_gate if m.gated else m.multi_up
        mu, md = m.multi_up, m.multi_down
        sc = getattr(hl, "stream_scales", None)
        if sc is None:
            # Host copy of the per-expert suh / svh addresses (g, u, d; the up tables stand in
            # for the gate when gateless), as the fused kernel's pointer tables take them
            sc = hl.stream_scales = np.array(
                [[getattr(l.inner, t).data_ptr() for l in ml.linears]
                 for ml in (mg, mu, md) for t in ("suh", "svh")], dtype = np.int64)
        fused_t, fbufs, recon = self.tiers
        g_off, u_off, d_off = hl.proj_off
        ctx = {}
        counts, offs = self.counts, self.offs
        nb = len(self.batches)
        for k, batch in enumerate(self.batches):
            b = k % ring.nbufs
            compute.wait_event(ring.ready[b])
            base = self.buf.data_ptr() + b * ring.per * ring.stride
            single = run_staged_experts(
                y, out_ext, [(j, e, int(offs[e]), int(counts[e])) for j, e in enumerate(batch)],
                self.token_sorted, self.weight_sorted, base, ring.stride, hl.proj_off, m.gated,
                lambda e: tuple(int(v) for v in sc[:, e]), (mg.K, mu.K, md.K),
                m.activation_fn_idx, m.act_limit, (mg.mcg, mg.mul1, mu.mcg, mu.mul1, md.mcg, md.mul1),
                fused_t, fbufs, m.mtile_ok, recon, ctx)
            if single:
                # Above the batched tier's row cap: single-expert groups of at most max_rows
                # rows each, as the GPU tiers' leftover path (BlockSparseMLP._run_hybrid_heavy)
                y_ext, tok_ext, w_ext = staged_recon_inputs(y, self.token_sorted, self.weight_sorted,
                                                            recon.dims_u[0], ctx)
                step = max(1, recon.max_rows)
                for j, e in single:
                    bb = base + j * ring.stride
                    ptrs = ([bb + g_off] if m.gated else None, [bb + u_off], [bb + d_off])
                    c, s0 = int(counts[e]), int(offs[e])
                    for off in range(0, c, step):
                        recon.run_group(y_ext, out_ext, tok_ext, w_ext, [e], [s0 + off],
                                        [min(step, c - off)], ptrs = ptrs)
            ring.consumed[b].record(compute)
            if k + ring.nbufs < nb:
                self._dma(k + ring.nbufs)
        return out_ext[:rows]


def mixed_rate_gpu_mask(widths: list[list[int]]) -> tuple[np.ndarray | None, tuple]:
    """Mixed-rate ("MixedK") layers: widths holds, per projection (gate, up, down order), each
    expert's trellis tile width. The GPU tiers take one K per projection, so only the experts of
    the layer's most common rate signature get GPU lanes (cache slots, zero-copy homes); the rest
    always run on their CPU workers, whose kernels take any K per expert. Returns (mask of
    GPU-eligible experts, or None when every expert shares one signature; that signature's
    widths per projection). Ties go to the signature seen first, so the choice is deterministic"""
    sigs = list(zip(*widths))
    counts = {}
    for s in sigs:
        counts[s] = counts.get(s, 0) + 1
    sig = max(counts, key = counts.get)
    if len(counts) == 1:
        return None, sig
    return np.array([s == sig for s in sigs], dtype = bool), sig


class Telemetry:
    """Optional (EXL3_HYBRID_MOE_PROFILE=1) GPU-side wait profile of the CPU lanes: per node,
    the time the compute stream spends from just before waiting for the worker's done flag to
    just after folding its partial in (decode), or waiting for the node's side stream (prefill).
    Event pairs are recorded on the stream and harvested later with Event.query(), never
    synchronized; at most `cap` pairs are pending, extra calls are counted as unsampled"""

    def __init__(self, num_nodes: int, cap: int = 512):
        self.cap = cap
        self.pending = []                 # (node, kind, e0, e1)
        self.wait_us = [0.0] * num_nodes
        self.waits = [0] * num_nodes
        self.max_us = [0.0] * num_nodes
        self.unsampled = 0

    def begin(self, stream):
        if len(self.pending) >= self.cap:
            self.unsampled += 1
            return None
        e0 = torch.cuda.Event(enable_timing = True)
        e0.record(stream)
        return e0

    def end(self, node: int, kind: str, e0, stream):
        if e0 is None:
            return
        e1 = torch.cuda.Event(enable_timing = True)
        e1.record(stream)
        self.pending.append((node, kind, e0, e1))

    def harvest(self):
        keep = []
        for node, kind, e0, e1 in self.pending:
            if not e1.query():
                keep.append((node, kind, e0, e1))
                continue
            us = e0.elapsed_time(e1) * 1000.0
            self.wait_us[node] += us
            self.waits[node] += 1
            self.max_us[node] = max(self.max_us[node], us)
        self.pending = keep

    def summary(self) -> dict:
        self.harvest()
        return dict(
            cpu_wait_us_total = [round(v, 1) for v in self.wait_us],
            cpu_wait_us_mean = [round(v / n, 1) if n else None for v, n in zip(self.wait_us, self.waits)],
            cpu_wait_us_max = [round(v, 1) for v in self.max_us],
            cpu_wait_samples = list(self.waits),
            unsampled = self.unsampled,
        )


def hybrid_requested(config) -> bool:
    return bool(getattr(config.infer_params, "moe_hybrid", False)) and HT.dispatch != "legacy"


class HybridLayer:
    """Per-module hybrid state"""

    def __init__(self, dispatcher, module, device, dims, shards, host_idx, cpuloc):
        self.dispatcher = dispatcher
        self.module_ref = weakref.ref(module)
        self.key = module.key
        self.uid = f"{dispatcher.tag}:{module.key}"
        self.device = torch.device(device)
        self.num_experts = module.num_experts
        self.topk = module.num_experts_per_tok
        self.dims = dims                    # dict g/u/d: (k, n, K) ; g None when gateless
        # Mixed-rate layer: bool (E,) of the experts with GPU lanes, whose K the dims carry; the
        # others never get a home or a cache slot and always run on their CPU workers. None = all
        self.gpu = dims.get("gpu")
        self.h = module.expert_size
        self.hi = dims["u"][0]
        self.ho = dims["d"][1]
        pb = {p: (dims[p][0] // 16) * (dims[p][1] // 16) * int(16 * dims[p][2]) * 2 if dims.get(p) else 0
              for p in ("g", "u", "d")}
        self.proj_bytes = pb
        self.proj_off = (0, pb["g"], pb["g"] + pb["u"])
        self.expert_bytes = pb["g"] + pb["u"] + pb["d"]
        self.shards = shards
        self.host_idx = host_idx            # per host: worker layer index, -1 = not registered
        self.cpuloc = cpuloc                # (num_hosts, E) worker-local index or -1
        self.cache_layer = None
        self.finalized = False
        # Prefill stream lane: host copy of the experts' suh / svh addresses, built at first use
        self.stream_scales = None
        # Decode-size static buffers (rows up to the slot capacity): masked selection
        cap = max(dispatcher.cap_rows, 1) * self.topk
        self.sel_buf = torch.empty((cap,), dtype = torch.long, device = self.device)
        self.lane_buf = torch.empty((EMAX,), dtype = torch.int32, device = self.device)
        self._worst = {}
        # Expert deferral (EXL3_MOE_DEFER): set per pass by _link_layers; mHC fold buffers
        # (v twice, rows x H; d, rows x h) allocated at the layer's first deferral
        self.defer_next = False
        self.defer_v = None
        self.defer_d = None
        # Partial deferral: the issue kernel's per-node CPU selections (nodes x decode rows *
        # topk, device), split from there into the immediate and the deferred slot
        self.defer_sel = None


class HybridPending:
    __slots__ = ("kind", "items", "fused_cap", "deferred", "stream")

    def __init__(self, kind, items, fused_cap = None, deferred = None, stream = None):
        self.kind = kind
        self.items = items
        self.fused_cap = fused_cap
        # Partial deferral (decode): the second job per node holding the deferred picks, until
        # defer() takes it over; combine() collects it if the call does not defer after all
        self.deferred = deferred
        # Prefill stream lane (StreamWork), computed and folded in by combine()
        self.stream = stream


# Module (or transformer block) attributes after the routed sum that a deferred CPU partial would
# bypass: it is added to the residual later, unprojected, unnormed, unscaled and unreduced
DEFER_BLOCKERS = ("latent_out", "routed_post_norm", "tp_reduce", "mlp_resid_scalar", "mlp_post_norm",
                  "layer_scalar_f")


class DeferredPartial:
    """Decode CPU partial of one layer left out of its routed sum (EXL3_MOE_DEFER) and folded into
    the residual at the end of the next transformer block (HybridDispatcher.fold). The jobs stay
    in the dispatcher's inflight list until then, so an interrupted forward still drains them.
    With hyperconnections (mHC) the partial enters the streams as v ⊗ d: v starts as the layer's
    post weights and is carried through every later site's combine (carry), v <- combᵀ v"""

    __slots__ = ("items", "dispatcher", "hl", "device", "rows", "v", "vi")

    def __init__(self, items, dispatcher, hl, device, rows):
        self.items = items
        self.dispatcher = dispatcher
        self.hl = hl
        self.device = device
        self.rows = rows
        self.v = None
        self.vi = 0

    def set_post(self, post):
        """mHC: keep the layer's post weights (rows, H); the mix scratch they live in is reused by
        the next site. Copied into the layer's static buffer (allocated once)"""
        H = post.shape[-1]
        buf = getattr(self.hl, "defer_v", None)
        if buf is None or buf.shape[2] != H:
            cap = max(self.dispatcher.decode_rows, self.rows, 1)
            buf = self.hl.defer_v = torch.empty((2, cap, H), dtype = torch.float, device = post.device)
            self.hl.defer_d = torch.empty((cap, self.hl.ho), dtype = torch.float, device = post.device)
        self.vi = 0
        self.v = buf[0, :self.rows]
        self.v.copy_(post.reshape(self.rows, H))

    def carry(self, comb):
        """mHC: a site applied x <- post ⊗ y + combᵀ x, so the pending v ⊗ d becomes (combᵀ v) ⊗ d"""
        R, H = self.v.shape
        self.vi ^= 1
        nv = self.hl.defer_v[self.vi, :R]
        torch.bmm(self.v.view(R, 1, H), comb.reshape(R, H, H), out = nv.view(R, 1, H))
        self.v = nv


class HybridDispatcher:

    @staticmethod
    def for_module(module, device) -> "HybridDispatcher | None":
        config = module.config
        if not hybrid_requested(config):
            return None
        comp = getattr(config.infer_params, "moe_cpu_component", "text")
        reg = getattr(config, "moe_hybrid_dispatchers", None)
        if reg is None:
            reg = {}
            config.moe_hybrid_dispatchers = reg
        d = reg.get(comp)
        if d is None:
            d = HybridDispatcher(config, comp, device)
            reg[comp] = d
        return d

    def __init__(self, config, comp: str, device):
        from .moe_numa_hosts import HostGroup
        from .moe_cpu_host import TUNING as HOST_TUNING
        self.config = config
        self.comp = comp
        self.tag = None                     # acquired with the first layer (acquire_tag)
        self.device0 = torch.device(device)
        node_policy = resolve_node_policy(HT.node_policy)
        self.node_policy = NODE_POLICIES.get(node_policy, 0)
        telemetry = node_policy == "numa_adaptive" or (HT.dispatch == "hybrid" and HT.zc_max is None)
        self.group = HostGroup(config, comp, self.device0.index, home = HT.home, telemetry = telemetry,
                               max_nodes = MAX_NODES)
        if self.group.numa and HT.home == "native_copy":
            # One native copy is enough: with replicated worker arenas only one host writes it
            from .moe_numa_hosts import memory_policy
            if memory_policy() == "replicate":
                keep = self.group.pref if self.group.pref >= 0 else 0
                for i, h in enumerate(self.group.hosts):
                    h.home = i == keep
        self.cap_rows = HOST_TUNING.cap_rows
        self.prefill_rows = HOST_TUNING.stream_min_rows
        self.decode_rows = decode_rows(self.cap_rows)
        self.layers: list[HybridLayer] = []
        self.notices = set()
        self._init_runtime(self.group.num_nodes)

    def _init_runtime(self, n: int):
        """Per-load state: measurements, telemetry, side streams. Reset when the last layer
        detaches (the workers shut down with it; a reload starts a new worker generation)"""
        self.devices: set = set()
        self.node_cost = torch.ones((n,), dtype = torch.float)
        self.cpu_us = [None] * n            # EMA of measured CPU microseconds per expert of a job
        self.tel_seen = [dict() for _ in range(n)]   # per host: slot -> last seq counted
        self.zc_us = None                   # measured microseconds to read one expert over PCIe
        self.hit_us = None                  # measured microseconds to read one expert from VRAM
        self.cpu_fixed_us = None            # measured CPU job round trip (empty job), worst node
        self.cpu_frac = HT.cpu_frac_init
        # Cost model handed to the issue kernel (HY_LC_*); zeros until ZC and CPU are measured
        self.lane_cost = torch.zeros((4,), dtype = torch.float)
        self.telemetry = Telemetry(n) if HT.profile else None
        self.cost_stale = 0                 # passes since the last decode-shaped CPU sample
        self.decode_tuner = DecodeThreadTuner(HT.decode_threads)
        self.prefill_fit = [PrefillCostFit() for _ in range(n)]
        self.marginal_fit = [MarginalCostFit() for _ in range(n)]
        self.prefill_cost = None            # (a, b) us, slowest node, from the fits
        self.prefill_stale = 0              # prefill calls without a CPU lane in a row
        self.prefill_gpu_min_last = None    # threshold of the last prefill call
        self.explore = False                # this pass uses the bootstrap share (re-measure)
        self.side = {}                      # (host index, device index) -> stream
        self.stream_rings = {}              # (device index, expert stride) -> StreamRing
        self.prefill_stream_last = None     # (experts, MiB) on the stream lane, last prefill call
        # Decode jobs reserved but not yet collected: [host, reservation, published]. Normally
        # empty between layer calls; anything left (an exception or interrupt between submit and
        # combine) is drained at the next pass start, or the slots' reuse would wait forever
        self.inflight = []
        self.finalized = False

    # ---- eligibility and load --------------------------------------------------------------

    def eligible(self, module) -> tuple[bool, str, dict | None]:
        """Header-only checks: the layer must run every tier of the GPU path from pointer tables
        (fused + batched reconstruct, no per-expert dequant objects) and every CPU kernel"""
        from ..modules.block_sparse_mlp_cpu import trellis_rate
        from ..modules.block_sparse_mlp import BATCH_RECON
        ip = module.config.infer_params
        if ip.no_reconstruct:
            return False, "no_reconstruct", None
        if not BATCH_RECON:
            return False, "EXL3_MOE_BATCH_RECON=0 (the hybrid prefill needs the batched tier)", None
        if module.num_local_experts != module.num_experts or module.routing_first is not None \
                or module.routing_device is not None:
            return False, "expert-sliced (TP) layer", None
        if module.latent_in is not None:
            return False, "latent MoE", None
        if module.num_experts > EMAX:
            return False, f"more than {EMAX} experts", None
        if not (module.activation_fn in ("silu", "gelu") if module.gated else module.activation_fn == "relu2"):
            return False, f"activation {module.activation_fn}", None
        if len(self.group.hosts) > module.num_experts:
            return False, "fewer experts than worker nodes", None
        stc = module.config.stc
        projs = ([("g", module.gates)] if module.gated else []) + [("u", module.ups), ("d", module.downs)]
        # Tile width (16 * K) of every expert's trellis per projection. Mixed-rate checkpoints
        # (one K per expert) share the (k, n) dims; the CPU kernels take any supported K per
        # expert, the GPU tiers one K per projection (see mixed_rate_gpu_mask)
        widths = []
        dims = {"g": None}
        for name, ls in projs:
            kn = None
            w = []
            for l in ls:
                if getattr(l, "fkey", None) or not l.is_exl3_storage(l.key):
                    return False, "experts not stored as separate EXL3 tensors", None
                t = stc.list_tensors(l.key)
                if l.key + ".bias" in t:
                    return False, "per-expert biases", None
                if l.key + ".mul1" not in t:
                    return False, "experts are not mul1", None
                shp = list(t[l.key + ".trellis"]["shape"])
                if kn is None:
                    kn = shp[:2]
                elif shp[:2] != kn:
                    return False, "experts of one projection differ in shape", None
                if trellis_rate(shp[2]) is None:
                    return False, "unsupported bitrate", None
                w.append(shp[2])
            widths.append(w)
            dims[name] = (kn[0] * 16, kn[1] * 16, None)
        gpu, sig = mixed_rate_gpu_mask(widths)
        for (name, _), tw in zip(projs, sig):
            dims[name] = dims[name][:2] + (trellis_rate(tw),)
        dims["gpu"] = gpu
        h = module.expert_size
        if dims["u"][0] != h or dims["d"][1] != h:
            return False, "padded hidden width", None
        if dims["d"][0] != dims["u"][1] or (dims["g"] and dims["g"][:2] != dims["u"][:2]):
            return False, "inconsistent projection shapes", None
        return True, "", dims

    def load_module(self, module, device, **kwargs) -> bool:
        """Hybrid claim of a block-sparse MoE layer: register its experts with the worker
        group (sharded across nodes), load everything else on the GPU, build the routed Linear
        objects over the cache's bounce block (no expert bytes are read into VRAM). The caller
        runs load_local / load_routing afterwards and then attach()"""
        ok, why, dims = self.eligible(module)
        if not ok:
            if why not in self.notices:
                self.notices.add(why)
                print(f" -- hybrid MoE: {module.key} stays on the regular path ({why})")
            return False
        device = torch.device(device)
        stc = module.config.stc
        hosts = self.group.hosts
        shards = self.group.shard(module.num_experts)
        act = {"silu": 0, "gelu": 1, "relu2": 2, "swiglu_oai": 3}[module.activation_fn]
        host_idx = []
        cpuloc = np.full((len(hosts), module.num_experts), -1, dtype = np.int32)
        gpu = dims.get("gpu")
        for i, (h, sh) in enumerate(zip(hosts, shards)):
            if not sh:
                host_idx.append(-1)
                continue
            idx = h.register_layer(
                module.key,
                [module.gates[e].key for e in sh] if module.gated else [],
                [module.ups[e].key for e in sh],
                [module.downs[e].key for e in sh],
                act, float(module.act_limit or 0.0),
                dims["u"][0], dims["d"][1], module.num_experts_per_tok,
                # The per-expert byte layout only holds for uniform-rate layers
                proj_dims = dict(g = dims["g"], u = dims["u"], d = dims["d"]) if gpu is None else None,
                aux = None,
                interm_fp32 = module.interm_dtype == torch.float,
                # CPU-only experts of a mixed-rate layer need no GPU-readable home copy
                home_mask = None if gpu is None else [bool(gpu[e]) for e in sh],
            )
            host_idx.append(idx)
            for j, e in enumerate(sh):
                cpuloc[i, e] = j

        if self.tag is None:
            self.tag = acquire_tag(self.comp)
        try:
            module.device = device
            experts = set(module.gates + module.ups + module.downs)
            for m in module.modules:
                if m not in experts:
                    m.load(device, **kwargs)

            hl = HybridLayer(self, module, device, dims, shards, host_idx, cpuloc)
            cache = ExpertCache.for_device(device)
            bounce = cache.bounce_for(hl.expert_bytes)
            projs = ([("g", module.gates)] if module.gated else []) + [("u", module.ups), ("d", module.downs)]
            offs = {"g": hl.proj_off[0], "u": hl.proj_off[1], "d": hl.proj_off[2]}
            for name, ls in projs:
                # CPU-only experts of a mixed-rate layer get Linear objects shaped like the GPU
                # experts: the uniform pointer tables need them, and no GPU lane reads them
                k, n, K = dims[name]
                shape = (k // 16, n // 16, int(16 * K))
                for e, l in enumerate(ls):
                    _bounce_linear(l, bounce, offs[name], hl.proj_bytes[name], device,
                                   shape = None if gpu is None or gpu[e] else shape)
        except BaseException:
            # Typically an OOM under the autosplit budget: module.hybrid is not set yet, so the
            # loader's unload cannot reach detach. Undo the worker registrations here, or the
            # retry on the next device counts the layer twice and the workers never shut down
            for h, idx in zip(hosts, host_idx):
                if idx >= 0:
                    h.unregister()
            if not self.layers:
                release_tag(self.tag)
                self.tag = None
            raise
        self.layers.append(hl)
        self.devices.add(device)
        module.hybrid = hl
        # Every tier addresses experts by router id through the tables; out-of-range ids (the
        # masked picks) are dropped by the range test of the fused decode kernels
        module.routing_first = 0
        module.routing_last = module.num_experts
        mixed = "" if gpu is None else \
            f", mixed rates: {int(gpu.sum())} at K {'/'.join(str(dims[p][2]) for p in ('g', 'u', 'd') if dims[p])} " \
            f"with GPU lanes, the rest CPU only"
        print(f" -- hybrid MoE experts: {module.key} ({module.num_experts} experts, "
              f"{hl.expert_bytes / 2**20:.2f} MiB each, {len(hosts)} worker node(s){mixed})")
        return True

    def attach(self, module):
        """After load_local: register the layer's pointer tables with the device's cache owner"""
        hl = module.hybrid
        cache = ExpertCache.for_device(hl.device)
        mu, md = module.multi_up, module.multi_down
        mg = module.multi_gate if module.gated else mu
        gproj = 0 if module.gated else 1
        tables = [(mg.ptrs_trellis, 0, 1, gproj), (mu.ptrs_trellis, 0, 1, 1), (md.ptrs_trellis, 0, 1, 2)]
        gu = getattr(module, "_gu_trellis_ptr", None)
        if gu is not None:
            # Stacked (gate, up) table handed to BC_BlockSparseMLP: not read by the current
            # kernels, kept consistent anyway
            tables += [(gu.view(-1), 0, 2, gproj), (gu.view(-1), 1, 2, 1)]
        if not module.gated:
            tables = tables[1:]
        hl.cache_layer = cache.register_layer(hl.uid, hl.key, hl.num_experts, hl.expert_bytes, hl.proj_off, tables)
        if self.finalized and self._hosts_ready(hl):
            self._finalize_layer(hl)

    def _hosts_ready(self, hl: HybridLayer) -> bool:
        """Every worker holding experts of this layer is started and has acked it (its homes and
        arena blocks are known). Otherwise the layer finalizes at the next begin_pass"""
        return all(idx < 0 or (h.started and h.acked > idx) for h, idx in zip(self.group.hosts, hl.host_idx))

    def detach(self, module):
        hl = getattr(module, "hybrid", None)
        if hl is None:
            return
        if hl.cache_layer is not None:
            ExpertCache.for_device(hl.device).unregister_layer(hl.cache_layer)
            hl.cache_layer = None
        for h, idx in zip(self.group.hosts, hl.host_idx):
            if idx >= 0:
                h.unregister()
        if hl in self.layers:
            self.layers.remove(hl)
        module.hybrid = None
        module.routing_first = None
        module.routing_last = None
        if not self.layers:
            # Last layer gone: the workers have shut down (homes unmapped). A reload starts a new
            # worker generation, so per-load state must not survive: finalize again from scratch
            self._init_runtime(self.group.num_nodes)
            release_tag(self.tag)
            self.tag = None

    # ---- finalize ---------------------------------------------------------------------------

    def finalize(self):
        """Workers started, homes mapped: hand every layer's homes and CPU locations to the
        cache, probe the PCIe read rate. Idempotent; layers attached later finalize on attach"""
        if self.finalized:
            return
        for h in self.group.hosts:
            h.ensure_started()
        self.group.after_start()
        self.finalized = True
        # The autosplit loaders' measured transients reach the cache floor directly
        # (moe_expert_cache.note_device_transient, max-merged per device)
        for hl in self.layers:
            if hl.cache_layer is not None:
                self._finalize_layer(hl)
        if HT.dispatch == "hybrid" and HT.zc_max is None:
            self._probe_pcie()
            self._probe_cpu_roundtrip()

    def _finalize_layer(self, hl: HybridLayer):
        from ..ext import exllamav3_ext as ext
        from .moe_cpu_host import TUNING as HOST_TUNING
        if hl.finalized:
            return
        E = hl.num_experts
        hosts = self.group.hosts
        native_ok = True
        if HT.home == "worker_arena":
            # The worker arena is GPU-readable only in native tile order
            for p in ("g", "u", "d"):
                if hl.dims.get(p) and HOST_TUNING.swizzle and ext.exl3_moe_cpu_swizzle_group(hl.dims[p][2]):
                    native_ok = False
            if not native_ok and "swz" not in self.notices:
                self.notices.add("swz")
                print(" -- hybrid MoE: EXL3_EXPERT_HOME=worker_arena with the packed CPU layout "
                      "(EXL3_MOE_CPU_SWIZZLE=1): no GPU-readable homes, non-resident experts take "
                      "CPU lanes (set EXL3_MOE_CPU_SWIZZLE=0 or EXL3_EXPERT_HOME=native_copy)")
        homes = np.zeros(E, dtype = np.int64)
        src = [None] * E
        if native_ok:
            for i, (h, sh, idx) in enumerate(zip(hosts, hl.shards, hl.host_idx)):
                if idx < 0 or not (h.home or h.map_arena):
                    continue
                for j, e in enumerate(sh):
                    if src[e] is not None or (hl.gpu is not None and not hl.gpu[e]):
                        continue
                    homes[e] = h.home_device_ptr(idx, j)
                    src[e] = (h, idx, j)
        missing = [e for e in range(E) if not homes[e] and (hl.cpuloc[:, e] < 0).all()]
        if missing:
            raise RuntimeError(f"hybrid MoE {hl.key}: experts {missing[:8]} have neither a GPU-readable home "
                               f"nor a CPU worker copy")
        # Unmapped GPU-eligible experts (CPU-only experts of a mixed-rate layer have no home by design)
        n_nozc = int(((homes == 0) & (hl.gpu if hl.gpu is not None else True)).sum())
        if n_nozc and native_ok and "nozc" not in self.notices:
            self.notices.add("nozc")
            print(f" -- hybrid MoE: {n_nozc} experts of {hl.key} have no GPU-readable home "
                  f"(registration failed or limited); they run on CPU lanes unless installed in the cache")
        nbytes = hl.expert_bytes

        def host_view(e, src = src, nbytes = nbytes):
            s = src[e]
            return None if s is None else s[0].home_host_view(s[1], s[2], nbytes)

        cache = ExpertCache.for_device(hl.device)
        cache.set_layer_cpuloc(hl.cache_layer, hl.cpuloc)
        cache.set_layer_homes(hl.cache_layer, homes, host_view if native_ok else None)
        hl.finalized = True

    def _probe_pcie(self):
        """Measured read rate of a registered home into VRAM (the cost of one zero-copy expert
        read is about one expert's bytes at this rate)"""
        h = next((h for h in self.group.hosts if any(h.home_dev) or any(getattr(h, "arena_dev", []))), None)
        if h is None or not self.layers:
            return
        views = h.home_views if any(h.home_dev) else h.arena_views
        devs = h.home_dev if any(h.home_dev) else h.arena_dev
        ci = next(i for i, d in enumerate(devs) if d)
        src = views[ci].view(torch.uint8)
        nb = min(src.numel(), 64 << 20)
        dev = self.layers[0].device
        with torch.cuda.device(dev):
            dst = torch.empty((nb,), dtype = torch.uint8, device = dev)
            e0, e1 = torch.cuda.Event(enable_timing = True), torch.cuda.Event(enable_timing = True)
            for _ in range(3):
                e0.record()
                dst.copy_(src[:nb], non_blocking = True)
                e1.record()
                e1.synchronize()
            ms = e0.elapsed_time(e1)
            # VRAM read rate (a device copy reads and writes each byte once): the decode cost of
            # a resident expert is about one read of its bytes
            src_d = torch.empty((nb,), dtype = torch.uint8, device = dev)
            for _ in range(3):
                e0.record()
                dst.copy_(src_d)
                e1.record()
                e1.synchronize()
            ms_d = e0.elapsed_time(e1)
        gbs = nb / max(ms, 1e-3) / 1e6
        eb = self.layers[0].expert_bytes
        self.zc_us = eb / (gbs * 1e3)
        gbs_d = 2 * nb / max(ms_d, 1e-3) / 1e6
        self.hit_us = eb / (gbs_d * 1e3)
        if HT.debug:
            print(f" -- hybrid MoE: host->device {gbs:.1f} GB/s, {self.zc_us:.0f} us per expert read; "
                  f"VRAM {gbs_d:.0f} GB/s, {self.hit_us:.1f} us per resident expert")

    def _probe_cpu_roundtrip(self):
        """Measured fixed cost of using a CPU node in a decode call: one empty gated job (all
        selections -1, so the worker does no work) from publish to the GPU seeing it done.
        Worst node; median of the timed repeats"""
        if not self.layers:
            return
        hl = self.layers[0]
        dev = hl.device
        worst = None
        with torch.cuda.device(dev):
            out = torch.zeros((1, hl.h), dtype = torch.float, device = dev)
            for h, idx in zip(self.group.hosts, hl.host_idx):
                if idx < 0 or not h.started:
                    continue
                samples = []
                for _ in range(6):
                    r = h.fused_reserve(idx, 1, dev)
                    r["slot"]["sel"].view(-1)[:hl.topk].fill_(-1)
                    r["counts"][r["slot_idx"]].zero_()
                    e0, e1 = torch.cuda.Event(enable_timing = True), torch.cuda.Event(enable_timing = True)
                    e0.record()
                    h.fused_publish(r)
                    h.fused_collect(r, out)
                    e1.record()
                    e1.synchronize()
                    samples.append(e0.elapsed_time(e1) * 1000.0)
                us = float(np.median(samples[1:]))
                worst = us if worst is None else max(worst, us)
        self.cpu_fixed_us = worst
        if HT.debug and worst is not None:
            print(f" -- hybrid MoE: CPU job round trip {worst:.0f} us (worst node)")

    # ---- passes -------------------------------------------------------------------------------

    def begin_pass(self, x):
        """Forward start: lazy finalize, cache sizing / headroom, telemetry and lane parameters"""
        if not self.layers:
            return
        if self.inflight:
            self._drain_inflight()
        if not self.finalized:
            self.finalize()
        for hl in self.layers:
            if not hl.finalized and hl.cache_layer is not None:
                for h, idx in zip(self.group.hosts, hl.host_idx):
                    if idx >= 0:
                        h.ensure_started()
                self._finalize_layer(hl)
        if x is not None and torch.is_tensor(x):
            rows = x.numel() if x.dtype in (torch.long, torch.int) else x.numel() // max(1, x.shape[-1])
        else:
            rows = 1
        for dev in self.devices:
            cache = ExpertCache.for_device(dev)
            stream = torch.cuda.current_stream(dev) if dev.type == "cuda" else None
            cache.begin_pass(rows, self.decode_rows + 1, stream)
            if rows > self.decode_rows:
                extra = max((self.worst_case(hl, rows) for hl in self.layers if hl.device == dev), default = 0)
                cache.ensure_headroom(extra)
        if HT.defer:
            self._link_layers()
        self._update_lane_params()
        frac = self.decode_tuner.on_pass(rows <= self.decode_rows, time.perf_counter())
        if frac is not None:
            for h in self.group.hosts:
                n = int(getattr(h, "threads", 0) or 0)
                setter = getattr(h, "set_thread_caps", None)
                if setter is not None:
                    setter(0, 0 if frac >= 1.0 or n <= 0 else max(1, round(frac * n)))
        if self.telemetry is not None:
            self.telemetry.harvest()

    def summary(self) -> dict:
        """Telemetry snapshot (reads device counters: call it between passes, not per token).
        Cache counters are per device and shared by every component on it"""
        out = dict(
            component = self.tag, layers = len(self.layers), nodes = self.group.num_nodes,
            lane_cost_us = dict(hit = float(self.lane_cost[LC_HIT]), zero_copy = float(self.lane_cost[LC_ZC]),
                                cpu_fixed = float(self.lane_cost[LC_CPU_FIXED]), cpu = float(self.lane_cost[LC_CPU])),
            cost_model = bool(self.lane_cost[LC_ZC] > 0 and self.lane_cost[LC_CPU] > 0),
            prefill_cost_us = None if self.prefill_cost is None else
                dict(per_expert = round(self.prefill_cost[0], 2), per_row = round(self.prefill_cost[1], 3)),
            prefill_gpu_min = self.prefill_gpu_min_last,
            prefill_stream = None if getattr(self, "prefill_stream_last", None) is None else
                dict(experts = self.prefill_stream_last[0], mib = self.prefill_stream_last[1]),
            cpu_frac_bootstrap = self.cpu_frac,
            cpu_us_per_expert = list(self.cpu_us),
            decode_threads = dict(fraction = self.decode_tuner.current(),
                                  token_ms = {str(k): round(v * 1e3, 3) for k, v in self.decode_tuner.results.items()}),
            caches = [ExpertCache.for_device(d).summary() for d in sorted(self.devices, key = str)],
        )
        if self.telemetry is not None:
            out["profile"] = self.telemetry.summary()
        return out

    def _update_lane_params(self):
        """Node costs and the CPU share of misses from measurements (telemetry, PCIe probe)"""
        changed = False
        fit_changed = False
        for i, h in enumerate(self.group.hosts):
            for slot, (seq, us, na, ne, rows) in enumerate(h.read_telemetry()):
                if not na or self.tel_seen[i].get(slot) == seq:
                    continue
                self.tel_seen[i][slot] = seq
                self.prefill_fit[i].add(ne, na, us)
                fit_changed = True
                if rows > COST_SAMPLE_MAX_ROWS:
                    continue
                # Per distinct expert: a job streams each selected expert's weights once, so that
                # is the unit the cost model compares with one zero-copy expert read
                v = us / max(ne, 1)
                self.marginal_fit[i].add(ne, us)
                self.cpu_us[i] = v if self.cpu_us[i] is None else (1 - HT.ema) * self.cpu_us[i] + HT.ema * v
                changed = True
        if fit_changed:
            fits = [f.solve() for f in self.prefill_fit]
            fits = [f for f in fits if f is not None]
            # Slowest node per term, as for decode: a CPU lane is never assumed faster than it is
            self.prefill_cost = (max(f[0] for f in fits), max(f[1] for f in fits)) if fits else None
        # Re-measure: once the cost model keeps every flexible miss on the GPU, decode yields no
        # new CPU samples and a stale (e.g. pessimistic) estimate would persist. After
        # EXL3_MOE_COST_REMEASURE passes without a sample, one pass uses the bootstrap share
        self.explore = False
        if changed:
            self.cost_stale = 0
        elif float(self.lane_cost[LC_CPU]) > 0:
            self.cost_stale += 1
            if HT.remeasure and self.cost_stale >= HT.remeasure:
                self.cost_stale = 0
                self.explore = True
        if not changed:
            return
        known = [c for c in self.cpu_us if c]
        if known:
            # Node choice compares marginal per-expert costs (fixed job cost excluded) once every
            # node's fit separates them; until then the plain per-expert average
            marg = [f.marginal() for f in self.marginal_fit]
            rel = marg if all(m is not None for m in marg) else self.cpu_us
            have = [c for c in rel if c]
            mean = sum(have) / len(have)
            for i, c in enumerate(rel):
                self.node_cost[i] = (c or mean) / mean
        if HT.zc_max is None and self.zc_us and known:
            # Measured cost model for the issue kernel (greedy min-max per call). Per expert CPU
            # cost: the slowest measured node, so a CPU lane is never assumed faster than it is
            self.lane_cost[LC_HIT] = float(self.hit_us or 0.0)
            self.lane_cost[LC_ZC] = float(self.zc_us)
            self.lane_cost[LC_CPU_FIXED] = float(self.cpu_fixed_us or 0.0)
            self.lane_cost[LC_CPU] = float(max(known))

    def worst_case(self, hl: HybridLayer, rows: int) -> int:
        """Upper bound on a hybrid layer's prefill transient for `rows` tokens: the GPU tiers'
        bound (fused, batched groups; the heavy per-expert fallback is bounded by one batched
        group of max_rows rows plus its fp32 down slab), the CPU lanes' staging (per node: the
        gathered rows, selections and the fp32 partial), the selection buffers and the stream
        lane (its VRAM ring and partial), so the cache gives the ring back before such a pass"""
        shape = stream_ring_shape(hl.expert_bytes) if prefill_stream_on(rows, hl.device.type == "cuda") else None
        w = hl._worst.get((rows, shape))
        if w is not None:
            return w
        m = hl.module_ref()
        if m is None:
            return 0
        A = rows * hl.topk
        gpu = sum(m.prefill_worst_case_parts(rows, A))
        recon = m.batch_recon if m.batch_recon not in (None, False) else None
        # One expert takes at most `rows` assignments per call (max_rows can be 2^30 with
        # EXL3_MOE_RECON_TILES=0)
        heavy_rows = min(recon.max_rows, rows) if recon is not None else 0
        gpu += heavy_rows * hl.ho * 4
        n = len(self.group.hosts)
        cpu = n * (rows * hl.ho * 4 + rows * hl.hi * 2 + rows * hl.topk * 12 + self.cap_rows * hl.ho * 4)
        sel = A * (8 + 4 * n) + EMAX * 4
        # Stream lane: the ring, its fp32 partial (with the padding sink row) and the sorted
        # assignment lists. Its batched groups run after the GPU tiers' and fit their bound
        stream = 0
        if shape is not None:
            nbufs, per, stride = shape
            stream = nbufs * per * stride + (rows + 1) * hl.h * 4 + A * 32 + EMAX * 8
        w = gpu + cpu + sel + stream
        hl._worst[(rows, shape)] = w
        return w

    # ---- expert deferral (EXL3_MOE_DEFER) -------------------------------------------------------

    def _link_layers(self):
        """Per layer: may it defer its CPU partial? Not the last hybrid layer, the next one on the
        same device, and at least two CPU slots per worker: the deferred job holds its slot
        unconsumed while the next layer reserves, and with one slot that reserve would wait on
        it forever. Partial deferral needs three (can_defer): a layer then takes two slots per
        worker (immediate, deferred), and between the deferred job's reserve and its fold at the
        end of the next block only that block's two reserves come, which must both land on
        other slots than the unconsumed one"""
        from .moe_cpu_host import TUNING as HOST_TUNING
        self.defer_slots = min((getattr(h, "num_slots", HOST_TUNING.num_slots) for h in self.group.hosts), default = 0)
        slots = self.defer_slots >= 2
        n = len(self.layers)
        for i, hl in enumerate(self.layers):
            hl.defer_next = slots and i + 1 < n and self.layers[i + 1].device == hl.device

    def can_defer(self, hl: HybridLayer, module, rows: int) -> bool:
        """This decode call's CPU partial may skip combine and fold into the residual later"""
        if not HT.defer or rows > self.decode_rows or not getattr(hl, "defer_next", False):
            return False
        if HT.defer_partial() and getattr(self, "defer_slots", 0) < 3:
            if "defer_slots" not in self.notices:
                self.notices.add("defer_slots")
                print(" !! hybrid MoE: EXL3_MOE_DEFER_MAX / EXL3_MOE_DEFER_MIN_W need EXL3_MOE_CPU_SLOTS >= 3, "
                      "not deferring")
            return False
        for a in DEFER_BLOCKERS:
            v = getattr(module, a, None)
            if v is not None and v is not False:
                return False
        return True

    def defer(self, hl: HybridLayer, pending: HybridPending, rows: int) -> DeferredPartial:
        """Take the deferred jobs out of pending (all of them, or with partial deferral the
        second job per node); they stay inflight until fold. combine() still collects what is
        left (the immediate jobs of a partial deferral)"""
        if pending.deferred is not None:
            items, pending.deferred = pending.deferred, None
        else:
            items, pending.items = pending.items, []
        return DeferredPartial(items, self, hl, hl.device, rows)

    def _reserve_deferred(self, hl: HybridLayer, rows: int, res: list) -> list:
        """Partial deferral: a second slot per node for the deferred picks, reserved after the
        immediate one (the worker runs jobs in ring order: immediate first)"""
        dres = []
        for i, h, r in res:
            rd = h.fused_reserve(hl.host_idx[i], rows, hl.device)
            dres.append((i, h, rd))
            self.inflight.append([h, rd, False])
        return dres

    def _defer_sel(self, hl: HybridLayer):
        """Device scratch the issue kernel writes the nodes' CPU selections into (instead of the
        immediate slot); rows of nodes without experts of the layer are never written, stay -1"""
        if getattr(hl, "defer_sel", None) is None:
            hl.defer_sel = torch.full((len(self.group.hosts), self.decode_rows * hl.topk), -1,
                                      dtype = torch.int32, device = hl.device)
        return hl.defer_sel

    def _split_deferred(self, hl: HybridLayer, y, w, rows: int, res: list, dres: list):
        """Partial deferral, stream-ordered after the issue kernel: per token, the CPU-lane picks
        with the smallest routing weights (at most HT.defer_max, all for 0; only those below
        HT.defer_min_w) go to the deferred slot, the others to the immediate one. The kernel
        staged x, w and the gate into the immediate slot already; the deferred slot gets them
        here, and both gates follow their selections. No host sync"""
        k = hl.topk
        n = rows * k
        s = hl.defer_sel[:, :n]
        cpu = (s >= 0).any(dim = 0).view(rows, k)
        wf = w.view(rows, k).float()
        dm = cpu
        if HT.defer_min_w is not None:
            dm = dm & (wf < HT.defer_min_w)
        cap = HT.defer_max
        if 0 < cap < k:
            # Rank of each pick among its token's deferrable picks by weight (stable: ties keep
            # router order); picks that may not defer sort last
            order = wf.masked_fill(~dm, float("inf")).argsort(dim = 1, stable = True)
            ar = torch.arange(k, device = w.device).expand(rows, k)
            rank = torch.empty_like(order).scatter_(1, order, ar)
            dm = dm & (rank < cap)
        dm = dm.view(-1)
        hi = hl.hi
        xd = y if y.shape[1] == hi else torch.nn.functional.pad(y, (0, hi - y.shape[1]))
        wd = w.view(-1).to(torch.half)
        for (i, h, ra), (_, _, rd) in zip(res, dres):
            si = s[i]
            sa = si.masked_fill(dm, -1)
            sd = si.masked_fill(~dm, -1)
            ra["slot"]["sel"].view(-1)[:n].copy_(sa, non_blocking = True)
            rd["slot"]["sel"].view(-1)[:n].copy_(sd, non_blocking = True)
            rd["slot"]["w"].view(-1)[:n].copy_(wd, non_blocking = True)
            rd["slot"]["x"].view(-1)[:rows * hi].copy_(xd.reshape(-1), non_blocking = True)
            ra["counts"][ra["slot_idx"]] = (sa >= 0).any()
            rd["counts"][rd["slot_idx"]] = (sd >= 0).any()

    def fold(self, rec: DeferredPartial, x2d, v = None):
        """Add a deferred CPU partial into the residual, stream-ordered. Plain residual (fp32,
        rows x h): collected straight into it. mHC (v given, (rows, H)): collected into the
        layer's static fp32 buffer, then x[h] += v[h] * d"""
        cuda = rec.device.type == "cuda"
        tel = self.telemetry if cuda else None
        compute = torch.cuda.current_stream(rec.device) if cuda else None
        if v is None:
            out = x2d.view(-1, x2d.shape[-1])
        else:
            out = rec.hl.defer_d[:rec.rows]
            out.zero_()
        for i, h, r in rec.items:
            e0 = tel.begin(compute) if tel is not None else None
            h.fused_collect(r, out)
            if tel is not None:
                tel.end(i, "defer", e0, compute)
            self.inflight = [x for x in self.inflight if x[1] is not r]
        if v is not None:
            R, H = v.shape
            xs = x2d.view(R, H, -1)
            if cuda:
                from ..ext import exllamav3_ext as ext
                ext.hc_apply(xs, out, v, None, None, None)
            else:
                xs.add_(v.unsqueeze(-1) * out.unsqueeze(-2))

    # ---- per layer call --------------------------------------------------------------------------

    def _drain_inflight(self):
        """Complete decode jobs an interrupted forward left behind, as empty jobs: selections set
        to -1 by a stream-ordered copy into the slot (after the reserve's wait for the previous
        tenant, so no in-flight job is disturbed), the slot gate cleared, published, collected
        into a scratch row. The worker sees an all-inactive job and skips it"""
        n = len(self.inflight)
        for h, r, published in self.inflight:
            dev = torch.device(r["dev"])
            with (torch.cuda.device(dev) if dev.type == "cuda" else contextlib.nullcontext()):
                if not published:
                    sel = r["slot"]["sel"].view(-1)
                    sel.copy_(torch.full((sel.numel(),), -1, dtype = torch.int32, device = dev), non_blocking = True)
                    r["counts"][r["slot_idx"]].zero_()
                    h.fused_publish(r)
                scratch = torch.zeros((r["rows"], r["ho"]), dtype = torch.float, device = dev)
                h.fused_collect(r, scratch)
        self.inflight = []
        print(f" !! hybrid MoE: drained {n} CPU job(s) left by an interrupted forward pass")

    def _nodes_decode(self, hl, rows):
        res = []
        nodes = torch.zeros((ND_ROWS, len(self.group.hosts)), dtype = torch.long)
        for i, (h, idx) in enumerate(zip(self.group.hosts, hl.host_idx)):
            if idx < 0:
                # No experts of this layer on that node (the kernel never gives it a lane:
                # cpuloc is -1 throughout); its column stays zero
                continue
            r = h.fused_reserve(idx, rows, hl.device)
            res.append((i, h, r))
            self.inflight.append([h, r, False])
            slot = r["slot"]
            nodes[ND_SEL, i] = slot["sel_dev"]
            nodes[ND_X, i] = slot["x_dev"]
            nodes[ND_W, i] = slot["w_dev"]
            nodes[ND_COUNT, i] = r["counts"].data_ptr()
            nodes[ND_SLOT, i] = r["slot_idx"]
            nodes[ND_XCAP, i] = slot["x_cap"]
            nodes[ND_SELCAP, i] = slot["sel_cap"]
        return res, nodes

    def _lane_args(self, n_picks: int):
        gpu_exact = HT.dispatch == "gpu_exact"
        gpu_max = -1 if (gpu_exact or HT.zc_max is None) else HT.zc_max
        cpu_frac = 0.0 if gpu_exact else self.cpu_frac
        lane_cost = None if (gpu_exact or HT.zc_max is not None or self.explore) else self.lane_cost
        admit = int(n_picks <= CACHE_TUNING.admit_max)
        return gpu_max, cpu_frac, admit, lane_cost

    def submit(self, hl: HybridLayer, y, selected_experts, routing_weights, defer: bool = False):
        """Lane assignment and CPU job issue for one layer call. Returns (pending, sel_gpu): the
        GPU tiers must run on sel_gpu (CPU-lane picks masked to num_experts). defer: the caller
        will defer this call's CPU partial (can_defer); with partial deferral (HT.defer_max /
        HT.defer_min_w) each node's CPU work is then split into an immediate and a deferred job"""
        rows = y.shape[0]
        cache = ExpertCache.for_device(hl.device)
        y = y.contiguous()
        w = routing_weights.contiguous()
        sel = selected_experts.contiguous()
        n = rows * hl.topk
        if rows <= self.decode_rows:
            res, nodes = self._nodes_decode(hl, rows)
            dres = None
            if defer and res and HT.defer_partial():
                dres = self._reserve_deferred(hl, rows, res)
                ds = self._defer_sel(hl)
                for i, _, _ in res:
                    nodes[ND_SEL, i] = ds[i].data_ptr()
                    nodes[ND_SELCAP, i] = ds.shape[1]
            sel_gpu = hl.sel_buf[:n]
            gpu_max, cpu_frac, admit, lane_cost = self._lane_args(n)
            # The admission copy (gather) is enqueued after the publish: the workers only read
            # what issue staged (selections, activations, weights), so they start while the GPU
            # copies admitted experts; the GPU tiers still run after the copy on this stream
            owed = cache.issue(hl.cache_layer, sel, y, w, sel_gpu, None, None, nodes, self.node_cost, hl.hi,
                               0, admit, gpu_max, cpu_frac, PREFILL_GPU_MIN_BOOT, self.node_policy,
                               self.group.pref, lane_cost, gather = False)
            if dres is not None:
                self._split_deferred(hl, y, w, rows, res, dres)
            for _, h, r in res + (dres or []):
                h.fused_publish(r)
                for entry in self.inflight:
                    if entry[1] is r:
                        entry[2] = True
            if owed:
                cache.gather(sel_gpu)
            return HybridPending("decode", res, deferred = dres), sel_gpu.view(rows, hl.topk)
        return self._submit_prefill(hl, cache, y, sel, w, rows, n)

    def _submit_prefill(self, hl, cache, y, sel, w, rows, n):
        from ..util.device_copy import host_to_device
        hosts = self.group.hosts
        N = len(hosts)
        dev = hl.device
        sel_gpu = torch.empty((n,), dtype = torch.long, device = dev)
        cpu_sel = torch.empty((N, n), dtype = torch.int32, device = dev)
        nodes = torch.zeros((ND_ROWS, N), dtype = torch.long)
        gpu_min = self._prefill_gpu_min(hl, cache, sel)
        self.prefill_gpu_min_last = gpu_min
        cache.issue(hl.cache_layer, sel, y, w, sel_gpu, hl.lane_buf, cpu_sel, nodes, self.node_cost, hl.hi,
                    1, 0, -1, 0.0, gpu_min, self.node_policy, self.group.pref)
        E = hl.num_experts
        cuda = dev.type == "cuda"
        # Stream lane: the call's expert counts join the readback below
        stream_on = prefill_stream_on(rows, cuda) and hl.cache_layer.host_view is not None
        parts = [hl.lane_buf[:E], cpu_sel.view(-1)]
        if stream_on:
            s = sel.view(-1)
            s = torch.where((s >= 0) & (s < E), s, E)
            parts.append(torch.bincount(s, minlength = E + 1)[:E].to(torch.int32))
        # One host sync for the tier plan and the CPU row compression (the GPU tiers sync on
        # their own counts anyway)
        both = torch.cat(parts).cpu().numpy()
        lanes = both[:E]
        cs = both[E: E + N * n].reshape(N, rows, hl.topk)
        stream = self._stream_begin(hl, cache, y, sel, w, sel_gpu, lanes, both[E + N * n:]) if stream_on else None
        compute = torch.cuda.current_stream(dev) if cuda else None
        items = []
        w2 = w.view(rows, hl.topk)
        for i, (h, idx) in enumerate(zip(hosts, hl.host_idx)):
            if idx < 0:
                continue
            m = cs[i]
            tidx = np.nonzero((m >= 0).any(axis = 1))[0]
            if not tidx.size:
                continue
            tidx_d = host_to_device(torch.from_numpy(tidx.astype(np.int64)), dev)
            y_c = y.index_select(0, tidx_d)
            w_c = w2.index_select(0, tidx_d)
            sel_c = host_to_device(torch.from_numpy(np.ascontiguousarray(m[tidx]).astype(np.int64)), dev)
            out_t = torch.empty((tidx.size, hl.h), dtype = torch.float, device = dev)
            if not cuda:
                # Reference backend (tests): the node's job runs inline, no streams
                jobs, rtmp = h._issue_compute(idx, y_c, sel_c, w_c, h.specs[idx], out_t, hl.h)
                h._collect_compute(jobs, out_t, rtmp, hl.h)
                items.append((i, tidx_d, out_t, None))
                continue
            side = self.side.get((i, dev.index))
            if side is None:
                side = self.side[(i, dev.index)] = torch.cuda.Stream(device = dev)
                cache.note_stream(side)
            side.wait_stream(compute)
            with torch.cuda.stream(side):
                with torch.cuda.device(dev):
                    jobs, rtmp = h._issue_compute(idx, y_c, sel_c, w_c, h.specs[idx], out_t, hl.h)
                    h._collect_compute(jobs, out_t, rtmp, hl.h)
                done = torch.cuda.Event()
                done.record(side)
            # The compute stream reads these after waiting for `done`; record_stream keeps the
            # caching allocator from reusing them for the side stream's later work too early
            for t in (y_c, w_c, sel_c, out_t, tidx_d):
                t.record_stream(side)
            items.append((i, tidx_d, out_t, done))
        # Zero-copy experts read host memory once per row tile in the fused kernel; capping the
        # fused tier moves larger experts to the batched tier (one read). That changes the tier
        # of resident experts too, so gpu_exact keeps the regular plan (exact, more PCIe reads).
        # Stream-lane experts are read from VRAM and do not count
        zc = lanes == LANE_ZC
        if stream is not None:
            zc = zc.copy()
            zc[stream.ids] = False
        any_zc = bool(zc.any()) and HT.dispatch != "gpu_exact"
        return HybridPending("prefill", items, HT.zc_fused_rows if any_zc else None, stream = stream), \
            sel_gpu.view(rows, hl.topk)

    def _stream_begin(self, hl, cache, y, sel, w, sel_gpu, lanes, counts):
        """Prefill stream lane for this call (StreamWork), or None: zero-copy experts with at
        least EXL3_EXPERT_PREFILL_STREAM_MIN_ROWS rows leave the GPU tiers' selection and
        their weights start copying into the VRAM ring"""
        m = hl.module_ref()
        shape = stream_ring_shape(hl.expert_bytes)
        if m is None or shape is None:
            self.prefill_stream_last = None
            return None
        tiers = StreamWork.layer_tiers(m, y)
        fused_t, fbufs, recon = tiers
        if fbufs is None and recon is None:
            self.prefill_stream_last = None
            return None
        # Without a batched tier only experts the fused tier takes can stream
        ids = select_stream_experts(lanes, counts, HT.stream_min_rows,
                                    None if recon is not None else fused_t)
        self.prefill_stream_last = (int(ids.size), round(int(ids.size) * hl.expert_bytes / 2**20, 1))
        if not ids.size:
            return None
        dev = hl.device
        key = (dev.index, shape[2])
        rings = self.__dict__.setdefault("stream_rings", {})
        ring = rings.get(key)
        if ring is None or (ring.nbufs, ring.per) != shape[:2]:
            ring = rings[key] = StreamRing(dev, *shape)
            cache.note_stream(ring.copy)
        mask_streamed(sel_gpu, ids, hl.num_experts)
        return StreamWork(ring, hl, m, tiers, y, sel.view(-1), w, ids, counts)

    def _prefill_gpu_min(self, hl, cache, sel) -> int:
        """Rows for a non-resident expert to go zero-copy in this prefill call: fixed by
        EXL3_EXPERT_PREFILL_GPU_MIN, else chosen from the call's counts and the measured costs
        (one host sync for the counts and residency). Bootstrap value until the costs exist, and
        once after EXL3_MOE_COST_REMEASURE calls that left the CPU idle, so a stale estimate that
        sends everything to the GPU gets new samples"""
        if HT.dispatch == "gpu_exact":
            return 0
        if HT.prefill_gpu_min is not None:
            return HT.prefill_gpu_min
        if self.prefill_cost is None or not self.zc_us:
            return PREFILL_GPU_MIN_BOOT
        if HT.remeasure and self.prefill_stale >= HT.remeasure:
            self.prefill_stale = 0
            return PREFILL_GPU_MIN_BOOT
        E, L = hl.num_experts, hl.cache_layer.L
        a = cache.arrays
        s = sel.view(-1)
        s = torch.where((s >= 0) & (s < E), s, E)
        counts = torch.bincount(s, minlength = E + 1)[:E]
        resident = a["slotof"][L * EMAX: L * EMAX + E] >= 0
        mapped = a["reg_home"][L * EMAX: L * EMAX + E] != 0
        loc = a["reg_cpuloc"][L * MAX_NODES * EMAX: (L + 1) * MAX_NODES * EMAX].view(MAX_NODES, EMAX)[:, :E]
        has_cpu = (loc >= 0).any(dim = 0)
        st = torch.stack([counts.to(torch.long), resident.long(), mapped.long(), has_cpu.long()]).cpu().numpy()
        counts, resident, mapped, has_cpu = st[0], st[1] > 0, st[2] > 0, st[3] > 0
        used = counts > 0
        miss = used & ~resident
        flex = miss & mapped & has_cpu
        forced_gpu = int((miss & mapped & ~has_cpu).sum())
        forced_cpu = miss & ~mapped & has_cpu
        t_gpu0 = float(self.hit_us or 0.0) * int((used & resident).sum()) + float(self.zc_us) * forced_gpu
        nodes = sum(1 for idx in hl.host_idx if idx >= 0)
        pa, pb = self.prefill_cost
        gpu_min = choose_prefill_gpu_min(counts[flex], t_gpu0, float(self.zc_us),
                                         float(self.cpu_fixed_us or 0.0), pa, pb, nodes,
                                         (int(forced_cpu.sum()), int(counts[forced_cpu].sum())))
        cpu_any = bool((counts[flex] < gpu_min).any()) if flex.any() else False
        self.prefill_stale = 0 if cpu_any else self.prefill_stale + 1
        return gpu_min

    def combine(self, hl: HybridLayer, final_hidden_states, pending: HybridPending):
        """Fold the CPU lanes' partials (and a prefill call's stream lane) into the routed sum
        (fp32, rows x h), stream-ordered"""
        if pending is None:
            return final_hidden_states
        f2 = final_hidden_states.view(-1, final_hidden_states.shape[-1])
        cuda = hl.device.type == "cuda"
        tel = self.telemetry if cuda else None
        compute = torch.cuda.current_stream(hl.device) if cuda else None
        if pending.kind == "decode":
            # A split call that did not defer after all collects its deferred jobs here too
            items = pending.items + (pending.deferred or [])
            pending.deferred = None
            for i, h, r in items:
                e0 = tel.begin(compute) if tel is not None else None
                h.fused_collect(r, f2)
                if tel is not None:
                    tel.end(i, "decode", e0, compute)
                self.inflight = [x for x in self.inflight if x[1] is not r]
            return final_hidden_states
        if pending.stream is not None:
            # Stream lane first: its batches run on the compute stream while the CPU lanes finish
            st, pending.stream = pending.stream, None
            f2.add_(st.run())
        for i, tidx, out_t, done in pending.items:
            e0 = tel.begin(compute) if tel is not None else None
            if done is not None:
                compute.wait_event(done)
            f2.index_add_(0, tidx, out_t)
            if tel is not None:
                tel.end(i, "prefill", e0, compute)
        return final_hidden_states


def _bounce_linear(l, bounce: torch.Tensor, off: int, nbytes: int, device, shape = None):
    """Give a routed expert Linear an EXL3 inner module whose trellis is a view of the cache's
    bounce block (valid VRAM, never computed from: the cache owns the table rows) and whose
    suh/svh are the checkpoint's own (VRAM). Mirrors Linear.load_exl3 without the trellis read.
    shape overrides the checkpoint's trellis shape (CPU-only experts of a mixed-rate layer)"""
    from ..modules.quant.exl3 import LinearEXL3
    from ..modules.quant.exl3_lib.quantize import codebook_mul1_mult
    stc = l.config.stc
    key = l.key
    if shape is None:
        shape = stc.list_tensors(key)[key + ".trellis"]["shape"]
    numel = shape[0] * shape[1] * shape[2]
    assert numel * 2 == nbytes, f"{key}: trellis size mismatch"
    trellis = bounce[off: off + nbytes].view(torch.int16).view(*shape)

    def opt(sub, **kw):
        k = key + sub
        return stc.get_tensor(k, device, optional = True, **kw) if stc.has_tensor(k) else None
    su = opt(".su", no_defer = True)
    suh = opt(".suh")
    sv = opt(".sv", no_defer = True)
    svh = opt(".svh")
    mul1 = stc.get_codebook_marker(key + ".mul1", codebook_mul1_mult)
    l.device = device
    l.used_alt_key = False
    l.inner = LinearEXL3(l.config, l.in_features, l.out_features, None, su, sv, suh, svh, trellis,
                         None, mul1, None, l.out_dtype, key = key)
    l.quant_type = "exl3"


def fold_deferred(params: dict, x):
    """Fold the deferred CPU partial in params["moe_fold"], if any, into the residual x (the
    forward loop's safety net: before a module that would not fold it, and after the last)"""
    rec = params.get("moe_fold")
    if rec is None:
        return
    params["moe_fold"] = None
    if "hc_pending" in params:
        from ..modules.hyperconnections import hc_flush
        hc_flush(params)
    rec.dispatcher.fold(rec, x, rec.v)


def finalize_all(config):
    for d in getattr(config, "moe_hybrid_dispatchers", {}).values():
        d.finalize()


def note_kv_staging(model):
    """Once per load, at the model's first forward (every cache is attached by then): the largest
    quantized-cache prefill staging scratch of the model's attention layers, per device, joins
    the expert cache floor. Autosplit loads measure it already (max-merged, not counted twice);
    loads with an explicit split would otherwise leave only the fixed reserve free for it"""
    if getattr(model, "_hybrid_kv_staging_noted", False):
        return
    model._hybrid_kv_staging_noted = True
    from .moe_expert_cache import note_device_transient
    margin = int(os.environ.get("EXL3_AUTOSPLIT_MARGIN_MB", 256)) << 20
    per_device = {}
    for m in model.get_cache_layers():
        fn = getattr(m, "qc_staging_bytes", None)
        nbytes = fn() if fn is not None else 0
        if nbytes and m.device is not None:
            dev = torch.device(m.device)
            if dev.type == "cuda":
                i = dev.index if dev.index is not None else 0
                per_device[i] = max(per_device.get(i, 0), nbytes)
    for i, nbytes in per_device.items():
        note_device_transient(i, nbytes + margin)


def begin_pass_all(config, x, component: str | None = None):
    """Pass start of the forwarding component only: components share one Config (text model,
    MTP head, vision tower), and a vision encode must not size the text model's MoE headroom"""
    for comp, d in getattr(config, "moe_hybrid_dispatchers", {}).items():
        if component is None or comp == component:
            d.begin_pass(x)


def idle_all():
    """Generator queue drained: every device's cache owner decays, checks and regrows"""
    for c in ExpertCache.all():
        c.idle_maintenance()


def release_all_caches():
    """Before another component loads: give back every cached chunk (homes keep the data)"""
    for c in ExpertCache.all():
        c.release_all()

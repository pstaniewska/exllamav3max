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
forward is just a forward with more rows, routed through the same decode (rows below the
streaming threshold) or prefill path. Nothing depends on q_len == 1.

Numerics: every GPU lane runs the unmodified ExLlamaV3 kernels on the original EXL3 weights,
whether read from VRAM or over PCIe, so with EXL3_EXPERT_DISPATCH_POLICY=gpu_exact the layer is
identical to the all-VRAM model. CPU lanes use the existing int8-activation CPU kernels (the same
approximation as -mcl / -mcs); the hybrid policy trades that for latency, explicitly.

Lane policy inputs are measured, not assumed: the PCIe rate comes from a timed copy out of a
registered home at finalize, each node's CPU cost from per-job worker telemetry (compute time per
assignment). Until both are measured, EXL3_MOE_ZC_CPU_FRAC_INIT sets the CPU share of misses.
"""

from __future__ import annotations
import os
import weakref
import numpy as np
import torch

from .moe_expert_cache import ExpertCache, TUNING as CACHE_TUNING
from .moe_hybrid_ref import EMAX, LANE_ZC

DISPATCH_POLICIES = ("legacy", "gpu_exact", "hybrid")
NODE_POLICIES = {"numa_local": 0, "numa_sharded": 0, "numa_balanced": 1, "numa_adaptive": 2}
HOME_MODES = ("native_copy", "worker_arena")


class HybridTuning:

    def __init__(self):
        self.enabled = os.environ.get("EXL3_HYBRID_MOE", "0") not in ("", "0")
        self.dispatch = os.environ.get("EXL3_EXPERT_DISPATCH_POLICY", "hybrid").strip().lower()
        assert self.dispatch in DISPATCH_POLICIES, f"EXL3_EXPERT_DISPATCH_POLICY must be one of {DISPATCH_POLICIES}"
        self.home = os.environ.get("EXL3_EXPERT_HOME", "native_copy").strip().lower()
        assert self.home in HOME_MODES, f"EXL3_EXPERT_HOME must be one of {HOME_MODES}"
        self.node_policy = os.environ.get("EXL3_NUMA_POLICY", "numa_local").strip().lower()
        assert self.node_policy in NODE_POLICIES or self.node_policy == "legacy", \
            f"EXL3_NUMA_POLICY must be legacy or one of {tuple(NODE_POLICIES)}"
        zc = os.environ.get("EXL3_MOE_ZC_MAX", "auto").strip().lower()
        self.zc_max = None if zc == "auto" else int(zc)
        self.cpu_frac_init = float(os.environ.get("EXL3_MOE_ZC_CPU_FRAC_INIT", 0.5))
        self.zc_fused_rows = int(os.environ.get("EXL3_EXPERT_CACHE_ZC_FUSED_ROWS", 32))
        self.prefill_gpu_min = int(os.environ.get("EXL3_EXPERT_PREFILL_GPU_MIN", 8))
        self.ema = float(os.environ.get("EXL3_MOE_ADAPTIVE_EMA", 0.1))
        self.debug = os.environ.get("EXL3_HYBRID_MOE_DEBUG", "0") not in ("", "0")


HT = HybridTuning()


def hybrid_requested(config) -> bool:
    return bool(getattr(config.infer_params, "moe_hybrid", False)) and HT.dispatch != "legacy"


class HybridLayer:
    """Per-module hybrid state"""

    def __init__(self, dispatcher, module, device, dims, shards, host_idx, cpuloc):
        self.dispatcher = dispatcher
        self.module_ref = weakref.ref(module)
        self.key = module.key
        self.uid = f"{dispatcher.comp}:{module.key}"
        self.device = torch.device(device)
        self.num_experts = module.num_experts
        self.topk = module.num_experts_per_tok
        self.dims = dims                    # dict g/u/d: (k, n, K) ; g None when gateless
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
        # Decode-size static buffers (rows below the streaming threshold): masked selection
        cap = max(dispatcher.cap_rows, 1) * self.topk
        self.sel_buf = torch.empty((cap,), dtype = torch.long, device = self.device)
        self.lane_buf = torch.empty((EMAX,), dtype = torch.int32, device = self.device)
        self._worst = {}


class HybridPending:
    __slots__ = ("kind", "items", "fused_cap")

    def __init__(self, kind, items, fused_cap = None):
        self.kind = kind
        self.items = items
        self.fused_cap = fused_cap


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
        self.device0 = torch.device(device)
        self.node_policy = NODE_POLICIES.get(HT.node_policy, 0)
        telemetry = HT.node_policy == "numa_adaptive" or (HT.dispatch == "hybrid" and HT.zc_max is None)
        self.group = HostGroup(config, comp, self.device0.index, home = HT.home, telemetry = telemetry)
        if self.group.numa and HT.home == "native_copy":
            # One native copy is enough: with replicated worker arenas only one host writes it
            from .moe_numa_hosts import memory_policy
            if memory_policy() == "replicate":
                keep = self.group.pref if self.group.pref >= 0 else 0
                for i, h in enumerate(self.group.hosts):
                    h.home = i == keep
        self.cap_rows = HOST_TUNING.cap_rows
        self.prefill_rows = HOST_TUNING.stream_min_rows
        self.layers: list[HybridLayer] = []
        self.devices: set = set()
        n = self.group.num_nodes
        self.node_cost = torch.ones((n,), dtype = torch.float)
        self.cpu_us = [None] * n            # EMA of measured CPU microseconds per assignment
        self.tel_seen = [dict() for _ in range(n)]   # per host: slot -> last seq counted
        self.zc_us = None                   # measured microseconds to read one expert over PCIe
        self.cpu_frac = HT.cpu_frac_init
        self.side = {}                      # (host index, device index) -> stream
        self.notices = set()
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
        dims = {"g": None}
        for name, ls in projs:
            shape0 = None
            for l in ls:
                if getattr(l, "fkey", None) or not l.is_exl3_storage(l.key):
                    return False, "experts not stored as separate EXL3 tensors", None
                t = stc.list_tensors(l.key)
                if l.key + ".bias" in t:
                    return False, "per-expert biases", None
                shp = list(t[l.key + ".trellis"]["shape"])
                if shape0 is None:
                    shape0 = shp
                    if l.key + ".mul1" not in t:
                        return False, "experts are not mul1", None
                elif shp != shape0:
                    return False, "experts of one projection differ in shape", None
            K = trellis_rate(shape0[2])
            if K is None:
                return False, "unsupported bitrate", None
            dims[name] = (shape0[0] * 16, shape0[1] * 16, K)
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
                proj_dims = dict(g = dims["g"], u = dims["u"], d = dims["d"]),
                aux = None,
                interm_fp32 = module.interm_dtype == torch.float,
            )
            host_idx.append(idx)
            for j, e in enumerate(sh):
                cpuloc[i, e] = j

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
            for l in ls:
                _bounce_linear(l, bounce, offs[name], hl.proj_bytes[name], device)
        self.layers.append(hl)
        self.devices.add(device)
        module.hybrid = hl
        # Every tier addresses experts by router id through the tables; out-of-range ids (the
        # masked picks) are dropped by the range test of the fused decode kernels
        module.routing_first = 0
        module.routing_last = module.num_experts
        print(f" -- hybrid MoE experts: {module.key} ({module.num_experts} experts, "
              f"{hl.expert_bytes / 2**20:.2f} MiB each, {len(hosts)} worker node(s))")
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
        if self.finalized:
            self._finalize_layer(hl)

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
        # The autosplit loader's largest measured transient per device (set after its module
        # loop) becomes part of the cache floor
        margin = int(os.environ.get("EXL3_AUTOSPLIT_MARGIN_MB", 256)) << 20
        for dev in self.devices:
            hint = getattr(self.config, "autosplit_transient", {}).get(dev.index or 0)
            if hint:
                ExpertCache.for_device(dev).note_transient(hint + margin)
        for hl in self.layers:
            if hl.cache_layer is not None:
                self._finalize_layer(hl)
        if HT.dispatch == "hybrid" and HT.zc_max is None:
            self._probe_pcie()

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
                    if src[e] is not None:
                        continue
                    homes[e] = h.home_device_ptr(idx, j)
                    src[e] = (h, idx, j)
        missing = [e for e in range(E) if not homes[e] and (hl.cpuloc[:, e] < 0).all()]
        if missing:
            raise RuntimeError(f"hybrid MoE {hl.key}: experts {missing[:8]} have neither a GPU-readable home "
                               f"nor a CPU worker copy")
        n_nozc = int((homes == 0).sum())
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
        gbs = nb / max(ms, 1e-3) / 1e6
        self.zc_us = self.layers[0].expert_bytes / (gbs * 1e3)
        if HT.debug:
            print(f" -- hybrid MoE: host->device {gbs:.1f} GB/s, {self.zc_us:.0f} us per expert read")

    # ---- passes -------------------------------------------------------------------------------

    def begin_pass(self, x):
        """Forward start: lazy finalize, cache sizing / headroom, telemetry and lane parameters"""
        if not self.layers:
            return
        if not self.finalized:
            self.finalize()
        if x is not None and torch.is_tensor(x):
            rows = x.numel() if x.dtype in (torch.long, torch.int) else x.numel() // max(1, x.shape[-1])
        else:
            rows = 1
        for dev in self.devices:
            cache = ExpertCache.for_device(dev)
            stream = torch.cuda.current_stream(dev)
            cache.begin_pass(rows, self.prefill_rows, stream)
            if rows >= self.prefill_rows:
                extra = max((self.worst_case(hl, rows) for hl in self.layers if hl.device == dev), default = 0)
                cache.ensure_headroom(extra)
        self._update_lane_params()

    def _update_lane_params(self):
        """Node costs and the CPU share of misses from measurements (telemetry, PCIe probe)"""
        changed = False
        for i, h in enumerate(self.group.hosts):
            for slot, (seq, us, na, ne) in enumerate(h.read_telemetry()):
                if not na or self.tel_seen[i].get(slot) == seq:
                    continue
                self.tel_seen[i][slot] = seq
                v = us / na
                self.cpu_us[i] = v if self.cpu_us[i] is None else (1 - HT.ema) * self.cpu_us[i] + HT.ema * v
                changed = True
        if not changed:
            return
        known = [c for c in self.cpu_us if c]
        if known:
            mean = sum(known) / len(known)
            for i, c in enumerate(self.cpu_us):
                self.node_cost[i] = (c or mean) / mean
        if HT.zc_max is None and self.zc_us and known:
            # Parallel nodes: aggregate CPU rate is the sum of per-node rates. Min-max split of
            # m misses between one PCIe stream (zc_us each) and the CPU nodes (t_cpu each):
            # share to CPU = zc / (zc + t_cpu)
            t_cpu = 1.0 / sum(1.0 / c for c in known)
            self.cpu_frac = float(self.zc_us / (self.zc_us + t_cpu))

    def worst_case(self, hl: HybridLayer, rows: int) -> int:
        """Upper bound on a hybrid layer's prefill transient for `rows` tokens: the GPU tiers'
        bound (fused, batched groups; the heavy per-expert fallback is bounded by one batched
        group of max_rows rows plus its fp32 down slab), the CPU lanes' staging (per node: the
        gathered rows, selections and the fp32 partial) and the selection buffers"""
        w = hl._worst.get(rows)
        if w is not None:
            return w
        m = hl.module_ref()
        if m is None:
            return 0
        A = rows * hl.topk
        gpu = sum(m.prefill_worst_case_parts(rows, A))
        recon = m.batch_recon if m.batch_recon not in (None, False) else None
        heavy_rows = recon.max_rows if recon is not None else 0
        gpu += heavy_rows * hl.ho * 4
        n = len(self.group.hosts)
        cpu = n * (rows * hl.ho * 4 + rows * hl.hi * 2 + rows * hl.topk * 12 + self.cap_rows * hl.ho * 4)
        sel = A * (8 + 4 * n) + EMAX * 4
        w = gpu + cpu + sel
        hl._worst[rows] = w
        return w

    # ---- per layer call --------------------------------------------------------------------------

    def _nodes_decode(self, hl, rows):
        res = []
        nodes = torch.zeros((5, len(self.group.hosts)), dtype = torch.long)
        for i, (h, idx) in enumerate(zip(self.group.hosts, hl.host_idx)):
            r = h.fused_reserve(idx, rows, hl.device)
            res.append((h, r))
            slot = r["slot"]
            nodes[0, i] = slot["sel_dev"]
            nodes[1, i] = slot["x_dev"]
            nodes[2, i] = slot["w_dev"]
            nodes[3, i] = r["counts"].data_ptr()
            nodes[4, i] = r["slot_idx"]
        return res, nodes

    def _lane_args(self, n_picks: int):
        gpu_exact = HT.dispatch == "gpu_exact"
        gpu_max = -1 if (gpu_exact or HT.zc_max is None) else HT.zc_max
        cpu_frac = 0.0 if gpu_exact else self.cpu_frac
        admit = int(n_picks <= CACHE_TUNING.admit_max)
        return gpu_max, cpu_frac, admit

    def submit(self, hl: HybridLayer, y, selected_experts, routing_weights):
        """Lane assignment and CPU job issue for one layer call. Returns (pending, sel_gpu): the
        GPU tiers must run on sel_gpu (CPU-lane picks masked to num_experts)"""
        rows = y.shape[0]
        cache = ExpertCache.for_device(hl.device)
        y = y.contiguous()
        w = routing_weights.contiguous()
        sel = selected_experts.contiguous()
        n = rows * hl.topk
        if rows < self.prefill_rows and rows <= self.cap_rows:
            res, nodes = self._nodes_decode(hl, rows)
            sel_gpu = hl.sel_buf[:n]
            gpu_max, cpu_frac, admit = self._lane_args(n)
            cache.issue(hl.cache_layer, sel, y, w, sel_gpu, None, None, nodes, self.node_cost, hl.hi,
                        0, admit, gpu_max, cpu_frac, HT.prefill_gpu_min, self.node_policy, self.group.pref)
            for h, r in res:
                h.fused_publish(r)
            return HybridPending("decode", res), sel_gpu.view(rows, hl.topk)
        return self._submit_prefill(hl, cache, y, sel, w, rows, n)

    def _submit_prefill(self, hl, cache, y, sel, w, rows, n):
        from ..util.device_copy import host_to_device
        hosts = self.group.hosts
        N = len(hosts)
        dev = hl.device
        sel_gpu = torch.empty((n,), dtype = torch.long, device = dev)
        cpu_sel = torch.empty((N, n), dtype = torch.int32, device = dev)
        nodes = torch.zeros((5, N), dtype = torch.long)
        gpu_min = 0 if HT.dispatch == "gpu_exact" else HT.prefill_gpu_min
        cache.issue(hl.cache_layer, sel, y, w, sel_gpu, hl.lane_buf, cpu_sel, nodes, self.node_cost, hl.hi,
                    1, 0, -1, 0.0, gpu_min, self.node_policy, self.group.pref)
        # One host sync for the tier plan and the CPU row compression (the GPU tiers sync on
        # their own counts anyway)
        both = torch.cat([hl.lane_buf[:hl.num_experts], cpu_sel.view(-1)]).cpu().numpy()
        lanes = both[:hl.num_experts]
        cs = both[hl.num_experts:].reshape(N, rows, hl.topk)
        compute = torch.cuda.current_stream(dev)
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
            for t in (y_c, w_c, sel_c, out_t, tidx_d):
                t.record_stream(side)
            items.append((tidx_d, out_t, done))
        any_zc = bool((lanes == LANE_ZC).any())
        return HybridPending("prefill", items, HT.zc_fused_rows if any_zc else None), sel_gpu.view(rows, hl.topk)

    def combine(self, hl: HybridLayer, final_hidden_states, pending: HybridPending):
        """Fold the CPU lanes' partials into the routed sum (fp32, rows x h), stream-ordered"""
        if pending is None:
            return final_hidden_states
        f2 = final_hidden_states.view(-1, final_hidden_states.shape[-1])
        if pending.kind == "decode":
            for h, r in pending.items:
                h.fused_collect(r, f2)
            return final_hidden_states
        compute = torch.cuda.current_stream(hl.device)
        for tidx, out_t, done in pending.items:
            compute.wait_event(done)
            f2.index_add_(0, tidx, out_t)
        return final_hidden_states


def _bounce_linear(l, bounce: torch.Tensor, off: int, nbytes: int, device):
    """Give a routed expert Linear an EXL3 inner module whose trellis is a view of the cache's
    bounce block (valid VRAM, never computed from: the cache owns the table rows) and whose
    suh/svh are the checkpoint's own (VRAM). Mirrors Linear.load_exl3 without the trellis read"""
    from ..modules.quant.exl3 import LinearEXL3
    from ..modules.quant.exl3_lib.quantize import codebook_mul1_mult
    stc = l.config.stc
    key = l.key
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


def finalize_all(config):
    for d in getattr(config, "moe_hybrid_dispatchers", {}).values():
        d.finalize()


def begin_pass_all(config, x):
    for d in getattr(config, "moe_hybrid_dispatchers", {}).values():
        d.begin_pass(x)


def idle_all():
    """Generator queue drained: every device's cache owner decays, checks and regrows"""
    for c in ExpertCache.all():
        c.idle_maintenance()


def release_all_caches():
    """Before another component loads: give back every cached chunk (homes keep the data)"""
    for c in ExpertCache.all():
        c.release_all()

"""
Hybrid MoE dispatcher (model/moe_hybrid_dispatch.py), decode path, on the CPU: a dispatcher wired
to a reference-backend expert cache and two fake worker hosts (the slot protocol of MoeCpuHost:
fused_reserve / fused_publish / fused_collect) must mask CPU-lane picks out of the GPU selection,
stage every CPU pick into the right node's slot, fold the nodes' partials into the routed sum,
and derive the CPU share of misses from measured costs only. No GPU needed; the prefill path
(side streams) and the CUDA kernels are covered on GPU hosts.
"""

import numpy as np
import pytest
import torch

from testlib import hybrid_isolated

pytestmark = pytest.mark.nogpu

H = hybrid_isolated.load()
D = H.dispatch
ref = H.ref


class FakeHost:
    def __init__(self, n, rows, hi, ho):
        self.sel = torch.full((n,), -9, dtype = torch.int32)
        self.x = torch.zeros((rows * hi,), dtype = torch.half)
        self.w = torch.zeros((n,), dtype = torch.half)
        self.out = torch.randn((rows, ho))
        self.counts = torch.zeros((8,), dtype = torch.int32)
        self.events = []
        self.tel = []

    def fused_reserve(self, idx, rows, device):
        self.events.append("reserve")
        slot = dict(sel_dev = self.sel.data_ptr(), x_dev = self.x.data_ptr(), w_dev = self.w.data_ptr(),
                    x_cap = self.x.numel(), sel_cap = self.sel.numel(), sel = self.sel)
        return dict(seq = 1, slot_idx = 0, slot = slot, counts = self.counts, ho = self.out.shape[1], dev = device, rows = rows)

    def fused_publish(self, r):
        self.events.append("publish")

    def fused_collect(self, r, final_2d):
        self.events.append("collect")
        if int(self.counts[0]):
            final_2d += self.out[:final_2d.shape[0], :final_2d.shape[1]]

    def read_telemetry(self):
        return self.tel

    # Prefill protocol (MoeCpuHost._issue_compute / _collect_compute): a fake expert function,
    # out[r] = sum_k w[r, k] * (local + 1 + 100 * node) * y[r] over this node's picks
    node = 0
    specs = [None] * 8
    unregistered = 0
    started = True
    acked = 8

    def _issue_compute(self, idx, y, sel, w, spec, out, h):
        out.zero_()
        for r in range(y.shape[0]):
            for k in range(sel.shape[1]):
                j = int(sel[r, k])
                if j >= 0:
                    out[r] += float(w[r, k]) * (j + 1 + 100 * self.node) * y[r].float()
        return [], None

    def _collect_compute(self, jobs, out, rtmp, h):
        pass

    def unregister(self):
        self.unregistered += 1


def make(E = 16, N = 2, rows = 1, topk = 4, h = 8, **ht):
    H.cache.ExpertCache._drop_registry()
    T = H.cache.CacheTuning()
    T.chunk_slots, T.lcap, T.gmax = 4, 8, 16
    T.reserve_bytes = T.min_free_bytes = T.grow_hysteresis = 0
    T.limit_bytes, T.warm, T.save_stats, T.policy = None, None, None, "clock"
    cache = H.cache.ExpertCache(torch.device("cpu"), "ref", mem_info = lambda: (1 << 20, 1 << 20), tuning = T)
    H.cache._CACHES[("cpu", 0)] = cache
    eb = 1024
    tabs = [torch.zeros(E, dtype = torch.long) for _ in range(3)]
    layer = cache.register_layer("text:l", "l", E, eb, (0, 256, 512), [(t, 0, 1, p) for p, t in enumerate(tabs)])
    home = torch.zeros((E * eb,), dtype = torch.uint8)
    cache.set_layer_homes(layer, np.array([home.data_ptr() + e * eb for e in range(E)]), lambda e: home[e * eb:(e + 1) * eb])
    loc = np.full((N, E), -1, dtype = np.int32)
    for e in range(E):
        loc[e % N, e] = e // N
    cache.set_layer_cpuloc(layer, loc)
    cache.begin_pass(1, 32)

    for k, v in ht.items():
        setattr(D.HT, k, v)
    d = object.__new__(D.HybridDispatcher)
    hosts = [FakeHost(rows * topk, rows, h, h) for _ in range(N)]
    for q, fh in enumerate(hosts):
        fh.node = q
    d.group = type("G", (), dict(hosts = hosts, pref = -1, num_nodes = N))()
    d.node_policy = 0
    d.cap_rows, d.prefill_rows = 64, 32
    d.decode_rows = D.decode_rows(d.cap_rows)
    d.layers, d.notices, d.tag, d.comp = [], set(), "text", "text"
    d._init_runtime(N)
    d.devices.add(torch.device("cpu"))
    hl = object.__new__(D.HybridLayer)
    hl.device = torch.device("cpu")
    hl.num_experts, hl.topk, hl.h, hl.hi, hl.ho = E, topk, h, h, h
    hl.cache_layer = layer
    hl.finalized = True
    hl.host_idx = list(range(N))
    hl.sel_buf = torch.empty((64 * topk,), dtype = torch.long)
    hl.lane_buf = torch.empty((ref.EMAX,), dtype = torch.int32)
    for k, v in ht.items():
        setattr(D.HT, k, v)
    return d, hl, hosts, cache


@pytest.fixture(autouse = True)
def restore_ht():
    saved = dict(vars(D.HT))
    yield
    vars(D.HT).update(saved)


def test_decode_masks_cpu_picks_and_combines():
    d, hl, hosts, cache = make(dispatch = "hybrid", zc_max = 2)
    sel = torch.tensor([[1, 2, 3, 4]])
    y = torch.randn((1, 8)).half()
    w = torch.rand((1, 4)).half()
    pend, sel_gpu = d.submit(hl, y, sel, w)
    on_cpu = [i for i in range(4) if int(sel_gpu[0, i]) == 16]
    assert len(on_cpu) == 2
    for q, h in enumerate(hosts):
        assert h.events == ["reserve", "publish"]
        for i in range(4):
            e = int(sel[0, i])
            expect = e // 2 if (i in on_cpu and e % 2 == q) else -1
            assert int(h.sel[i]) == expect
    final = torch.zeros((1, 8))
    out = d.combine(hl, final, pend)
    want = sum(h.out for h in hosts if int(h.counts[0]))
    assert torch.allclose(out, want)


def test_gpu_exact_never_uses_cpu_lanes():
    d, hl, hosts, cache = make(dispatch = "gpu_exact", zc_max = None)
    sel = torch.tensor([[5, 6, 7, 8]])
    pend, sel_gpu = d.submit(hl, torch.randn((1, 8)).half(), sel, torch.rand((1, 4)).half())
    assert torch.equal(sel_gpu, sel)
    assert all(int(h.counts[0]) == 0 for h in hosts)


def fake_expert_sum(sel, w, y, lanes, E, N):
    """Expected CPU partial of the fake hosts for the lanes the call got"""
    out = torch.zeros((y.shape[0], y.shape[1]))
    for r in range(sel.shape[0]):
        for k in range(sel.shape[1]):
            e = int(sel[r, k])
            q = int(lanes[e])
            if q >= 0:
                out[r] += float(w[r, k]) * (e // N + 1 + 100 * q) * y[r].float()
    return out


def test_cost_model_only_from_measurements():
    d, hl, hosts, cache = make(dispatch = "hybrid", zc_max = None)
    d._update_lane_params()
    assert float(d.lane_cost.abs().sum()) == 0 and d.cpu_frac == 0.5     # nothing measured
    d.zc_us, d.hit_us, d.cpu_fixed_us = 200.0, 20.0, 30.0
    hosts[0].tel = [(5, 400, 8, 4, 2)]          # 100 us per distinct expert (8 assignments)
    hosts[1].tel = [(9, 300, 2, 2, 1),          # 150 us per distinct expert
                    (11, 9000, 64, 2, 32)]      # prefill-shaped job: not a decode cost sample
    d._update_lane_params()
    assert d.lane_cost.tolist() == pytest.approx([20.0, 200.0, 30.0, 150.0])   # slowest node
    before = d.cpu_us[:]
    d._update_lane_params()                     # same records again: not recounted
    assert d.cpu_us == before
    _, _, _, lc = d._lane_args(4)
    assert lc is d.lane_cost
    D.HT.dispatch = "gpu_exact"
    assert d._lane_args(4)[3] is None           # exact mode never passes costs


@pytest.mark.parametrize("decode_max", [None, 31])
@pytest.mark.parametrize("rows", [1, 8, 9, 31, 32, 33, 64, 65, 128, 256, 512])
def test_row_boundaries_pick_the_path_and_fold_cpu_partials(rows, decode_max):
    # rows <= the decode boundary (EXL3_EXPERT_DECODE_MAX_ROWS, default the slot capacity 64; 31
    # is the former streaming-threshold boundary): decode path (slot protocol); otherwise the
    # prefill path (CPU rows compacted per node, folded in after the GPU tiers). Admission only
    # for calls with <= EXL3_EXPERT_CACHE_ADMIT_MAX picks
    topk, E, N = 4, 32, 2
    d, hl, hosts, cache = make(E = E, N = N, rows = rows, topk = topk, dispatch = "hybrid", zc_max = 1,
                               decode_max_rows = decode_max)
    assert d.decode_rows == (64 if decode_max is None else decode_max)
    hl.hi = hl.h = 8
    torch.manual_seed(rows)
    sel = torch.stack([torch.randperm(E)[:topk] for _ in range(rows)])
    y = torch.randn((rows, 8)).half()
    w = torch.rand((rows, topk)).half()
    admits0 = cache.stats()["admits"]
    pend, sel_gpu = d.submit(hl, y, sel, w)
    decode = rows <= d.decode_rows
    assert pend.kind == ("decode" if decode else "prefill")
    if decode:
        for h in hosts:
            assert h.events == ["reserve", "publish"]
    else:
        assert all(h.events == [] for h in hosts)
    lanes = torch.full((E,), ref.LANE_NONE, dtype = torch.int32)
    for i, e in enumerate(sel.view(-1).tolist()):
        if int(sel_gpu.view(-1)[i]) == E:
            q = [q for q in range(N) if e % N == q][0]
            lanes[e] = q
        else:
            assert int(sel_gpu.view(-1)[i]) == e
            lanes[e] = ref.LANE_ZC
    admitted = cache.stats()["admits"] - admits0
    if not decode or rows * topk > cache.T.admit_max:
        assert admitted == 0
    final = torch.zeros((rows, 8))
    out = d.combine(hl, final, pend)
    if decode:
        want = sum(h.out[:rows] for h in hosts if int(h.counts[0]))
        want = want if torch.is_tensor(want) else torch.zeros((rows, 8))
        assert torch.allclose(out, want)
    else:
        assert torch.allclose(out, fake_expert_sum(sel, w, y, lanes, E, N), atol = 1e-2)
        assert pend.fused_cap in (None, D.HT.zc_fused_rows)


def test_prefill_gpu_exact_keeps_the_fused_plan():
    d, hl, hosts, cache = make(E = 16, N = 2, rows = 80, dispatch = "gpu_exact", zc_max = None)
    sel = torch.stack([torch.randperm(16)[:4] for _ in range(80)])
    pend, sel_gpu = d.submit(hl, torch.randn((80, 8)).half(), sel, torch.rand((80, 4)).half())
    assert pend.kind == "prefill" and pend.fused_cap is None
    assert torch.equal(sel_gpu, sel) and pend.items == []


def test_decode_skips_a_node_without_experts_of_the_layer():
    d, hl, hosts, cache = make(dispatch = "hybrid", zc_max = 0)
    hl.host_idx = [0, -1]
    loc = np.full((2, 16), -1, dtype = np.int32)
    loc[0] = np.arange(16)
    cache.set_layer_cpuloc(hl.cache_layer, loc)
    pend, sel_gpu = d.submit(hl, torch.randn((1, 8)).half(), torch.tensor([[1, 2, 3, 4]]), torch.rand((1, 4)).half())
    assert hosts[1].events == [] and hosts[0].events == ["reserve", "publish"]
    assert (sel_gpu == 16).all()
    d.combine(hl, torch.zeros((1, 8)), pend)


def test_component_tags_are_unique_and_released():
    D._LIVE_TAGS.clear()
    tags = [D.acquire_tag(c) for c in ("text", "mtp", "text", "text")]
    assert tags == ["text", "mtp", "text#1", "text#2"]
    D.release_tag("text")
    assert D.acquire_tag("text") == "text"      # a reload gets the plain name back
    D._LIVE_TAGS.clear()


def test_last_detach_resets_per_load_state():
    d, hl, hosts, cache = make(dispatch = "hybrid", zc_max = None)
    d.layers = [hl]
    d.finalized = True
    d.zc_us = 5.0
    d.lane_cost[:] = 1.0
    module = type("M", (), {})()
    module.hybrid = hl
    d.detach(module)
    assert hosts[0].unregistered == 1 and hosts[1].unregistered == 1
    assert not d.finalized and not d.devices and d.zc_us is None and float(d.lane_cost.sum()) == 0
    assert hl.cache_layer is None and cache.layers == {}


def test_attach_finalizes_only_when_workers_are_ready():
    d, hl, hosts, cache = make(dispatch = "hybrid", zc_max = None)
    assert d._hosts_ready(hl)
    hosts[1].started = False
    assert not d._hosts_ready(hl)
    hosts[1].started, hosts[1].acked = True, 1
    assert not d._hosts_ready(hl)               # layer index 1 not acked yet
    hl.host_idx = [0, -1]
    assert d._hosts_ready(hl)


def test_load_module_failure_undoes_worker_registration(monkeypatch):
    d, hl, hosts, cache = make(dispatch = "hybrid", zc_max = None)
    for fh in hosts:
        fh.register_layer = lambda *a, **k: 0
    d.group.shard = lambda E: [list(range(0, E, 2)), list(range(1, E, 2))]
    dims = dict(g = (16, 16, 2), u = (16, 16, 2), d = (16, 16, 2))
    monkeypatch.setattr(d, "eligible", lambda m: (True, "", dims))

    class Boom:
        def load(self, *a, **k):
            raise MemoryError("CUDA out of memory (simulated)")

    class L:
        def __init__(self, i):
            self.key = f"e{i}"

    m = type("M", (), {})()
    m.key, m.num_experts, m.num_experts_per_tok, m.gated = "mlp", 4, 2, True
    m.gates, m.ups, m.downs = [L(i) for i in range(4)], [L(i) for i in range(4)], [L(i) for i in range(4)]
    m.modules = [Boom()]
    m.activation_fn, m.act_limit, m.interm_dtype = "silu", None, torch.half
    m.config = type("C", (), dict(stc = None))()
    with pytest.raises(MemoryError):
        d.load_module(m, "cpu")
    assert [h.unregistered for h in hosts] == [1, 1]
    assert not hasattr(m, "hybrid") or m.hybrid is None


def test_worst_case_is_bounded_by_the_rows():
    d, hl, hosts, cache = make(dispatch = "hybrid", zc_max = None)
    recon = type("R", (), dict(max_rows = 1 << 30))()
    mod = type("M", (), dict(batch_recon = recon, prefill_worst_case_parts = lambda self, r, a: (0,)))()
    hl.module_ref = lambda: mod
    hl._worst = {}
    w = d.worst_case(hl, 2048)
    assert w < (1 << 30)                        # was 2^30 * ho * 4 with RECON_TILES=0


def test_telemetry_harvest_without_sync():
    class Ev:
        def __init__(self, done, t):
            self.done, self.t = done, t

        def query(self):
            return self.done

        def elapsed_time(self, other):
            return other.t - self.t

    t = D.Telemetry(2, cap = 2)
    t.pending = [(0, "decode", Ev(True, 0.0), Ev(True, 0.5)), (1, "decode", Ev(True, 0.0), Ev(False, 0.0))]
    t.harvest()
    assert t.waits == [1, 0] and t.wait_us[0] == pytest.approx(500.0) and len(t.pending) == 1
    s = t.summary()
    assert s["cpu_wait_us_mean"][0] == pytest.approx(500.0)


def test_worker_death_and_oversized_jobs_raise():
    h = object.__new__(H.host.MoeCpuHost)
    h.started, h.cap_rows, h.specs = True, 4, [dict(topk = 2, ho = 8)]
    h.v_abort = np.zeros((1,), dtype = np.uint32)
    with pytest.raises(ValueError, match = "slot capacity"):
        h.fused_reserve(0, 5, torch.device("cpu"))
    h.v_abort[0] = 1
    with pytest.raises(RuntimeError, match = "abort"):
        h.fused_reserve(0, 1, torch.device("cpu"))
    # After the host synchronized with a forward that a dead worker's released waits let
    # through, the iteration's results are discarded with an error
    cfg = type("C", (), {})()
    cfg.moe_cpu_hosts = {"text@0": h}
    with pytest.raises(RuntimeError, match = "died or timed out"):
        H.host.check_workers(cfg, None)
    h.v_abort[0] = 0
    H.host.check_workers(cfg, None)


def test_interrupted_forward_leaves_no_stuck_slot():
    # An exception between submit and combine (or before publish) must not leave a reserved
    # job unpublished or uncollected: the slot's next tenant would wait for it forever
    d, hl, hosts, cache = make(dispatch = "hybrid", zc_max = 0)
    d.layers = [hl]
    d.finalized = True
    pend, _ = d.submit(hl, torch.randn((1, 8)).half(), torch.tensor([[1, 2, 3, 4]]), torch.rand((1, 4)).half())
    assert len(d.inflight) == 2 and all(p for _, _, p in d.inflight)
    # combine never runs (interrupted); a reservation without publish as well
    r = hosts[0].fused_reserve(0, 1, torch.device("cpu"))
    d.inflight.append([hosts[0], r, False])
    hosts[0].sel.fill_(5)
    d._update_lane_params = lambda: None
    d.begin_pass(None)
    assert d.inflight == []
    assert (hosts[0].sel == -1).all()                       # stale selections cleared before publish
    assert hosts[0].events.count("collect") == 2 and hosts[1].events.count("collect") == 1
    assert hosts[0].events.count("publish") == 2


def test_cost_model_remeasures_when_cpu_samples_stop():
    d, hl, hosts, cache = make(dispatch = "hybrid", zc_max = None, remeasure = 3)
    d.zc_us, d.hit_us, d.cpu_fixed_us = 200.0, 20.0, 30.0
    hosts[0].tel = [(5, 400, 4, 4, 1)]
    d._update_lane_params()
    assert d._lane_args(4)[3] is d.lane_cost and not d.explore
    for i in range(3):                          # no new decode-shaped CPU samples
        d._update_lane_params()
    assert d.explore and d._lane_args(4)[3] is None   # one pass on the bootstrap share
    d._update_lane_params()
    assert not d.explore
    hosts[0].tel = [(6, 380, 4, 4, 1)]          # a new sample resets the count
    d._update_lane_params()
    assert d.cost_stale == 0


@pytest.mark.parametrize("env, want", [(None, 64), (31, 31), (200, 64), (0, 0)])
def test_decode_boundary_is_capped_by_the_slot_capacity(env, want):
    D.HT.decode_max_rows = env
    assert D.decode_rows(64) == want


def test_kv_staging_joins_the_floor_once_per_load():
    # An explicit-split load has no autosplit measurement: the quantized-cache prefill staging
    # scratch (largest per device, plus the autosplit margin) must reach the cache floor at the
    # first forward, max-merged with what autosplit noted, and only once per load
    class Attn:
        def __init__(self, device, nbytes):
            self.device, self.nbytes, self.calls = device, nbytes, 0

        def qc_staging_bytes(self):
            self.calls += 1
            return self.nbytes

    class Model:
        def __init__(self, layers):
            self.layers = layers

        def get_cache_layers(self):
            return self.layers

    H.cache.ExpertCache._drop_registry()
    margin = 256 << 20
    layers = [Attn("cuda:0", 300 << 20), Attn("cuda:0", 500 << 20), Attn("cuda:1", 0), Attn(None, 1 << 30),
              Attn(torch.device("cpu"), 1 << 30)]
    model = Model(layers)
    H.cache.note_device_transient(1, 7 << 20)
    D.note_kv_staging(model)
    assert H.cache._DEVICE_TRANSIENT[("cuda", 0)] == (500 << 20) + margin
    assert H.cache._DEVICE_TRANSIENT[("cuda", 1)] == 7 << 20      # nothing staged there
    D.note_kv_staging(model)
    assert all(a.calls == 1 for a in layers)
    H.cache.note_device_transient(0, 2 << 30)                       # autosplit measured more
    model._hybrid_kv_staging_noted = False                          # next load
    D.note_kv_staging(model)
    assert H.cache._DEVICE_TRANSIENT[("cuda", 0)] == 2 << 30
    H.cache.ExpertCache._drop_registry()


def test_prefill_cost_fit_separates_per_expert_and_per_row_cost():
    f = D.PrefillCostFit()
    assert f.solve() is None
    for _ in range(20):                         # decode-shaped only: one assignment per expert
        f.add(4, 4, 4 * 300 + 4 * 20)
    assert f.solve() is None                    # a and b not separable yet
    for ne, na in ((8, 120), (3, 90), (16, 200), (5, 15)):
        f.add(ne, na, ne * 300 + na * 20)
    a, b = f.solve()
    assert a == pytest.approx(300, rel = 1e-6) and b == pytest.approx(20, rel = 1e-6)
    g = D.PrefillCostFit()
    for ne, na in ((2, 50), (4, 10), (6, 90)):
        g.add(ne, na, 5.0 * na)                 # pure per-row cost: a clamps to zero
    a, b = g.solve()
    assert a == pytest.approx(0, abs = 1e-6) and b == pytest.approx(5, rel = 1e-6)


def test_choose_prefill_gpu_min_balances_pcie_and_cpu():
    counts = np.array([1, 2, 2, 3, 5, 8, 13, 21, 34, 55, 0])
    pick = lambda **kw: D.choose_prefill_gpu_min(counts, **dict(dict(t_gpu0 = 0.0, t_zc = 1000.0, t_fixed = 50.0,
                                                                      a = 100.0, b = 10.0, nodes = 2), **kw))
    # Cheap CPU: every expert fits on the CPU before one PCIe read would finish
    assert pick(a = 1.0, b = 0.1) == 56
    # CPU slower than PCIe for any expert: everything stays on the GPU
    assert pick(a = 1e6, b = 1e3) == 1
    # In between: the small experts go to the CPU, the threshold never splits equal counts
    t = pick()
    assert 1 < t <= 55
    c = np.sort(counts[counts > 0])
    k = int((c < t).sum())
    worst = lambda k: max((50.0 + (100 * k + 10 * c[:k].sum()) / 2) if k else 0.0, (c.size - k) * 1000.0)
    assert worst(k) == min(worst(j) for j in range(c.size + 1) if j in (0, c.size) or c[j] != c[j - 1])
    # More GPU work already queued (hits): more experts move to the CPU
    assert pick(t_gpu0 = 5000.0) >= t
    assert D.choose_prefill_gpu_min(np.zeros(4, dtype = np.int64), 0, 1, 0, 1, 1, 1) == D.PREFILL_GPU_MIN_BOOT


def test_prefill_threshold_bootstrap_fixed_and_measured():
    topk, E, N, rows = 4, 32, 2, 128
    torch.manual_seed(7)
    sel = torch.stack([torch.randperm(E)[:topk] for _ in range(rows)])
    sel[:, 0] = 0                               # one heavy expert (128 rows), the rest ~12 each
    y = torch.randn((rows, 8)).half()
    w = torch.rand((rows, topk)).half()

    def run(**ht):
        d, hl, hosts, cache = make(E = E, N = N, rows = rows, topk = topk, dispatch = "hybrid", zc_max = None, **ht)
        return d, hl, hosts, cache

    d, hl, hosts, cache = run(prefill_gpu_min = None)
    pend, sel_gpu = d.submit(hl, y, sel, w)
    assert d.prefill_gpu_min_last == D.PREFILL_GPU_MIN_BOOT     # nothing measured yet

    d, hl, hosts, cache = run(prefill_gpu_min = 40)
    d.submit(hl, y, sel, w)
    assert d.prefill_gpu_min_last == 40                         # fixed by the env

    # Measured: PCIe read 5000 us per expert, CPU 100 us per expert + 5 us per row
    d, hl, hosts, cache = run(prefill_gpu_min = None)
    d.zc_us, d.hit_us, d.cpu_fixed_us = 5000.0, 50.0, 30.0
    for q, h in enumerate(hosts):
        h.tel = [(1, 4 * 100 + 4 * 5, 4, 4, 1), (2, 8 * 100 + 160 * 5, 160, 8, 40)]
    d._update_lane_params()
    assert d.prefill_cost == pytest.approx((100.0, 5.0))
    pend, sel_gpu = d.submit(hl, y, sel, w)
    t = d.prefill_gpu_min_last
    assert t > int((sel == 0).sum())            # cheap CPU: even the heavy expert stays on the CPU
    assert int((sel_gpu == E).sum()) == rows * topk
    final = torch.zeros((rows, 8))
    lanes = torch.tensor([e % N for e in range(E)])
    assert torch.allclose(d.combine(hl, final, pend), fake_expert_sum(sel, w, y, lanes, E, N), atol = 1e-2)
    assert d.summary()["prefill_gpu_min"] == t

    # Slow CPU per row: the heavy expert is read over PCIe, the small ones stay on the CPU
    d.prefill_cost = (100.0, 200.0)
    pend, sel_gpu = d.submit(hl, y, sel, w)
    t = d.prefill_gpu_min_last
    assert 1 < t <= int((sel == 0).sum())
    assert bool((sel_gpu[:, 0] == 0).all())
    assert int((sel_gpu == E).sum()) > 0
    d.combine(hl, torch.zeros((rows, 8)), pend)


def test_prefill_threshold_remeasures_when_the_cpu_stays_idle():
    topk, E, N, rows = 4, 16, 2, 100
    d, hl, hosts, cache = make(E = E, N = N, rows = rows, topk = topk, dispatch = "hybrid", zc_max = None,
                               prefill_gpu_min = None, remeasure = 2)
    d.zc_us, d.hit_us, d.cpu_fixed_us = 10.0, 1.0, 1.0
    d.prefill_cost = (1e5, 1e4)                  # pessimistic: the CPU never wins
    sel = torch.stack([torch.randperm(E)[:topk] for _ in range(rows)])
    y, w = torch.randn((rows, 8)).half(), torch.rand((rows, topk)).half()
    got = []
    for _ in range(4):
        pend, _ = d.submit(hl, y, sel, w)
        d.combine(hl, torch.zeros((rows, 8)), pend)
        got.append(d.prefill_gpu_min_last)
    assert got[2] == D.PREFILL_GPU_MIN_BOOT and got[0] == got[1] == got[3] != D.PREFILL_GPU_MIN_BOOT


def test_mixed_rate_gpu_mask_picks_the_most_common_signature():
    # Uniform layer: no mask, the one signature
    assert D.mixed_rate_gpu_mask([[96, 96], [96, 96], [64, 64]]) == (None, (96, 96, 64))
    # Gate/up/down widths per expert: (96, 96, 64) x 3, (80, 80, 64) x 2, (96, 96, 80) x 1
    g = [96, 80, 96, 96, 80, 96]
    u = [96, 80, 96, 96, 80, 96]
    d = [64, 64, 64, 64, 64, 80]
    mask, sig = D.mixed_rate_gpu_mask([g, u, d])
    assert sig == (96, 96, 64)
    assert mask.tolist() == [True, False, True, True, False, False]
    # Ties go to the signature seen first
    mask, sig = D.mixed_rate_gpu_mask([[64, 80], [64, 80]])
    assert sig == (64, 64) and mask.tolist() == [True, False]


def test_eligible_accepts_mixed_rates_with_one_shape(monkeypatch):
    import sys, types
    stub = types.ModuleType(f"{H.dispatch.__package__.rsplit('.', 1)[0]}.modules.block_sparse_mlp")
    stub.BATCH_RECON = True
    monkeypatch.setitem(sys.modules, stub.__name__, stub)
    d, hl, hosts, cache = make(dispatch = "hybrid", zc_max = None)
    E, h, i = 6, 256, 128
    gu_w = [96, 80, 96, 96, 80, 96]
    d_w = [64, 64, 64, 64, 64, 80]
    shapes = {}
    for e in range(E):
        shapes[f"g{e}"] = [h // 16, i // 16, gu_w[e]]
        shapes[f"u{e}"] = [h // 16, i // 16, gu_w[e]]
        shapes[f"d{e}"] = [i // 16, h // 16, d_w[e]]

    class Stc:
        def list_tensors(self, key):
            return {key + ".trellis": dict(shape = shapes[key]), key + ".mul1": {}}

    class L:
        fkey = None
        def __init__(self, key):
            self.key = key
        def is_exl3_storage(self, key):
            return True

    m = types.SimpleNamespace(
        config = types.SimpleNamespace(stc = Stc(), infer_params = types.SimpleNamespace(no_reconstruct = False)),
        num_local_experts = E, num_experts = E, routing_first = None, routing_device = None, latent_in = None,
        activation_fn = "silu", gated = True, expert_size = h,
        gates = [L(f"g{e}") for e in range(E)], ups = [L(f"u{e}") for e in range(E)], downs = [L(f"d{e}") for e in range(E)])
    ok, why, dims = d.eligible(m)
    assert ok, why
    assert dims["g"] == (h, i, 6) and dims["u"] == (h, i, 6) and dims["d"] == (i, h, 4)
    assert dims["gpu"].tolist() == [True, False, True, True, False, False]
    # Uniform layer: no mask
    for e in range(E):
        shapes[f"g{e}"][2] = shapes[f"u{e}"][2] = 96
        shapes[f"d{e}"][2] = 64
    ok, why, dims = d.eligible(m)
    assert ok and dims["gpu"] is None
    # Different (k, n) within a projection stays ineligible
    shapes["d5"] = [i // 16, 2 * h // 16, 64]
    assert d.eligible(m)[:2] == (False, "experts of one projection differ in shape")


@pytest.mark.parametrize("rows", [1, 64])
def test_experts_without_a_home_always_take_cpu_lanes(rows):
    """CPU-only experts of a mixed-rate layer (no home copy): never a GPU lane, decode or prefill,
    and the warm start never installs them"""
    topk, E, N = 4, 16, 2
    d, hl, hosts, cache = make(E = E, N = N, rows = rows, topk = topk, dispatch = "hybrid", zc_max = 16,
                               prefill_gpu_min = 0)
    layer = hl.cache_layer
    eb = 1024
    home = torch.zeros((E * eb,), dtype = torch.uint8)
    gpu = np.array([e % 4 != 3 for e in range(E)])
    cache.set_layer_homes(layer, np.array([home.data_ptr() + e * eb if gpu[e] else 0 for e in range(E)]),
                          lambda e: home[e * eb:(e + 1) * eb] if gpu[e] else None)
    torch.manual_seed(3)
    sel = torch.stack([torch.randperm(E)[:topk] for _ in range(rows)])
    y, w = torch.randn((rows, 8)).half(), torch.rand((rows, topk)).half()
    for _ in range(3):
        pend, sel_gpu = d.submit(hl, y, sel, w)
        d.combine(hl, torch.zeros((rows, 8)), pend)
        on_gpu = sel_gpu[sel_gpu < E]
        assert on_gpu.numel() > 0
        assert bool(torch.from_numpy(gpu)[on_gpu].all())
        assert int((sel_gpu == E).sum()) >= int((~torch.from_numpy(gpu))[sel].sum())


def test_choose_prefill_gpu_min_counts_forced_cpu_work():
    counts = np.array([1, 2, 2, 3, 5, 8, 13, 21, 34, 55])
    kw = dict(t_gpu0 = 0.0, t_zc = 1000.0, t_fixed = 50.0, a = 100.0, b = 10.0, nodes = 2)
    t0 = D.choose_prefill_gpu_min(counts, **kw)
    # CPU already busy with experts it must take: fewer flexible experts go to it
    t1 = D.choose_prefill_gpu_min(counts, cpu0 = (20, 400), **kw)
    assert t1 <= t0
    c = np.sort(counts)
    base = 100.0 * 20 + 10.0 * 400
    worst = lambda k: max(50.0 + (base + 100 * k + 10 * c[:k].sum()) / 2, (c.size - k) * 1000.0)
    k = int((c < t1).sum())
    assert worst(k) == min(worst(j) for j in range(c.size + 1) if j in (0, c.size) or c[j] != c[j - 1])


@pytest.mark.parametrize("memory, expect", [("replicate", "numa_adaptive"), ("shard", "numa_local")])
def test_numa_policy_auto_spreads_replicated_experts(monkeypatch, memory, expect):
    monkeypatch.setenv("EXL3_NUMA_MEMORY_POLICY", memory)
    assert D.resolve_node_policy("auto") == expect
    assert D.resolve_node_policy("numa_balanced") == "numa_balanced"


def test_marginal_cost_fit_excludes_the_fixed_job_cost():
    # Node A: 1000 us fixed + 100 us per expert, seen only with 1-2 experts per job;
    # node B: 100 us fixed + 150 us per expert, seen with 6-8. Per-expert averages call A dearer
    # (~900 vs ~165 us), the marginal costs B (150 vs 100 us)
    a, b = D.MarginalCostFit(), D.MarginalCostFit()
    for e in (1, 2) * 20:
        a.add(e, 1000 + 100 * e)
    for e in (6, 7, 8) * 20:
        b.add(e, 100 + 150 * e)
    assert abs(a.marginal() - 100) < 1e-6 and abs(b.marginal() - 150) < 1e-6
    # One job size only: f and c cannot be separated
    c = D.MarginalCostFit()
    for _ in range(10):
        c.add(4, 800)
    assert c.marginal() is None


def test_decode_thread_tuner_settles_on_the_fastest_fraction():
    t = D.DecodeThreadTuner()
    # Token time by fraction: fastest at 0.75
    cost = {1.0: 0.080, 0.875: 0.070, 0.75: 0.050, 0.625: 0.055, 0.5: 0.060, 0.375: 0.075}
    now, applied = 0.0, [t.on_pass(True, 0.0)]
    per = (t.SETTLE + t.SAMPLES) * len(t.FRACTIONS) + 4
    for _ in range(per):
        now += cost[t.current()]
        f = t.on_pass(True, now)
        if f is not None:
            applied.append(f)
    assert applied[0] == 1.0 and t.best == 0.75 and t.current() == 0.75
    # A prefill pass or an idle gap does not count as a token
    assert t.on_pass(False, now + 0.01) is None
    assert t.on_pass(True, now + 5.0) is None
    # After the hold, only the winner's neighbours are searched again
    t.hold = 1
    t.on_pass(True, now + 5.05)
    t.on_pass(True, now + 5.1)
    assert t.cands == [0.875, 0.75, 0.625]


def test_decode_thread_tuner_keeps_the_larger_pool_within_margin():
    t = D.DecodeThreadTuner()
    # 0.875 is 2% faster than the full pool, inside the noise margin; 0.5 is slower
    t.results = {1.0: 0.0530, 0.875: 0.0520, 0.75: 0.0604, 0.5: 0.0575}
    assert t._pick() == 1.0
    # A clearly faster smaller share still wins
    t.results = {1.0: 0.080, 0.875: 0.070, 0.75: 0.050}
    assert t._pick() == 0.75


def test_decode_thread_tuner_fixed_and_off():
    assert D.DecodeThreadTuner(0.5).on_pass(True, 0.0) == 0.5
    off = D.DecodeThreadTuner(-1.0)
    assert off.on_pass(True, 0.0) is None and off.on_pass(True, 0.05) is None


def defer_setup(rows = 1, **ht):
    """Dispatcher with two hybrid layers on one device (the first may defer), every miss on the CPU"""
    d, hl, hosts, cache = make(rows = rows, dispatch = "hybrid", zc_max = 0, defer = True, **ht)
    hl2 = object.__new__(D.HybridLayer)
    hl2.device = torch.device("cpu")
    hl2.finalized, hl2.cache_layer = True, None
    d.layers = [hl, hl2]
    d.finalized = True
    d._link_layers()
    return d, hl, hl2, hosts, cache


def defer_module(**kw):
    m = type("M", (), {})()
    m.latent_out = m.routed_post_norm = None
    m.tp_reduce = False
    for k, v in kw.items():
        setattr(m, k, v)
    return m


def cpu_partial(hosts, rows):
    out = torch.zeros((rows, 8))
    for h in hosts:
        if int(h.counts[0]):
            out += h.out[:rows]
    return out


def test_defer_skips_collect_until_fold():
    d, hl, hl2, hosts, cache = defer_setup()
    assert hl.defer_next and not hl2.defer_next
    assert d.can_defer(hl, defer_module(), 1)
    pend, sel_gpu = d.submit(hl, torch.randn((1, 8)).half(), torch.tensor([[1, 2, 3, 4]]), torch.rand((1, 4)).half())
    rec = d.defer(hl, pend, 1)
    assert all(h.events == ["reserve", "publish"] for h in hosts)    # nothing waits for the CPU yet
    assert len(d.inflight) == 2                                      # still drained if interrupted
    x = torch.zeros((1, 1, 8))
    params = {"moe_fold": rec}
    D.fold_deferred(params, x)
    assert params["moe_fold"] is None and d.inflight == []
    assert all(h.events == ["reserve", "publish", "collect"] for h in hosts)
    assert torch.allclose(x.view(1, 8), cpu_partial(hosts, 1))
    D.fold_deferred(params, x)                                       # nothing pending: no-op
    assert all(h.events.count("collect") == 1 for h in hosts)


@pytest.mark.parametrize("rows", [65, 128])
def test_prefill_never_defers(rows):
    d, hl, hl2, hosts, cache = defer_setup(rows = rows)
    assert not d.can_defer(hl, defer_module(), rows)
    sel = torch.stack([torch.randperm(16)[:4] for _ in range(rows)])
    pend, _ = d.submit(hl, torch.randn((rows, 8)).half(), sel, torch.rand((rows, 4)).half())
    assert pend.kind == "prefill"


def test_interrupted_deferral_is_drained_by_begin_pass():
    d, hl, hl2, hosts, cache = defer_setup()
    pend, _ = d.submit(hl, torch.randn((1, 8)).half(), torch.tensor([[1, 2, 3, 4]]), torch.rand((1, 4)).half())
    d.defer(hl, pend, 1)                                             # the fold never runs
    d._update_lane_params = lambda: None
    d.begin_pass(None)
    assert d.inflight == []
    assert all(h.events.count("collect") == 1 for h in hosts)


def test_can_defer_rejections():
    d, hl, hl2, hosts, cache = defer_setup()
    m = defer_module()
    assert d.can_defer(hl, m, 1)
    assert not d.can_defer(hl, m, d.decode_rows + 1)                 # prefill-shaped call
    assert not d.can_defer(hl2, m, 1)                                # last hybrid layer
    for k, v in (("latent_out", object()), ("routed_post_norm", object()), ("tp_reduce", True),
                 ("mlp_resid_scalar", torch.ones(1)), ("mlp_post_norm", object()), ("layer_scalar_f", 0.5)):
        assert not d.can_defer(hl, defer_module(**{k: v}), 1), k
    hl2.device = torch.device("meta")                                # next hybrid layer elsewhere
    d._link_layers()
    assert not hl.defer_next
    hl2.device = torch.device("cpu")
    for h in hosts:
        h.num_slots = 1                                              # the next reserve would wait on it
    d._link_layers()
    assert not d.can_defer(hl, m, 1)
    for h in hosts:
        h.num_slots = 2
    d._link_layers()
    assert d.can_defer(hl, m, 1)
    D.HT.defer = False
    assert not d.can_defer(hl, m, 1)


def hc_site(x, y, post, comb):
    """HyperConnection.apply_ reference: x <- post ⊗ y + combᵀ x, (R, H, D)"""
    return post.unsqueeze(-1) * y.unsqueeze(-2) + torch.matmul(comb.transpose(-1, -2), x)


def test_hc_fold_is_exact():
    # Layer L's partial d enters the streams as post_m ⊗ d; deferred past the next block's two
    # sites (comb_a, comb_m) it must still land as (comb_mᵀ comb_aᵀ post_m) ⊗ d
    R, Hc, Dm = 2, 4, 8
    d, hl, hl2, hosts, cache = defer_setup(rows = R)
    torch.manual_seed(11)
    sel = torch.stack([torch.randperm(16)[:4] for _ in range(R)])
    pend, _ = d.submit(hl, torch.randn((R, 8)).half(), sel, torch.rand((R, 4)).half())
    rec = d.defer(hl, pend, R)
    part = cpu_partial(hosts, R)
    assert part.abs().sum() > 0

    x0 = torch.randn((R, Hc, Dm))
    post_m = torch.rand((R, Hc)) * 2
    y_a, y_m = torch.randn((R, Dm)), torch.randn((R, Dm))
    post_a, post_b = torch.rand((R, Hc)) * 2, torch.rand((R, Hc)) * 2
    comb_a, comb_m = torch.rand((R, Hc, Hc)), torch.rand((R, Hc, Hc))

    want = x0 + post_m.unsqueeze(-1) * part.unsqueeze(-2)
    want = hc_site(hc_site(want, y_a, post_a, comb_a), y_m, post_b, comb_m)

    rec.set_post(post_m.view(1, R, Hc))
    post_m.fill_(-1.0)                                               # mix scratch reused: v was copied
    x = hc_site(hc_site(x0.clone(), y_a, post_a, comb_a), y_m, post_b, comb_m)
    rec.carry(comb_a.view(1, R, Hc, Hc))
    rec.carry(comb_m)
    x = x.view(1, R, Hc, Dm).contiguous()
    params = {"moe_fold": rec}
    D.fold_deferred(params, x)
    assert torch.allclose(x.view(R, Hc, Dm), want, atol = 1e-5)
    assert d.inflight == [] and all(h.events.count("collect") == 1 for h in hosts)
    # Static buffers: a second deferral reuses them
    v_buf, d_buf = hl.defer_v, hl.defer_d
    pend, _ = d.submit(hl, torch.randn((R, 8)).half(), sel, torch.rand((R, 4)).half())
    rec = d.defer(hl, pend, R)
    rec.set_post(torch.rand((R, Hc)))
    rec.carry(comb_a)
    assert hl.defer_v is v_buf and hl.defer_d is d_buf
    assert rec.v.data_ptr() in (v_buf[0].data_ptr(), v_buf[1].data_ptr())
    d.fold(rec, torch.zeros((R, Hc, Dm)), rec.v)


class SlotHost(FakeHost):
    """FakeHost with a ring of slots (as MoeCpuHost) whose collect runs the fake expert function
    on what the slot holds: out[r] += w[r, k] * (local + 1 + 100 * node) * x[r] per pick >= 0"""

    def __init__(self, rows, topk, h, num_slots):
        super().__init__(rows * topk, rows, h, h)
        self.num_slots, self.next_slot, self.topk, self.h = num_slots, 0, topk, h
        self.ring = [dict(sel = torch.full((rows * topk,), -9, dtype = torch.int32),
                          x = torch.zeros((rows * h,), dtype = torch.half),
                          w = torch.zeros((rows * topk,), dtype = torch.half)) for _ in range(num_slots)]
        self.collected = []

    def fused_reserve(self, idx, rows, device):
        self.events.append("reserve")
        k = self.next_slot
        self.next_slot = (k + 1) % self.num_slots
        s = self.ring[k]
        slot = dict(sel_dev = s["sel"].data_ptr(), x_dev = s["x"].data_ptr(), w_dev = s["w"].data_ptr(),
                    x_cap = s["x"].numel(), sel_cap = s["sel"].numel(), **s)
        return dict(seq = 1, slot_idx = k, slot = slot, counts = self.counts, ho = self.h, dev = device, rows = rows)

    def fused_collect(self, r, final_2d):
        self.events.append("collect")
        k, rows = r["slot_idx"], r["rows"]
        picks = set()
        if int(self.counts[k]):
            s = self.ring[k]
            sel = s["sel"][:rows * self.topk].view(rows, self.topk)
            w = s["w"][:rows * self.topk].view(rows, self.topk)
            x = s["x"][:rows * self.h].view(rows, self.h)
            for i in range(rows):
                for j in range(self.topk):
                    e = int(sel[i, j])
                    if e >= 0:
                        final_2d[i] += float(w[i, j]) * (e + 1 + 100 * self.node) * x[i].float()
                        picks.add((i, j))
        self.collected.append(picks)


def split_setup(rows = 2, num_slots = 3, **ht):
    """defer_setup with slot-ring hosts that compute their partial from the staged slot"""
    d, hl, hl2, _, cache = defer_setup(rows = rows, **ht)
    hosts = [SlotHost(rows, 4, 8, num_slots) for _ in range(2)]
    for q, fh in enumerate(hosts):
        fh.node = q
    d.group.hosts = hosts
    d._link_layers()
    return d, hl, hl2, hosts, cache


def split_case(d, hl, rows, w):
    torch.manual_seed(5)
    y = torch.randn((rows, 8)).half()
    sel = torch.stack([torch.randperm(16)[:4] for _ in range(rows)])
    pend, sel_gpu = d.submit(hl, y, sel, w, True)
    cpu = {(i, j) for i in range(rows) for j in range(4) if int(sel_gpu[i, j]) == 16}
    assert cpu, "expected CPU-lane picks"

    def partial(picks):
        out = torch.zeros((rows, 8))
        for i, j in picks:
            e = int(sel[i, j])
            out[i] += float(w[i, j]) * (e // 2 + 1 + 100 * (e % 2)) * y[i].float()
        return out
    return pend, cpu, partial


def lowest(cpu, w, cap, min_w = None):
    want = set()
    for i in range(w.shape[0]):
        c = sorted((float(w[i, j]), j) for j in range(w.shape[1])
                   if (i, j) in cpu and (min_w is None or float(w[i, j]) < min_w))
        want |= {(i, j) for _, j in (c[:cap] if cap else c)}
    return want


@pytest.mark.parametrize("cap", [1, 2, 3])
def test_partial_defer_takes_the_lowest_weight_cpu_picks(cap):
    rows = 2
    d, hl, hl2, hosts, cache = split_setup(rows, defer_max = cap)
    assert d.can_defer(hl, defer_module(), rows)
    w = torch.tensor([[0.5, 0.125, 0.25, 0.375], [0.0625, 0.75, 0.1875, 0.3125]]).half()
    pend, cpu, partial = split_case(d, hl, rows, w)
    assert pend.deferred is not None and len(d.inflight) == 4
    assert all(h.events == ["reserve"] * 2 + ["publish"] * 2 for h in hosts)
    rec = d.defer(hl, pend, rows)
    # Immediate part: collected in the layer (combine), before the layer output
    f = torch.zeros((rows, 8))
    d.combine(hl, f, pend)
    assert all(h.events.count("collect") == 1 for h in hosts) and len(d.inflight) == 2
    deferred = lowest(cpu, w, cap)
    assert set().union(*(h.collected[0] for h in hosts)) == cpu - deferred
    assert torch.allclose(f, partial(cpu - deferred), atol = 1e-3)
    # Deferred part: folded at the end of the next block
    x = torch.zeros((1, rows, 8))
    D.fold_deferred({"moe_fold": rec}, x)
    assert d.inflight == [] and all(h.events.count("collect") == 2 for h in hosts)
    assert set().union(*(h.collected[1] for h in hosts)) == deferred
    assert torch.allclose(x.view(rows, 8), partial(deferred), atol = 1e-3)
    assert torch.allclose(f + x.view(rows, 8), partial(cpu), atol = 1e-3)


@pytest.mark.parametrize("cap", [0, 1])
def test_partial_defer_keeps_heavy_experts_immediate(cap):
    rows = 2
    d, hl, hl2, hosts, cache = split_setup(rows, defer_max = cap, defer_min_w = 0.3)
    w = torch.tensor([[0.5, 0.125, 0.25, 0.375], [0.0625, 0.75, 0.1875, 0.3125]]).half()
    pend, cpu, partial = split_case(d, hl, rows, w)
    rec = d.defer(hl, pend, rows)
    f = torch.zeros((rows, 8))
    d.combine(hl, f, pend)
    x = torch.zeros((rows, 8))
    d.fold(rec, x)
    deferred = lowest(cpu, w, cap, 0.3)
    assert all(float(w[i, j]) < 0.3 for i, j in deferred)
    assert torch.allclose(f, partial(cpu - deferred), atol = 1e-3)
    assert torch.allclose(x, partial(deferred), atol = 1e-3)


def test_defer_max_zero_defers_every_cpu_pick_in_one_job():
    rows = 2
    d, hl, hl2, hosts, cache = split_setup(rows)
    assert not D.HT.defer_partial()
    w = torch.rand((rows, 4)).half()
    pend, cpu, partial = split_case(d, hl, rows, w)
    assert pend.deferred is None and len(d.inflight) == 2
    assert all(h.events == ["reserve", "publish"] for h in hosts)
    rec = d.defer(hl, pend, rows)
    f = torch.zeros((rows, 8))
    d.combine(hl, f, pend)                                           # nothing left to collect
    assert f.abs().sum() == 0 and all(h.events.count("collect") == 0 for h in hosts)
    x = torch.zeros((rows, 8))
    d.fold(rec, x)
    assert torch.allclose(x, partial(cpu), atol = 1e-3)


def test_split_call_that_does_not_defer_collects_both_jobs():
    rows = 2
    d, hl, hl2, hosts, cache = split_setup(rows, defer_max = 1)
    w = torch.rand((rows, 4)).half()
    pend, cpu, partial = split_case(d, hl, rows, w)
    f = torch.zeros((rows, 8))
    d.combine(hl, f, pend)
    assert d.inflight == [] and all(h.events.count("collect") == 2 for h in hosts)
    assert torch.allclose(f, partial(cpu), atol = 1e-3)


def test_interrupted_partial_deferral_is_drained_by_begin_pass():
    rows = 2
    d, hl, hl2, hosts, cache = split_setup(rows, defer_max = 1)
    pend, cpu, partial = split_case(d, hl, rows, torch.rand((rows, 4)).half())
    d.defer(hl, pend, rows)                                          # neither collect nor fold runs
    d._update_lane_params = lambda: None
    d.begin_pass(None)
    assert d.inflight == [] and all(h.events.count("collect") == 2 for h in hosts)


def test_partial_defer_needs_three_slots():
    d, hl, hl2, hosts, cache = split_setup(1, num_slots = 2)
    m = defer_module()
    assert d.can_defer(hl, m, 1)                                     # whole-partial deferral: two
    D.HT.defer_max = 2
    assert not d.can_defer(hl, m, 1)
    D.HT.defer_max, D.HT.defer_min_w = 0, 0.1
    assert not d.can_defer(hl, m, 1)
    for h in hosts:
        h.num_slots = 3
    d._link_layers()
    assert d.can_defer(hl, m, 1)


# ---- prefill stream lane (EXL3_EXPERT_PREFILL_STREAM) -------------------------------------------

@pytest.mark.parametrize("mode, rows, cuda, want", [
    ("auto", 2047, True, False), ("auto", 2048, True, True), ("auto", 1 << 14, False, False),
    ("1", 65, True, True), ("1", 65, False, False), ("0", 1 << 14, True, False),
])
def test_prefill_stream_threshold(mode, rows, cuda, want):
    D.HT.stream, D.HT.stream_rows = mode, 2048
    assert D.prefill_stream_on(rows, cuda) is want


def test_stream_ring_shape_respects_slots_and_mib():
    D.HT.stream_slots, D.HT.stream_mb, D.HT.stream_bufs = 32, 512, 4
    assert D.stream_ring_shape(3 << 20) == (4, 8, 3 << 20)          # 32 slots, 96 MiB
    nb, per, stride = D.stream_ring_shape(17 << 20)                 # MiB-bound: 30 experts
    assert stride == 17 << 20 and nb * per * stride <= 512 << 20 and (nb, per) == (4, 7)
    assert D.stream_ring_shape((17 << 20) + 1)[2] % 256 == 0        # cache-slot alignment
    D.HT.stream_slots = 3
    assert D.stream_ring_shape(1 << 20) == (3, 1, 1 << 20)          # fewer experts than buffers
    assert D.stream_ring_shape(300 << 20) is None                   # cannot double-buffer


def test_select_stream_experts_takes_zero_copy_experts_hottest_first():
    E = 8
    lanes = np.full(ref.EMAX, ref.LANE_NONE, dtype = np.int32)
    lanes[[1, 2, 3, 5, 6]] = ref.LANE_ZC
    lanes[4] = ref.LANE_GPU                                         # resident: never streamed
    lanes[0] = 1                                                    # CPU node 1
    counts = np.array([90, 3, 40, 1, 70, 200, 0, 5], dtype = np.int32)
    assert D.select_stream_experts(lanes, counts, 1).tolist() == [5, 2, 1, 3]
    assert D.select_stream_experts(lanes, counts, 4).tolist() == [5, 2]
    assert D.select_stream_experts(lanes, counts, 1, max_rows = 64).tolist() == [2, 1, 3]
    assert D.select_stream_experts(lanes[:E], counts, 1000).size == 0


def test_mask_streamed_drops_picks_like_cpu_lanes():
    E = 8
    sel_gpu = torch.tensor([0, 3, 5, 8, 3, 7])                      # 8: already a CPU pick
    D.mask_streamed(sel_gpu, np.array([3, 7]), E)
    assert sel_gpu.tolist() == [0, 8, 5, 8, 8, 8]


def test_prefill_stream_lane_stays_off_on_the_reference_backend():
    d, hl, hosts, cache = make(E = 16, N = 2, rows = 80, dispatch = "hybrid", zc_max = None,
                               stream = "1", stream_rows = 1)
    sel = torch.stack([torch.randperm(16)[:4] for _ in range(80)])
    pend, sel_gpu = d.submit(hl, torch.randn((80, 8)).half(), sel, torch.rand((80, 4)).half())
    assert pend.kind == "prefill" and pend.stream is None


def test_worst_case_counts_the_stream_ring():
    d, hl, hosts, cache = make(dispatch = "hybrid", zc_max = None)
    mod = type("M", (), dict(batch_recon = None, prefill_worst_case_parts = lambda self, r, a: (0,)))()
    hl.module_ref = lambda: mod
    hl._worst = {}
    hl.expert_bytes = 3 << 20
    hl.device = torch.device("cuda", 0)                             # a device object, never used
    D.HT.stream, D.HT.stream_rows = "0", 2048
    D.HT.stream_slots, D.HT.stream_mb, D.HT.stream_bufs = 32, 512, 4
    base, small = d.worst_case(hl, 4096), d.worst_case(hl, 1024)
    D.HT.stream = "auto"
    with_ring = d.worst_case(hl, 4096)
    assert with_ring - base >= 32 * (3 << 20) + 4097 * hl.h * 4
    assert d.worst_case(hl, 1024) == small                          # below the threshold: no ring
    D.HT.stream = "0"
    assert d.worst_case(hl, 4096) == base                           # cached per setting


def test_run_staged_experts_tables_and_tiers(monkeypatch):
    # The shared batch body of the streamed prefill and the stream lane, with the kernels stubbed:
    # pointer tables address the staging buffer per (position, projection), the fused tier is
    # split into row tiles, heavier experts form batched groups with buffer pointers, and experts
    # above the batched tier's row cap come back to the caller
    for name in ("silu_mul", "gelu_mul", "silu_oai_mul", "relu2_mul", "relu_mul"):
        monkeypatch.setattr(H.ext, name, lambda *a: None, raising = False)
    launches, groups = [], []

    def exl3_moe(y, out, ec, tok, wts, b0, b1, b2, b3, act, Kg, Ku, Kd, *rest):
        tbl = [t.tolist() for t in rest[:9]]
        flags = rest[9:15]
        lo, hi, mt = rest[-3:]
        launches.append(dict(tbl = tbl, ec = ec.tolist(), tok = tok.tolist(), n_act = rest[16],
                             tier = (lo, hi, mt), flags = flags, K = (Kg, Ku, Kd)))

    monkeypatch.setattr(H.ext, "exl3_moe", exl3_moe, raising = False)

    class Recon:
        max_rows, cap, dims_u = 512, 4, (8, 16, 3)

        def run_group(self, y_ext, out_ext, tok_ext, w_ext, ids, starts, counts, ptrs = None):
            groups.append((ids, starts, counts, ptrs, y_ext.shape[0], tok_ext[-1].item()))

    rows, base, stride, off = 1000, 1 << 32, 4096, (0, 1024, 2048)
    counts = [5, 20, 40, 300, 900]
    starts = np.concatenate(([0], np.cumsum(counts)))[:-1].tolist()
    items = [(j, 10 + j, starts[j], c) for j, c in enumerate(counts)]
    tok = torch.arange(sum(counts)) % rows
    wts = torch.rand(sum(counts)).half()
    y = torch.randn((rows, 8)).half()
    out_ext = torch.zeros((rows + 1, 8))
    scales = lambda e: tuple(e * 100 + i for i in range(6))
    single = H.host.run_staged_experts(
        y, out_ext, items, tok, wts, base, stride, off, True, scales, (3, 4, 5), 0, 0.0,
        (False, True) * 3, 64, (None,) * 4, True, Recon(), {})
    assert single == [(4, 14)]
    assert [l["tier"] for l in launches] == [(33, 64, 64), (17, 32, 32), (1, 16, 16)]
    assert [l["n_act"] for l in launches] == [1, 1, 1]
    t = launches[0]
    assert t["ec"] == counts + [0] and t["K"] == (3, 4, 5) and t["flags"] == (False, True) * 3
    assert t["tbl"][0] == [base + j * stride for j in range(5)]
    assert t["tbl"][3] == [base + j * stride + 1024 for j in range(5)]
    assert t["tbl"][6] == [base + j * stride + 2048 for j in range(5)]
    assert t["tbl"][1] == [(10 + j) * 100 for j in range(5)] and t["tbl"][8] == [(10 + j) * 100 + 5 for j in range(5)]
    assert t["tok"] == tok.tolist()
    ids, gs, gc, ptrs, ny, sentinel = groups[0]
    assert len(groups) == 1 and ids == [13] and gs == [starts[3]] and gc == [300]
    assert ptrs == ([base + 3 * stride], [base + 3 * stride + 1024], [base + 3 * stride + 2048])
    assert ny == rows + 1 and sentinel == rows                    # padded inputs and the sink row

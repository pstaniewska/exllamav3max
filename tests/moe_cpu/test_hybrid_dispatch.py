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


@pytest.mark.parametrize("rows", [1, 8, 9, 31, 32, 64, 128, 256, 512])
def test_row_boundaries_pick_the_path_and_fold_cpu_partials(rows):
    # rows < prefill_rows (32) and <= slot capacity: decode path (slot protocol); otherwise the
    # prefill path (CPU rows compacted per node, folded in after the GPU tiers). Admission only
    # for calls with <= EXL3_EXPERT_CACHE_ADMIT_MAX picks
    topk, E, N = 4, 32, 2
    d, hl, hosts, cache = make(E = E, N = N, rows = rows, topk = topk, dispatch = "hybrid", zc_max = 1)
    hl.hi = hl.h = 8
    torch.manual_seed(rows)
    sel = torch.stack([torch.randperm(E)[:topk] for _ in range(rows)])
    y = torch.randn((rows, 8)).half()
    w = torch.rand((rows, topk)).half()
    admits0 = cache.stats()["admits"]
    pend, sel_gpu = d.submit(hl, y, sel, w)
    decode = rows < d.prefill_rows and rows <= d.cap_rows
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
    d, hl, hosts, cache = make(E = 16, N = 2, rows = 40, dispatch = "gpu_exact", zc_max = None)
    sel = torch.stack([torch.randperm(16)[:4] for _ in range(40)])
    pend, sel_gpu = d.submit(hl, torch.randn((40, 8)).half(), sel, torch.rand((40, 4)).half())
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

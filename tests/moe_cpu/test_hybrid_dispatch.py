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
        slot = dict(sel_dev = self.sel.data_ptr(), x_dev = self.x.data_ptr(), w_dev = self.w.data_ptr())
        return dict(seq = 1, slot_idx = 0, slot = slot, counts = self.counts, ho = self.out.shape[1], dev = device, rows = rows)

    def fused_publish(self, r):
        self.events.append("publish")

    def fused_collect(self, r, final_2d):
        self.events.append("collect")
        if int(self.counts[0]):
            final_2d += self.out[:final_2d.shape[0], :final_2d.shape[1]]

    def read_telemetry(self):
        return self.tel


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

    d = object.__new__(D.HybridDispatcher)
    hosts = [FakeHost(rows * topk, rows, h, h) for _ in range(N)]
    d.group = type("G", (), dict(hosts = hosts, pref = -1, num_nodes = N))()
    d.node_policy = 0
    d.cap_rows, d.prefill_rows = 64, 32
    d.node_cost = torch.ones((N,), dtype = torch.float)
    d.cpu_us = [None] * N
    d.tel_seen = [dict() for _ in range(N)]
    d.zc_us = None
    d.cpu_frac = 0.5
    hl = object.__new__(D.HybridLayer)
    hl.device = torch.device("cpu")
    hl.num_experts, hl.topk, hl.h, hl.hi, hl.ho = E, topk, h, h, h
    hl.cache_layer = layer
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


def test_cpu_share_from_measured_costs_only():
    d, hl, hosts, cache = make(dispatch = "hybrid", zc_max = None)
    d._update_lane_params()
    assert d.cpu_frac == 0.5                    # nothing measured: initial share
    d.zc_us = 200.0
    hosts[0].tel = [(5, 400, 4, 4)]             # 100 us per assignment
    hosts[1].tel = [(9, 200, 2, 2)]             # 100 us per assignment
    d._update_lane_params()
    t_cpu = 1.0 / (1.0 / 100 + 1.0 / 100)
    assert d.cpu_frac == pytest.approx(200 / (200 + t_cpu))
    before = d.cpu_us[:]
    d._update_lane_params()                     # same records again: not recounted
    assert d.cpu_us == before

"""
Hybrid MoE expert cache (model/moe_expert_cache.py) and its lane / replacement policy, run on the
CPU against the reference implementation of the kernels (model/moe_hybrid_ref.py, the documented
mirror of exllamav3_ext/moe_hybrid.cu). Pointer tables, slots, homes and worker slot sections are
real host memory here, so every address the policy writes is checked by reading it back.

Covered: registration and table ownership, CLOCK admission with the in-call pin, gather copies,
eviction back to homes, lane assignment (GPU quota, CPU fraction, node policies, prefill
threshold, experts without a GPU-readable home), budget sizing against a memory oracle,
shrinking to the floor and regrowing, size classes sharing one budget, unregistration, warm start.
Not covered here (GPU host): the CUDA kernels themselves, see test_hybrid_kernels_parity.
"""

import json

import numpy as np
import pytest
import torch

from testlib import hybrid_isolated

pytestmark = pytest.mark.nogpu

H = hybrid_isolated.load()
ref = H.ref
EMAX = ref.EMAX


def tuning(**kw):
    T = H.cache.CacheTuning()
    T.chunk_slots = 4
    T.lcap = 8
    T.gmax = 16
    T.reserve_bytes = 0
    T.min_free_bytes = 0
    T.grow_hysteresis = 0
    T.limit_bytes = None
    T.policy = "clock"
    T.warm = None
    T.save_stats = None
    T.debug = False
    for k, v in kw.items():
        setattr(T, k, v)
    return T


class Mem:
    """Memory oracle: a device with `free` bytes before the cache, which the cache's own chunks
    consume"""

    def __init__(self, free, total = None):
        self.free = free
        self.total = total or free
        self.cache = None

    def __call__(self):
        return self.free - (self.cache.cache_bytes() if self.cache else 0), self.total


def make_cache(free = 1 << 20, **kw):
    mem = Mem(free)
    c = H.cache.ExpertCache(torch.device("cpu"), backend = "ref", mem_info = mem, tuning = tuning(**kw))
    mem.cache = c
    return c, mem


class FakeLayer:
    def __init__(self, cache, uid, E = 16, eb = 1024, mapped = None, gated = True):
        self.E, self.eb = E, eb
        self.proj_off = (0, eb // 4, eb // 2) if gated else (0, 0, eb // 2)
        self.tg = torch.zeros(E, dtype = torch.long)
        self.tu = torch.zeros(E, dtype = torch.long)
        self.td = torch.zeros(E, dtype = torch.long)
        self.gu = torch.zeros(E * 2, dtype = torch.long)
        tables = [(self.tg, 0, 1, 0), (self.tu, 0, 1, 1), (self.td, 0, 1, 2), (self.gu, 0, 2, 0), (self.gu, 1, 2, 1)]
        self.layer = cache.register_layer(uid, uid, E, eb, self.proj_off, tables)
        g = torch.Generator().manual_seed(hash(uid) & 0xFFFF)
        self.home = torch.randint(0, 255, (E * eb,), dtype = torch.uint8, generator = g)
        self.homes = np.array([self.home.data_ptr() + e * eb for e in range(E)], dtype = np.int64)
        if mapped is not None:
            self.homes[~np.asarray(mapped)] = 0
        cache.set_layer_homes(self.layer, self.homes, lambda e: self.home[e * eb: (e + 1) * eb])

    def rows(self, e):
        return int(self.tg[e]), int(self.tu[e]), int(self.td[e]), int(self.gu[2 * e]), int(self.gu[2 * e + 1])

    def expect(self, base):
        g, u, d = (base + o for o in self.proj_off)
        return g, u, d, g, u


class Nodes:
    """Worker slot sections of N nodes for decode-mode issue calls"""

    def __init__(self, N, n, rows, hi):
        self.sel = [torch.full((n,), -7, dtype = torch.int32) for _ in range(N)]
        self.x = [torch.zeros((rows * hi,), dtype = torch.half) for _ in range(N)]
        self.w = [torch.zeros((n,), dtype = torch.half) for _ in range(N)]
        self.count = [torch.full((8,), -1, dtype = torch.int32) for _ in range(N)]
        t = torch.zeros((ref.ND_ROWS, N), dtype = torch.long)
        for i in range(N):
            t[ref.ND_SEL, i] = self.sel[i].data_ptr()
            t[ref.ND_X, i] = self.x[i].data_ptr()
            t[ref.ND_W, i] = self.w[i].data_ptr()
            t[ref.ND_COUNT, i] = self.count[i].data_ptr()
            t[ref.ND_SLOT, i] = 0
            t[ref.ND_XCAP, i] = rows * hi           # real capacities of the buffers above
            t[ref.ND_SELCAP, i] = n
        self.t = t
        self.cost = torch.ones((N,), dtype = torch.float)


def issue(cache, fl, sel, rows = None, hi = 8, nodes = None, prefill = 0, admit = 1, gpu_max = -1,
          cpu_frac = 0.0, gmin = 0, policy = 0, pref = -1, h = 8, lane_cost = None):
    sel = torch.tensor(sel, dtype = torch.long).view(-1)
    n = sel.numel()
    rows = rows or n
    topk = n // rows
    y = torch.randn((rows, h)).half()
    w = (torch.rand((n,)) + 0.1).half()
    sel_gpu = torch.empty((n,), dtype = torch.long)
    lane = torch.empty((EMAX,), dtype = torch.int32)
    N = nodes.t.shape[1] if nodes is not None else 0
    cpu_sel = torch.empty((max(N, 1), n), dtype = torch.int32) if prefill else None
    nt = nodes.t if nodes is not None else torch.zeros((ref.ND_ROWS, 0), dtype = torch.long)
    cost = nodes.cost if nodes is not None else torch.ones((0,), dtype = torch.float)
    lc = torch.tensor(lane_cost, dtype = torch.float) if lane_cost is not None else None
    cache.issue(fl.layer, sel, y, w, sel_gpu, lane, cpu_sel, nt, cost, hi, prefill, admit, gpu_max,
                cpu_frac, gmin, policy, pref, lc)
    return dict(sel = sel, sel_gpu = sel_gpu, lane = lane[:fl.E].clone(), y = y, w = w, cpu_sel = cpu_sel, topk = topk)


def check_invariants(cache, layers):
    st = cache.ref_state
    for s in range(cache.scap):
        o = int(st.owner[s])
        if o >= 0:
            assert int(st.slotof[o]) == s, "owner/slotof disagree"
            assert int(st.pin[s]) >= 0, "disabled slot has a tenant"
    for fl in layers:
        L = fl.layer.L
        for e in range(fl.E):
            s = int(st.slotof[L * EMAX + e])
            if s >= 0:
                assert int(st.owner[s]) == L * EMAX + e
                base = int(st.slot_base[s])
            else:
                base = int(fl.homes[e]) or int(st.reg_bounce[L])
            assert fl.rows(e) == fl.expect(base), f"table rows of expert {e} point elsewhere"


def slot_bytes(cache, fl, e):
    st = cache.ref_state
    s = int(st.slotof[fl.layer.L * EMAX + e])
    c = fl.layer.cls
    return cache._slot_tensor(c, s)[:fl.eb]


# ---------------------------------------------------------------------------------------------

def test_registration_points_rows_home_then_bounce():
    c, _ = make_cache()
    fl = FakeLayer(c, "text:l0", mapped = [e % 3 != 0 for e in range(16)])
    check_invariants(c, [fl])
    bounce = int(c.ref_state.reg_bounce[fl.layer.L])
    assert fl.rows(0) == fl.expect(bounce)
    assert fl.rows(1) == fl.expect(int(fl.homes[1]))
    with pytest.raises(RuntimeError):
        c.register_layer("text:l0", "l0", 16, 1024, (0, 256, 512), [])


def test_sizing_respects_free_memory_floor_and_limit():
    c, mem = make_cache(free = 10 * 4 * 1024 + 100, reserve_bytes = 2 * 4 * 1024)
    fl = FakeLayer(c, "a")
    c.begin_pass(1, 32)
    chunk = 4 * 1024
    assert c.cache_bytes() == 8 * chunk          # (free - floor) // chunk
    c2, _ = make_cache(free = 1 << 20, limit_bytes = 3 * 4 * 1024)
    FakeLayer(c2, "b")
    c2.begin_pass(1, 32)
    assert c2.cache_bytes() == 3 * chunk


def test_admission_copies_home_bytes_and_repoints():
    c, _ = make_cache()
    fl = FakeLayer(c, "a")
    c.begin_pass(1, 32)
    r = issue(c, fl, [3, 5, 7, 9])
    assert (r["sel_gpu"] == r["sel"]).all()
    for e in (3, 5, 7, 9):
        assert int(r["lane"][e]) == ref.LANE_GPU
        assert torch.equal(slot_bytes(c, fl, e), fl.home[e * fl.eb: (e + 1) * fl.eb])
    assert c.stats()["admits"] == 4
    check_invariants(c, [fl])
    # Second call: hits, no new admissions
    r = issue(c, fl, [3, 5])
    st = c.stats()
    assert st["admits"] == 4 and st["hits"] == 2


def test_clock_never_evicts_slots_used_in_the_same_call():
    c, mem = make_cache(free = 4 * 1024)        # exactly one chunk: 4 slots
    fl = FakeLayer(c, "a")
    c.begin_pass(1, 32)
    issue(c, fl, [0, 1, 2, 3])
    # 4 resident; a call hitting 0 and 1 and missing 4 and 5 may only evict 2 and 3
    r = issue(c, fl, [0, 1, 4, 5])
    st = c.ref_state
    resident = {e for e in range(fl.E) if int(st.slotof[fl.layer.L * EMAX + e]) >= 0}
    assert {0, 1} <= resident and {4, 5} <= resident and not ({2, 3} & resident)
    check_invariants(c, [fl])
    # All slots pinned by the call itself: misses stay zero-copy
    r = issue(c, fl, [0, 1, 4, 5, 6])
    assert int(r["lane"][6]) == ref.LANE_ZC and c.stats()["no_victim"] >= 1
    check_invariants(c, [fl])


def test_random_routing_keeps_invariants_and_hits():
    c, _ = make_cache(free = 6 * 4 * 1024)
    a, b = FakeLayer(c, "a", E = 32), FakeLayer(c, "b", E = 32)
    c.begin_pass(1, 32)
    rng = np.random.default_rng(0)
    p = 1.0 / np.arange(1, 33) ** 1.2
    p /= p.sum()
    for step in range(300):
        fl = a if step % 2 == 0 else b
        rows = int(rng.integers(1, 4))
        sel = np.concatenate([rng.choice(32, size = 4, replace = False, p = p) for _ in range(rows)])
        r = issue(c, fl, sel.tolist(), rows = rows)
        for e in set(sel.tolist()):
            assert int(r["lane"][e]) in (ref.LANE_GPU, ref.LANE_ZC)
        check_invariants(c, [a, b])
    st = c.stats()
    assert st["hits"] > st["admits"] > 0
    assert st["no_lane"] == 0


def test_cpu_lanes_quota_and_staging():
    c, _ = make_cache()
    fl = FakeLayer(c, "a")
    shard = np.full((2, fl.E), -1, dtype = np.int32)
    for e in range(fl.E):
        shard[e % 2, e] = e // 2
    c.set_layer_cpuloc(fl.layer, shard)
    c.begin_pass(1, 32)
    nodes = Nodes(2, 4, 1, 8)
    r = issue(c, fl, [2, 3, 4, 5], rows = 1, nodes = nodes, admit = 0, gpu_max = 1)
    lanes = {e: int(r["lane"][e]) for e in (2, 3, 4, 5)}
    assert sum(1 for v in lanes.values() if v in (ref.LANE_ZC, ref.LANE_GPU)) == 1
    for e, v in lanes.items():
        if v >= 0:
            assert v == e % 2                    # shard owner
    for i in range(4):
        e = int(r["sel"][i])
        if lanes[e] >= 0:
            assert int(r["sel_gpu"][i]) == fl.E   # masked for the GPU
            assert int(nodes.sel[lanes[e]][i]) == e // 2
        for q in range(2):
            if lanes[e] != q:
                assert int(nodes.sel[q][i]) == -1
    for q in range(2):
        active = any(v == q for v in lanes.values())
        assert int(nodes.count[q][0]) == int(active)
        if active:
            assert torch.equal(nodes.w[q], r["w"])
            assert torch.equal(nodes.x[q].view(1, 8), r["y"])


def _cpu_layer(c, E = 16):
    fl = FakeLayer(c, "a", E = E)
    c.set_layer_cpuloc(fl.layer, np.arange(fl.E, dtype = np.int32)[None, :])   # every expert on node 0
    return fl


def test_cpu_lane_miss_is_admitted_in_the_background():
    # gpu_max = 0 sends every flexible miss to the CPU. The CPU still computes this call's picks
    # (lane and masking unchanged), but a free slot takes the expert for the next calls
    c, _ = make_cache(bg_admit = 1)
    fl = _cpu_layer(c)
    c.begin_pass(1, 32)
    nodes = Nodes(1, 2, 1, 8)
    r = issue(c, fl, [3, 4], rows = 1, nodes = nodes, gpu_max = 0)
    assert int(r["lane"][3]) == 0 and int(r["lane"][4]) == 0
    assert (r["sel_gpu"] == fl.E).all()
    st = c.stats()
    assert st["bg_admits"] == 1 and st["admits"] == 1        # bounded: one per call
    resident = [e for e in (3, 4) if int(c.ref_state.slotof[fl.layer.L * EMAX + e]) >= 0]
    assert len(resident) == 1
    e = resident[0]
    assert torch.equal(slot_bytes(c, fl, e), fl.home[e * fl.eb: (e + 1) * fl.eb])
    check_invariants(c, [fl])
    # Next call: the admitted expert is a hit
    r = issue(c, fl, [e], rows = 1, nodes = nodes, gpu_max = 0)
    assert int(r["lane"][e]) == ref.LANE_GPU and c.stats()["hits"] == 1


def test_background_admission_never_displaces_a_hotter_expert():
    c, _ = make_cache(free = 4 * 1024, bg_admit = 1)        # one chunk: 4 slots
    fl = _cpu_layer(c)
    c.begin_pass(1, 32)
    nodes = Nodes(1, 4, 1, 8)
    for _ in range(8):                                       # 0..3 resident and hot
        issue(c, fl, [0, 1, 2, 3], rows = 1, nodes = nodes)
    st = c.ref_state
    assert all(int(st.slotof[fl.layer.L * EMAX + e]) >= 0 for e in range(4))
    before = c.stats()
    for _ in range(3):                                       # a cold expert on the CPU lane
        r = issue(c, fl, [9], rows = 1, nodes = Nodes(1, 1, 1, 8), gpu_max = 0)
        assert int(r["lane"][9]) == 0
    after = c.stats()
    assert after["bg_admits"] == before["bg_admits"]
    assert after["bg_rejects"] > before["bg_rejects"]
    assert int(st.slotof[fl.layer.L * EMAX + 9]) < 0
    assert all(int(st.slotof[fl.layer.L * EMAX + e]) >= 0 for e in range(4))
    check_invariants(c, [fl])


def test_background_admission_replaces_a_colder_expert():
    c, _ = make_cache(free = 4 * 1024, bg_admit = 1)
    fl = _cpu_layer(c)
    c.begin_pass(1, 32)
    issue(c, fl, [0, 1, 2, 3], rows = 1, nodes = Nodes(1, 4, 1, 8))    # resident, used once
    for _ in range(6):                                                  # expert 9 gets hot on the CPU lane
        issue(c, fl, [9], rows = 1, nodes = Nodes(1, 1, 1, 8), gpu_max = 0)
    st = c.ref_state
    assert int(st.slotof[fl.layer.L * EMAX + 9]) >= 0
    assert c.stats()["bg_admits"] >= 1
    check_invariants(c, [fl])


def test_bg_admit_zero_keeps_cpu_lane_misses_out_of_the_cache():
    c, _ = make_cache(bg_admit = 0)
    fl = _cpu_layer(c)
    c.begin_pass(1, 32)
    for _ in range(4):
        issue(c, fl, [3, 4], rows = 1, nodes = Nodes(1, 2, 1, 8), gpu_max = 0)
    st = c.stats()
    assert st["admits"] == 0 and st["bg_admits"] == 0 and st["bg_rejects"] == 0
    check_invariants(c, [fl])


def _hot_then_cold_on_gpu_lane(gpu_gate):
    c, _ = make_cache(free = 4 * 1024, bg_admit = 0, gpu_gate = gpu_gate)   # one chunk: 4 slots
    fl = FakeLayer(c, "a")
    c.begin_pass(1, 32)
    for _ in range(8):                                       # 0..3 resident and hot
        issue(c, fl, [0, 1, 2, 3])
    r = issue(c, fl, [9])                                    # a cold GPU-lane miss
    return c, fl, r


def test_gpu_gate_runs_a_colder_miss_zero_copy_without_evicting():
    c, fl, r = _hot_then_cold_on_gpu_lane(1)
    st = c.ref_state
    assert int(r["lane"][9]) == ref.LANE_ZC
    assert int(st.slotof[fl.layer.L * EMAX + 9]) < 0
    assert all(int(st.slotof[fl.layer.L * EMAX + e]) >= 0 for e in range(4))
    assert c.stats()["gpu_rejects"] == 1
    check_invariants(c, [fl])


def test_gpu_gate_zero_admits_every_gpu_lane_miss():
    c, fl, r = _hot_then_cold_on_gpu_lane(0)
    assert int(r["lane"][9]) == ref.LANE_GPU
    assert int(c.ref_state.slotof[fl.layer.L * EMAX + 9]) >= 0
    assert c.stats()["gpu_rejects"] == 0
    check_invariants(c, [fl])


def test_cpu_frac_and_unmapped_homes():
    c, _ = make_cache()
    mapped = [e != 6 for e in range(16)]
    fl = FakeLayer(c, "a", mapped = mapped)
    loc = np.full((1, fl.E), -1, dtype = np.int32)
    loc[0] = np.arange(fl.E)
    c.set_layer_cpuloc(fl.layer, loc)
    c.begin_pass(1, 32)
    nodes = Nodes(1, 4, 1, 8)
    r = issue(c, fl, [6, 7, 8, 9], rows = 1, nodes = nodes, admit = 0, cpu_frac = 1.0)
    assert all(int(r["lane"][e]) == 0 for e in (6, 7, 8, 9))
    r = issue(c, fl, [6, 7, 8, 9], rows = 1, nodes = nodes, admit = 0, cpu_frac = 0.0)
    assert int(r["lane"][6]) == 0                # no GPU-readable home: CPU regardless
    assert all(int(r["lane"][e]) == ref.LANE_ZC for e in (7, 8, 9))


def test_no_lane_counts_when_neither_home_nor_worker():
    c, _ = make_cache()
    fl = FakeLayer(c, "a", mapped = [e != 2 for e in range(16)])
    c.begin_pass(1, 32)
    r = issue(c, fl, [2, 3], admit = 0)
    assert int(r["sel_gpu"][0]) == fl.E and c.stats()["no_lane"] == 1
    with pytest.raises(RuntimeError):
        c.idle_maintenance()


def test_node_policies_balance_replicas():
    c, _ = make_cache()
    fl = FakeLayer(c, "a")
    loc = np.tile(np.arange(fl.E, dtype = np.int32), (2, 1))       # replicated on both nodes
    c.set_layer_cpuloc(fl.layer, loc)
    c.begin_pass(1, 32)
    nodes = Nodes(2, 4, 1, 8)
    r = issue(c, fl, [1, 2, 3, 4], nodes = nodes, admit = 0, cpu_frac = 1.0, policy = 0, pref = 1, rows = 1)
    assert all(int(r["lane"][e]) == 1 for e in (1, 2, 3, 4))          # local: preferred node
    r = issue(c, fl, [1, 2, 3, 4], nodes = nodes, admit = 0, cpu_frac = 1.0, policy = 1, rows = 1)
    assert sorted(int(r["lane"][e]) for e in (1, 2, 3, 4)) == [0, 0, 1, 1]
    nodes.cost[:] = torch.tensor([3.0, 1.0])
    r = issue(c, fl, [1, 2, 3, 4], nodes = nodes, admit = 0, cpu_frac = 1.0, policy = 2, rows = 1)
    assert sum(int(r["lane"][e]) == 1 for e in (1, 2, 3, 4)) == 3   # cheaper node takes more


def test_prefill_mode_threshold_no_admission():
    c, _ = make_cache()
    fl = FakeLayer(c, "a")
    loc = np.arange(fl.E, dtype = np.int32)[None]
    c.set_layer_cpuloc(fl.layer, loc)
    c.begin_pass(1, 32)
    nodes = Nodes(1, 1, 1, 8)
    sel = [1] * 10 + [2] * 3 + [3] * 3
    r = issue(c, fl, sel, rows = 8, nodes = nodes, prefill = 1, gmin = 5)
    assert int(r["lane"][1]) == ref.LANE_ZC and int(r["lane"][2]) == 0 and int(r["lane"][3]) == 0
    assert c.stats()["admits"] == 0
    cs = r["cpu_sel"][0]
    for i, e in enumerate(sel):
        assert int(cs[i]) == (e if e != 1 else -1)


def test_shrink_and_regrow_follow_the_floor():
    chunk = 4 * 1024
    c, mem = make_cache(free = 6 * chunk)
    fl = FakeLayer(c, "a", E = 32)
    c.begin_pass(1, 32)
    assert c.cache_bytes() == 6 * chunk
    issue(c, fl, list(range(16)), rows = 4)
    # Something else takes 3 chunks' worth: the next prefill pass gives them back
    mem.free -= 3 * chunk
    c.begin_pass(64, 32)
    assert c.cache_bytes() == 3 * chunk
    check_invariants(c, [fl])
    st = c.ref_state
    disabled = [s for s in range(c.scap) if int(st.pin[s]) < 0]
    assert len(disabled) >= 3 * 4
    for _ in range(20):
        issue(c, fl, list(range(16, 32)), rows = 4)
        for s in disabled:
            assert int(st.owner[s]) < 0
    check_invariants(c, [fl])
    mem.free += 3 * chunk
    c.idle_maintenance()
    assert c.cache_bytes() == 6 * chunk


def test_release_all_then_regrow_at_next_pass():
    c, mem = make_cache(free = 4 * 4 * 1024)
    fl = FakeLayer(c, "a")
    c.begin_pass(1, 32)
    issue(c, fl, [0, 1, 2, 3])
    c.release_all()
    assert c.cache_bytes() == 0
    check_invariants(c, [fl])
    assert all(fl.rows(e) == fl.expect(int(fl.homes[e])) for e in range(fl.E))
    c.begin_pass(1, 32)
    assert c.cache_bytes() == 4 * 4 * 1024


def test_size_classes_share_one_budget():
    c, mem = make_cache(free = 12 * 4 * 1024)
    small = FakeLayer(c, "small", E = 16, eb = 1024)
    big = FakeLayer(c, "big", E = 16, eb = 2048)
    c.begin_pass(1, 32)
    total = c.cache_bytes()
    assert total <= 12 * 4 * 1024
    by = {cl.slot_bytes: cl.allocated_bytes() for cl in c.classes}
    assert by[1024] > 0 and by[2048] > 0
    issue(c, small, [1, 2])
    issue(c, big, [1, 2])
    st = c.ref_state
    for fl in (small, big):
        cl = fl.layer.cls
        for e in (1, 2):
            s = int(st.slotof[fl.layer.L * EMAX + e])
            assert cl.lo <= s < cl.lo + cl.cap
    check_invariants(c, [small, big])


def test_unregister_clears_registry_and_reuses_index():
    c, _ = make_cache()
    a = FakeLayer(c, "a")
    b = FakeLayer(c, "b")
    c.begin_pass(1, 32)
    issue(c, a, [0, 1, 2, 3])
    L = a.layer.L
    c.unregister_layer(a.layer)
    st = c.ref_state
    assert all(int(v) == 0 for v in st.reg_tables[L * ref.MAX_TABLES: (L + 1) * ref.MAX_TABLES])
    assert all(int(st.owner[s]) < 0 or int(st.owner[s]) // EMAX != L for s in range(c.scap))
    for _ in range(10):
        issue(c, b, [4, 5, 6, 7])
    check_invariants(c, [b])
    a2 = FakeLayer(c, "a2")
    assert a2.layer.L == L


def test_warm_start_installs_ranked_experts(tmp_path):
    stats = {"a": [0.0] * 16}
    stats["a"][5] = 10.0
    stats["a"][9] = 7.0
    p = tmp_path / "warm.json"
    p.write_text(json.dumps(stats))
    c, _ = make_cache(warm = str(p))
    fl = FakeLayer(c, "a")
    c.begin_pass(1, 32)
    st = c.ref_state
    for e in (5, 9):
        assert int(st.slotof[fl.layer.L * EMAX + e]) >= 0
        assert torch.equal(slot_bytes(c, fl, e), fl.home[e * fl.eb: (e + 1) * fl.eb])
    check_invariants(c, [fl])
    out = tmp_path / "save.json"
    c.save_stats(str(out))
    saved = json.loads(out.read_text())
    assert saved["a"][5] >= 10.0


def test_policy_none_never_admits():
    c, _ = make_cache(policy = "none")
    fl = FakeLayer(c, "a")
    c.begin_pass(1, 32)
    r = issue(c, fl, [1, 2])
    assert c.cache_bytes() == 0 and c.stats()["admits"] == 0
    assert all(int(r["lane"][e]) == ref.LANE_ZC for e in (1, 2))


def test_register_mapped_home_fallback(monkeypatch):
    h = H.host
    state = h._HOME_PIN
    state.update(registered = 0, fail_run = 0, disabled = False, logged = 0, notices = set())
    calls = []

    def fail(ptr, size, flags = 0):
        calls.append(ptr)
        raise RuntimeError("cudaHostRegister failed: out of memory")

    monkeypatch.setattr(h, "cuda_host_register", fail)
    assert h.register_mapped_home(0x1000, 1 << 30, "t") == 0
    assert not state["disabled"]
    assert h.register_mapped_home(0x2000, 1 << 30, "t") == 0
    assert state["disabled"]
    assert h.register_mapped_home(0x3000, 1 << 30, "t") == 0 and len(calls) == 2

    state.update(registered = 0, fail_run = 0, disabled = False, logged = 0, notices = set())
    monkeypatch.setattr(h, "cuda_host_register", lambda ptr, size, flags = 0: None)
    monkeypatch.setattr(h, "cuda_host_get_device_pointer", lambda ptr: ptr + 0x7000_0000_0000)
    monkeypatch.setattr(h, "_HOME_PIN_LIMIT", 3 << 30)
    assert h.register_mapped_home(0x1000, 1 << 30, "t") == 0x1000 + 0x7000_0000_0000
    assert h.register_mapped_home(0x2000, 2 << 30, "t") != 0
    assert h.register_mapped_home(0x3000, 1 << 30, "t") == 0      # over the pin limit


# ---- hardening round: regressions and stress ------------------------------------------------

def gpu_read_ok(cache, fl, r):
    """What the GPU tiers would read for this call is the right data: every GPU-lane expert's
    rows point at its home, or at a slot that holds exactly its home bytes"""
    st = cache.ref_state
    for e in set(int(x) for x in r["sel"].tolist()):
        ln = int(r["lane"][e])
        if ln not in (ref.LANE_GPU, ref.LANE_ZC):
            continue
        s = int(st.slotof[fl.layer.L * EMAX + e])
        if ln == ref.LANE_GPU:
            assert s >= 0, "GPU lane without a slot"
            assert fl.rows(e) == fl.expect(int(st.slot_base[s]))
            assert torch.equal(slot_bytes(cache, fl, e), fl.home[e * fl.eb: (e + 1) * fl.eb]), "stale slot"
        else:
            assert s < 0 and fl.rows(e) == fl.expect(int(fl.homes[e])), "zero-copy row not at home"


def class_range(cache, cl):
    ctl = cache.ref_state.ctl
    return int(ctl[ref.CTL_CLASS_LO + 2 * cl.id]), int(ctl[ref.CTL_CLASS_HI + 2 * cl.id])


def test_clock_range_is_the_enabled_prefix():
    # Reviewer's case: a layout sized for a large device total, a small budget enabled. The scan
    # must stay inside the enabled prefix, so admissions never end in no_victim while free
    # (unpinned) slots exist
    chunk = 4 * 1024
    mem = Mem(free = 4 * chunk, total = 4000 * chunk)
    c = H.cache.ExpertCache(torch.device("cpu"), backend = "ref", mem_info = mem, tuning = tuning())
    mem.cache = c
    fl = FakeLayer(c, "a", E = 64)
    c.begin_pass(1, 32)
    cl = fl.layer.cls
    assert cl.cap >= 3000 * 4                    # layout covers the device
    assert class_range(c, cl) == (cl.lo, cl.lo + 4 * 4)
    rng = np.random.default_rng(3)
    for _ in range(300):
        sel = rng.choice(64, size = 4, replace = False).tolist()
        r = issue(c, fl, sel, rows = 1)
        gpu_read_ok(c, fl, r)
    assert c.stats()["no_victim"] == 0
    # Shrink and regrow move the range with the prefix
    mem.free -= 2 * chunk
    c.begin_pass(64, 32)
    assert class_range(c, cl) == (cl.lo, cl.lo + 2 * 4)
    mem.free += 2 * chunk
    c.idle_maintenance()
    assert class_range(c, cl) == (cl.lo, cl.lo + 4 * 4)
    check_invariants(c, [fl])


def test_interleaved_admission_eviction_shrink_regrow_mixed_sizes():
    # Three expert sizes (three classes), four layers from three components on one owner;
    # routing calls interleaved with shrink / regrow / release_all / unregister, as the host
    # thread would issue them between passes. After every step: invariants, and the data each
    # GPU lane would read is correct
    chunk = 4 * 1024
    mem = Mem(free = 40 * chunk)
    c = H.cache.ExpertCache(torch.device("cpu"), backend = "ref", mem_info = mem, tuning = tuning())
    mem.cache = c
    layers = [FakeLayer(c, "text:l0", E = 32, eb = 1024), FakeLayer(c, "text:l1", E = 32, eb = 1024),
              FakeLayer(c, "mtp:l0", E = 24, eb = 2048), FakeLayer(c, "text#1:l0", E = 16, eb = 512)]
    c.begin_pass(1, 32)
    rng = np.random.default_rng(11)
    live = list(layers)
    for step in range(400):
        op = rng.integers(0, 20)
        if op == 0:
            mem.free -= int(rng.integers(1, 6)) * chunk
            c.begin_pass(64, 32)
        elif op == 1:
            mem.free = 40 * chunk
            c.idle_maintenance()
        elif op == 2 and step % 7 == 0:
            c.release_all()
            c.begin_pass(1, 32)
        elif op == 3 and step == 200:
            gone = live.pop(3)                 # the draft's only layer: its class dies
            c.unregister_layer(gone.layer)
            assert not any(x is not None for x in gone.layer.cls.chunks)
        else:
            fl = live[int(rng.integers(0, len(live)))]
            rows = int(rng.integers(1, 5))
            sel = np.concatenate([rng.choice(fl.E, size = 4, replace = False) for _ in range(rows)])
            r = issue(c, fl, sel.tolist(), rows = rows)
            gpu_read_ok(c, fl, r)
        check_invariants(c, live)
        assert c.cache_bytes() <= max(0, mem.free)
    st = c.stats()
    assert st["admits"] > 0 and st["evictions"] > 0 and st["no_lane"] == 0
    assert st["admit_bytes"] > 0


def test_speculative_verify_rows_and_admission_cap():
    # A verify forward is rows = jobs * (1 + window): same decode path, more picks. Above
    # EXL3_EXPERT_CACHE_ADMIT_MAX picks nothing is admitted (hits still count)
    c, _ = make_cache(admit_max = 64)
    fl = FakeLayer(c, "a", E = 64)
    c.begin_pass(1, 32)
    rng = np.random.default_rng(5)
    for jobs, window in [(1, 3), (2, 4), (4, 7), (8, 7)]:
        rows = jobs * (1 + window)
        sel = np.concatenate([rng.choice(64, size = 8, replace = False) for _ in range(rows)])
        before = c.stats()["admits"]
        admit = int(sel.size <= c.T.admit_max)
        r = issue(c, fl, sel.tolist(), rows = rows, admit = admit)
        gpu_read_ok(c, fl, r)
        if not admit:
            assert c.stats()["admits"] == before
        check_invariants(c, [fl])


def test_capacity_guard_rejects_oversized_decode_jobs():
    # Regression for the earlier segfault: a decode call with more rows than the node slot
    # holds wrote past the worker buffer. The reference (like the binding) now refuses it
    c, _ = make_cache()
    fl = FakeLayer(c, "a")
    c.set_layer_cpuloc(fl.layer, np.arange(fl.E, dtype = np.int32)[None])
    c.begin_pass(1, 32)
    nodes = Nodes(1, 4, 1, 8)                   # one row, four picks
    with pytest.raises(ValueError, match = "capacity"):
        issue(c, fl, [1, 2, 3, 4, 5, 6, 7, 8], rows = 2, nodes = nodes, admit = 0)
    # A partly filled node column is refused, an all-zero one is an absent node
    bad = Nodes(2, 4, 1, 8)
    bad.t[ref.ND_X, 1] = 0
    with pytest.raises(ValueError, match = "complete"):
        issue(c, fl, [1, 2, 3, 4], rows = 1, nodes = bad, admit = 0)
    absent = Nodes(2, 4, 1, 8)
    absent.t[:, 1] = 0
    r = issue(c, fl, [1, 2, 3, 4], rows = 1, nodes = absent, admit = 0, cpu_frac = 1.0)
    assert all(int(r["lane"][e]) == 0 for e in (1, 2, 3, 4))


def test_pick_node_float32_tie_matches_kernel():
    # Reviewer's numbers: float32 products tie (kernel keeps node 0), float64 would not
    c, _ = make_cache()
    fl = FakeLayer(c, "a")
    c.set_layer_cpuloc(fl.layer, np.tile(np.arange(fl.E, dtype = np.int32), (2, 1)))
    load = [19, 10] + [0] * 6
    cost = [np.float32(1.4554425), np.float32(2.646259)]
    assert np.float32(np.float32(20) * cost[0]) == np.float32(np.float32(11) * cost[1])
    assert ref.pick_node(c.ref_state, fl.layer.L, 3, 1, 2, 2, -1, load, cost) == 0


def lane_makespan(lane, sel_experts, hits, lc, nodes_of):
    """GPU time vs every CPU node's time for the lanes a call got (the cost model's own terms)"""
    t_gpu = hits * lc[0]
    t_node = {}
    for e in sel_experts:
        ln = lane[e]
        if ln in (ref.LANE_ZC, ref.LANE_GPU):
            t_gpu += lc[1]
        elif ln >= 0:
            t_node[ln] = t_node.get(ln, lc[2]) + lc[3]
    return max([t_gpu] + list(t_node.values()))


@pytest.mark.parametrize("seed", range(6))
def test_cost_model_cpu_never_lengthens_the_slowest_lane(seed):
    rng = np.random.default_rng(seed)
    c, _ = make_cache()
    E = 48
    fl = FakeLayer(c, "a", E = E, mapped = [e % 11 != 0 for e in range(E)])
    loc = np.full((2, E), -1, dtype = np.int32)
    for e in range(E):
        if e % 7 != 0:                          # some experts have no CPU copy
            loc[e % 2, e] = e // 2
    c.set_layer_cpuloc(fl.layer, loc)
    c.begin_pass(1, 32)
    nodes = Nodes(2, 16, 2, 8)
    for _ in range(25):
        lc = [float(np.float32(rng.uniform(1, 20))) for _ in range(4)]
        lc[2] = float(np.float32(rng.uniform(0, 60)))
        rows = int(rng.integers(1, 3))
        sel = np.concatenate([rng.choice(E, size = 8, replace = False) for _ in range(rows)]).tolist()
        resident = {e for e in set(sel) if int(c.ref_state.slotof[fl.layer.L * EMAX + e]) >= 0}
        r = issue(c, fl, sel, rows = rows, nodes = nodes, admit = 0, lane_cost = lc)
        lane = {e: int(r["lane"][e]) for e in set(sel)}
        misses = [e for e in set(sel) if e not in resident]
        got = lane_makespan(lane, misses, len(resident), lc, None)
        # Baseline: every flexible miss (GPU-readable home and a CPU copy) on the GPU
        base = dict(lane)
        for e in misses:
            if fl.homes[e] and (loc[:, e] >= 0).any():
                base[e] = ref.LANE_ZC
        want = lane_makespan(base, misses, len(resident), lc, None)
        assert got <= want * (1 + 1e-6), "a CPU lane lengthened the call's slowest lane"
        for e in misses:
            if not fl.homes[e]:
                assert lane[e] >= 0 or not (loc[:, e] >= 0).any()     # unmapped: CPU or no lane
            elif not (loc[:, e] >= 0).any():
                assert lane[e] == ref.LANE_ZC                        # no CPU copy: GPU


def test_cost_model_ties_stay_on_gpu():
    c, _ = make_cache()
    fl = FakeLayer(c, "a")
    c.set_layer_cpuloc(fl.layer, np.arange(fl.E, dtype = np.int32)[None])
    c.begin_pass(1, 32)
    nodes = Nodes(1, 2, 1, 8)
    # Two misses, zc = 10, cpu = 10, no fixed cost. All-GPU: 20. One on CPU: max(10, 10) = 10 <
    # 20, so exactly one goes to the CPU; the second would tie (max(10, 20) vs max(20, 10))
    r = issue(c, fl, [3, 4], rows = 1, nodes = nodes, admit = 0, lane_cost = [0.0, 10.0, 0.0, 10.0])
    lanes = sorted(int(r["lane"][e]) for e in (3, 4))
    assert lanes == [ref.LANE_ZC, 0]
    # A fixed CPU cost that makes the CPU no better: everything stays on the GPU
    r = issue(c, fl, [5, 6], rows = 1, nodes = nodes, admit = 0, lane_cost = [0.0, 10.0, 10.0, 10.0])
    assert all(int(r["lane"][e]) == ref.LANE_ZC for e in (5, 6))
    st = c.stats()
    assert st["zero_copy_bytes"] == st["zero_copy"] * fl.eb


def test_per_node_cpu_counters():
    c, _ = make_cache()
    fl = FakeLayer(c, "a")
    shard = np.full((2, fl.E), -1, dtype = np.int32)
    for e in range(fl.E):
        shard[e % 2, e] = e // 2
    c.set_layer_cpuloc(fl.layer, shard)
    c.begin_pass(1, 32)
    issue(c, fl, [1, 2, 3, 5], rows = 1, nodes = Nodes(2, 4, 1, 8), admit = 0, cpu_frac = 1.0)
    st = c.stats()
    assert st["cpu_by_node"][:2] == [1, 3] and st["cpu"] == 4
    s = c.summary()
    assert "vram_limit_mib" in s and "slot_utilization" in s


def test_headroom_extra_is_not_double_counted_and_no_churn():
    chunk = 4 * 1024
    c, mem = make_cache(free = 10 * chunk, min_free_bytes = 0, grow_hysteresis = 0)
    fl = FakeLayer(c, "a", E = 64)
    c.note_transient(3 * chunk)                 # measured transient already holds the MoE bound
    c.begin_pass(1, 32)
    assert c.cache_bytes() == 7 * chunk
    assert c.ensure_headroom(2 * chunk) == 0    # extra below the transient: nothing to give back
    freed = c.ensure_headroom(5 * chunk)        # larger than the transient: shrink to it
    assert freed == 2 * chunk and c.cache_bytes() == 5 * chunk
    for _ in range(3):                          # idle regrow stops short of it: no churn
        c.idle_maintenance()
        assert c.ensure_headroom(5 * chunk) == 0
    assert c.cache_bytes() == 5 * chunk


def test_device_transient_from_any_component_raises_the_floor():
    chunk = 4 * 1024
    H.cache.ExpertCache._drop_registry()
    c, mem = make_cache(free = 10 * chunk)
    FakeLayer(c, "a")
    H.cache.note_device_transient(0, 6 * chunk, device_type = "cpu")
    H.cache.note_device_transient(0, 2 * chunk, device_type = "cpu")      # smaller: ignored
    c.begin_pass(1, 32)
    assert c.cache_bytes() == 4 * chunk
    H.cache.ExpertCache._drop_registry()


def test_stats_saved_when_each_component_leaves(tmp_path):
    out = tmp_path / "stats.json"
    c, _ = make_cache(save_stats = str(out))
    a = FakeLayer(c, "text:l0")
    d = FakeLayer(c, "text#1:l0")
    c.begin_pass(1, 32)
    issue(c, a, [1, 2])
    issue(c, d, [3])
    c.unregister_layer(d.layer)                  # draft leaves first: snapshot
    assert set(json.loads(out.read_text())) >= {"text#1:l0"}
    issue(c, a, [1])
    c.unregister_layer(a.layer)                  # main leaves: saved again, current history
    saved = json.loads(out.read_text())
    assert saved["text:l0"][1] >= 2.0
    # A second model in the same process saves too
    b = FakeLayer(c, "text:l0")
    c.begin_pass(1, 32)
    issue(c, b, [7, 7])
    c.unregister_layer(b.layer)
    assert json.loads(out.read_text())["text:l0"][7] >= 2.0


def test_dead_class_is_reused_and_classes_reset_when_empty():
    c, _ = make_cache(lcap = 16)
    live = [FakeLayer(c, f"l{i}", E = 4, eb = 256 * (i + 1)) for i in range(ref.MAX_CLASSES)]
    c.begin_pass(1, 32)
    with pytest.raises(RuntimeError, match = "distinct expert sizes"):
        FakeLayer(c, "x", E = 4, eb = 256 * 20)
    c.unregister_layer(live[2].layer)
    nl = FakeLayer(c, "x", E = 4, eb = 256 * 20)
    assert nl.layer.cls.id == 2
    c.begin_pass(1, 32)
    check_invariants(c, live[:2] + live[3:] + [nl])
    for fl in live[:2] + live[3:] + [nl]:
        c.unregister_layer(fl.layer)
    assert c.classes == [] and c.cache_bytes() == 0


def test_limit_below_one_chunk_gets_a_notice(capsys):
    c, _ = make_cache(limit_bytes = 2 * 1024)    # chunk = 4 slots x 1 KiB
    FakeLayer(c, "a")
    c.begin_pass(1, 32)
    assert c.cache_bytes() == 0
    assert "fits EXL3_EXPERT_CACHE_LIMIT_GB" in capsys.readouterr().out
    # Two classes under a modest limit: the small one still gets a chunk
    c2, _ = make_cache(limit_bytes = 5 * 4 * 1024)
    FakeLayer(c2, "big", E = 64, eb = 4096)
    small = FakeLayer(c2, "small", E = 4, eb = 256)
    c2.begin_pass(1, 32)
    assert small.layer.cls.allocated_bytes() > 0


def test_grow_skips_a_class_whose_chunk_does_not_fit():
    # Room for small chunks only: the big class must not starve the small one
    c, _ = make_cache(free = 3 * 4 * 256 + 100)
    big = FakeLayer(c, "big", E = 64, eb = 4096)
    small = FakeLayer(c, "small", E = 8, eb = 256)
    c.begin_pass(1, 32)
    assert big.layer.cls.allocated_bytes() == 0 and small.layer.cls.allocated_bytes() > 0


def test_warm_start_runs_once_per_owner_generation(tmp_path):
    p = tmp_path / "warm.json"
    p.write_text(json.dumps({"a": [0.0] * 5 + [10.0] + [0.0] * 10}))
    c, _ = make_cache(warm = str(p))
    fl = FakeLayer(c, "a")
    c.begin_pass(1, 32)
    for _ in range(3):
        issue(c, fl, [2])
    h2 = float(c.ref_state.hist[fl.layer.L * EMAX + 2])
    FakeLayer(c, "b", eb = 2048)                 # new class: re-layout
    c.begin_pass(1, 32)
    assert float(c.ref_state.hist[fl.layer.L * EMAX + 2]) == h2   # live history kept


def test_stats_log_keeps_newest_counter_lines_only(tmp_path, monkeypatch):
    path = tmp_path / "cache_stats.jsonl"
    c, _ = make_cache(stats_log = str(path), stats_log_s = 60, stats_log_max = 3)
    fl = FakeLayer(c, "a")
    c.begin_pass(1, 32)
    clock = [1000.0]
    monkeypatch.setattr(H.cache.time, "time", lambda: clock[0])
    for i in range(5):
        issue(c, fl, [1, 2, 3])
        c.idle_maintenance()
        c.idle_maintenance()                 # same interval: no second line
        clock[0] += 61
    lines = [json.loads(x) for x in path.read_text().splitlines()]
    assert len(lines) == 3
    assert set(lines[-1]) >= {"t", "device", "hit_rate", "hits", "misses", "admits", "bg_admits"}
    assert all(l["hits"] + l["misses"] == 3 for l in lines)      # per-interval deltas
    assert lines[-1]["hit_rate"] == 1.0

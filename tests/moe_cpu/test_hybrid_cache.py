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
        t = torch.zeros((5, N), dtype = torch.long)
        for i in range(N):
            t[0, i] = self.sel[i].data_ptr()
            t[1, i] = self.x[i].data_ptr()
            t[2, i] = self.w[i].data_ptr()
            t[3, i] = self.count[i].data_ptr()
            t[4, i] = 0
        self.t = t
        self.cost = torch.ones((N,), dtype = torch.float)


def issue(cache, fl, sel, rows = None, hi = 8, nodes = None, prefill = 0, admit = 1, gpu_max = -1,
          cpu_frac = 0.0, gmin = 0, policy = 0, pref = -1, h = 8):
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
    if nodes is not None and not prefill:
        assert nodes.x[0].numel() >= rows * hi and nodes.sel[0].numel() >= n, "node buffers too small"
    nt = nodes.t if nodes is not None else torch.zeros((5, 0), dtype = torch.long)
    cost = nodes.cost if nodes is not None else torch.ones((0,), dtype = torch.float)
    cache.issue(fl.layer, sel, y, w, sel_gpu, lane, cpu_sel, nt, cost, hi, prefill, admit, gpu_max,
                cpu_frac, gmin, policy, pref)
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

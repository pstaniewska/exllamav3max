"""
exllamav3_ext/moe_hybrid.cu against its reference (model/moe_hybrid_ref.py): the same sequence of
layer calls on a CUDA expert cache (tables in VRAM, homes in pinned + mapped host memory read
through their device alias) and on a reference cache (all host memory) must produce the same
lanes, masked selections, slot assignments, CLOCK state and counters, and every admitted slot must
hold its expert's home bytes. Needs a GPU; not runnable in the CPU-only CI of this change.
"""

import numpy as np
import pytest
import torch

from exllamav3.ext import exllamav3_ext as ext
from exllamav3.model import moe_expert_cache as mec
from exllamav3.model import moe_hybrid_ref as ref
from exllamav3.util.pinned_arena import PinnedArena

pytestmark = pytest.mark.moe


def _tuning():
    T = mec.CacheTuning()
    T.chunk_slots, T.lcap, T.gmax = 4, 8, 16
    T.reserve_bytes = T.min_free_bytes = T.grow_hysteresis = 0
    T.limit_bytes, T.policy, T.warm, T.save_stats, T.debug = None, "clock", None, None, False
    return T


class _Mem:
    def __init__(self, free):
        self.free, self.cache = free, None

    def __call__(self):
        return self.free - self.cache.cache_bytes(), self.free


def _cache(device, free):
    m = _Mem(free)
    c = mec.ExpertCache(device, "cuda" if device.type == "cuda" else "ref", mem_info = m, tuning = _tuning())
    m.cache = c
    return c


def _layer(c, uid, E, eb, home_bytes, device, keep):
    tabs = [torch.zeros(E, dtype = torch.long, device = device) for _ in range(3)]
    tables = [(tabs[0], 0, 1, 0), (tabs[1], 0, 1, 1), (tabs[2], 0, 1, 2)]
    layer = c.register_layer(uid, uid, E, eb, (0, eb // 4, eb // 2), tables)
    if device.type == "cuda":
        arena = PinnedArena(E * eb, "test homes", mapped = True, required = True)
        arena.tensor[:E * eb].copy_(home_bytes)
        dev = ext.cuda_host_get_device_pointer(arena.ptr)
        homes = np.array([dev + e * eb for e in range(E)], dtype = np.int64)
        view = lambda e: arena.tensor[e * eb: (e + 1) * eb]
        keep.append(arena)
    else:
        hb = home_bytes.clone()
        homes = np.array([hb.data_ptr() + e * eb for e in range(E)], dtype = np.int64)
        view = lambda e: hb[e * eb: (e + 1) * eb]
        keep.append(hb)
    c.set_layer_homes(layer, homes, view)
    loc = np.full((2, E), -1, dtype = np.int32)
    for e in range(E):
        loc[e % 2, e] = e // 2
    c.set_layer_cpuloc(layer, loc)
    keep.append(tabs)
    return layer


def _state(c):
    a = c.arrays
    return {k: a[k].cpu().clone() for k in ("slotof", "owner", "refb", "pin", "ctl", "stats", "hist")}


@pytest.mark.parametrize("cpu_frac,lane_cost", [(0.0, None), (0.5, None), (0.5, [3.0, 20.0, 15.0, 9.0]),
                                                (0.5, [0.0, 7.0, 0.0, 7.0])])
@pytest.mark.parametrize("node_policy", [0, 2])
def test_issue_matches_reference(device, cpu_frac, lane_cost, node_policy):
    mec.ExpertCache._drop_registry()
    E, eb = 24, 4096
    rng = np.random.default_rng(1)
    home_bytes = torch.randint(0, 255, (E * eb,), dtype = torch.uint8)
    keep = []
    caches, layers = [], []
    for dev in (torch.device(device), torch.device("cpu")):
        c = _cache(dev, 3 * 4 * eb)
        layers.append(_layer(c, "a", E, eb, home_bytes, dev, keep))
        c.begin_pass(1, 32)
        caches.append(c)
    for step in range(60):
        rows = int(rng.integers(1, 5))
        sel = torch.from_numpy(np.concatenate([rng.choice(E, 4, replace = False) for _ in range(rows)]))
        y = torch.randn((rows, 64)).half()
        w = torch.rand((sel.numel(),)).half()
        outs = []
        for c, layer in zip(caches, layers):
            dev = c.device
            n = sel.numel()
            nodes = torch.zeros((ref.ND_ROWS, 2), dtype = torch.long)
            bufs = []
            for q in range(2):
                hs = torch.zeros((n,), dtype = torch.int32, device = dev)
                hx = torch.zeros((rows * 64,), dtype = torch.half, device = dev)
                hw = torch.zeros((n,), dtype = torch.half, device = dev)
                cnt = torch.zeros((8,), dtype = torch.int32, device = dev)
                nodes[:, q] = torch.tensor([hs.data_ptr(), hx.data_ptr(), hw.data_ptr(), cnt.data_ptr(), 0,
                                            hx.numel(), hs.numel()])
                bufs.append((hs, hx, hw, cnt))
            sel_gpu = torch.empty((n,), dtype = torch.long, device = dev)
            lane = torch.empty((ref.EMAX,), dtype = torch.int32, device = dev)
            # Node costs with float32 near-ties (adaptive policy) and the lane cost model: the
            # reference must reproduce the kernel's float32 arithmetic exactly
            node_cost = torch.tensor([1.4554425, 2.646259], dtype = torch.float)
            lc = torch.tensor(lane_cost, dtype = torch.float) if lane_cost is not None else None
            c.issue(layer, sel.to(dev), y.to(dev), w.to(dev), sel_gpu, lane, None, nodes,
                    node_cost, 64, 0, 1, -1, cpu_frac, 0, node_policy, -1, lc)
            if dev.type == "cuda":
                torch.cuda.synchronize(dev)
            outs.append((sel_gpu.cpu(), lane.cpu()[:E], [tuple(b.cpu() for b in bb) for bb in bufs], _state(c)))
        (sg0, l0, b0, s0), (sg1, l1, b1, s1) = outs
        assert torch.equal(sg0, sg1) and torch.equal(l0, l1), f"step {step}: lanes differ"
        for k in s0:
            assert torch.equal(s0[k], s1[k]), f"step {step}: state {k} differs"
        for q in range(2):
            assert torch.equal(b0[q][0], b1[q][0]) and torch.equal(b0[q][3], b1[q][3])
            if int(b0[q][3][0]):
                assert torch.equal(b0[q][1], b1[q][1]) and torch.equal(b0[q][2], b1[q][2])
    # Admitted slots hold the home bytes on the GPU
    c = caches[0]
    st = c.arrays["slotof"].cpu()
    for e in range(E):
        s = int(st[layers[0].L * ref.EMAX + e])
        if s >= 0:
            got = c._slot_tensor(layers[0].cls, s)[:eb].cpu()
            assert torch.equal(got, home_bytes[e * eb: (e + 1) * eb])

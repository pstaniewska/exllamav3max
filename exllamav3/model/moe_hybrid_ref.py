"""
Reference (host, NumPy) implementation of the hybrid MoE cache kernels in
exllamav3_ext/moe_hybrid.cu, over the same state layout.

Two uses:
- ExpertCache(backend = "ref") runs the cache on CPU tensors with these functions in place of the
  CUDA kernels, so the owner's host logic (registration, budget, shrink/regrow, warm start,
  quiesce ordering, unregistration) and the replacement policy can be tested without a GPU.
- On a CUDA host, tests/moe_cpu/test_hybrid_cache.py compares the kernels against this file
  call by call (lanes, selections, slot state, table rows).

The functions mirror the kernels step by step; keep them in sync. Memory referenced by address in
the state (pointer tables, slots, homes, worker slot sections) is real host memory in ref mode
and is read/written through ctypes, exactly as the kernels do through device pointers.
"""

from __future__ import annotations
import ctypes
import numpy as np

# Constants shared with moe_hybrid.cuh (single Python source; moe_expert_cache imports these)
EMAX = 512
MAX_TABLES = 6
MAX_NODES = 8
MAX_CLASSES = 8

S_SLOTOF, S_OWNER, S_REFB, S_PIN, S_SLOTBASE, S_CTL, S_STATS, S_GJOBS, S_HIST = range(9)
S_REG_TABLES, S_REG_TSTRIDE, S_REG_TPROJ, S_REG_PROJOFF, S_REG_HOME = 9, 10, 11, 12, 13
S_REG_BOUNCE, S_REG_BYTES, S_REG_CLASS, S_REG_CPULOC = 14, 15, 16, 17
S_SCAP, S_LCAP, S_GMAX = 18, 19, 20
S_COUNT = 21

CTL_EPOCH, CTL_GCOUNT, CTL_HAND, CTL_CLASS_LO, CTL_CLASS_HI = 0, 1, 8, 16, 17
CTL_SIZE = 32

ST_HITS, ST_MISSES, ST_ADMITS, ST_EVICTS, ST_NO_VICTIM, ST_ZC, ST_CPU, ST_CALLS = range(8)
ST_NO_LANE, ST_PREFILL = 8, 9
STATS_SIZE = 16
STAT_NAMES = ["hits", "misses", "admits", "evictions", "no_victim", "zero_copy", "cpu", "calls",
              "no_lane", "prefill_calls"]

LANE_NONE, LANE_GPU, LANE_ZC = -2, -1, -3

OP_EVICT, OP_DISABLE, OP_ENABLE, OP_INSTALL, OP_RESET_LAYER = 0, 1, 2, 3, 4

SCAN_MAX = 8192


class RefState:
    """NumPy views over the state arrays of an ExpertCache whose tensors live on the CPU"""

    def __init__(self, arrays: dict, scap: int, lcap: int, gmax: int):
        a = {k: (v.numpy() if hasattr(v, "numpy") else v) for k, v in arrays.items()}
        self.slotof = a["slotof"]
        self.owner = a["owner"]
        self.refb = a["refb"]
        self.pin = a["pin"]
        self.slot_base = a["slot_base"]
        self.ctl = a["ctl"]
        self.stats = a["stats"]
        self.gjobs = a["gjobs"]
        self.hist = a["hist"]
        self.reg_tables = a["reg_tables"]
        self.reg_tstride = a["reg_tstride"]
        self.reg_tproj = a["reg_tproj"]
        self.reg_projoff = a["reg_projoff"]
        self.reg_home = a["reg_home"]
        self.reg_bounce = a["reg_bounce"]
        self.reg_bytes = a["reg_bytes"]
        self.reg_class = a["reg_class"]
        self.reg_cpuloc = a["reg_cpuloc"]
        self.scap, self.lcap, self.gmax = scap, lcap, gmax


def _poke_i64(addr: int, value: int):
    ctypes.c_int64.from_address(addr).value = int(value)


def _arr(addr: int, ctype, n: int):
    return np.ctypeslib.as_array((ctype * n).from_address(addr))


def home(st: RefState, L: int, e: int) -> int:
    h = int(st.reg_home[L * EMAX + e])
    return h if h else int(st.reg_bounce[L])


def repoint(st: RefState, L: int, e: int, base: int):
    for t in range(MAX_TABLES):
        r = L * MAX_TABLES + t
        tab = int(st.reg_tables[r])
        if not tab:
            continue
        p = int(st.reg_tproj[r])
        _poke_i64(tab + 8 * e * int(st.reg_tstride[r]), base + int(st.reg_projoff[L * 3 + p]))


def evict(st: RefState, s: int):
    o = int(st.owner[s])
    if o < 0:
        return
    L2, e2 = divmod(o, EMAX)
    st.slotof[o] = -1
    repoint(st, L2, e2, home(st, L2, e2))
    st.owner[s] = -1
    st.stats[ST_EVICTS] += 1


def install(st: RefState, L: int, e: int, s: int):
    evict(st, s)
    key = L * EMAX + e
    st.owner[s] = key
    st.slotof[key] = s
    repoint(st, L, e, int(st.slot_base[s]))


def cpuloc(st: RefState, L: int, node: int, e: int) -> int:
    return int(st.reg_cpuloc[(L * MAX_NODES + node) * EMAX + e])


def pick_node(st, L, e, cnt, nn, policy, pref, load, cost) -> int:
    best, best_c = -1, 0.0
    for nd in range(nn):
        if cpuloc(st, L, nd, e) < 0:
            continue
        if policy == 0:
            if nd == pref:
                return nd
            if best < 0:
                best = nd
            continue
        c = float(load[nd]) if policy == 1 else float(load[nd] + cnt) * float(cost[nd])
        if best < 0 or c < best_c:
            best, best_c = nd, c
    return best


def issue(st: RefState, nodes: dict, L: int, E: int, sel: np.ndarray, y: np.ndarray | None, hi: int,
          w: np.ndarray, sel_gpu: np.ndarray, lane_out: np.ndarray | None, cpu_sel: np.ndarray | None,
          prefill: int, admit: int, gpu_max: int, cpu_frac: float, prefill_gpu_min: int,
          node_policy: int, pref_node: int):
    """Mirror of moe_hybrid_issue_kernel (+ moe_hybrid_stage_x_kernel in decode mode).
    nodes: {"n", "h_sel", "h_x", "h_w", "dev_count", "slot", "cost"} (lists of ints/floats)"""
    n = sel.size
    sel = sel.reshape(-1)
    hbase = L * EMAX
    cnt = np.zeros(EMAX, dtype = np.int64)
    lane = np.full(EMAX, LANE_NONE, dtype = np.int64)

    ep = (int(st.ctl[CTL_EPOCH]) + 1) & 0x3FFFFFFF
    if ep == 0:
        ep = 1
    st.ctl[CTL_EPOCH] = ep
    st.ctl[CTL_GCOUNT] = 0

    for e in sel:
        if 0 <= e < E:
            cnt[e] += 1

    hits = 0
    miss = []
    for e in range(E):
        c = int(cnt[e])
        if not c:
            continue
        st.hist[hbase + e] += float(c)
        s = int(st.slotof[hbase + e])
        if s >= 0:
            st.refb[s] = 1
            st.pin[s] = ep
            lane[e] = LANE_GPU
            hits += 1
        else:
            miss.append((e, float(st.hist[hbase + e])))

    st.stats[ST_CALLS] += 1
    st.stats[ST_HITS] += hits
    st.stats[ST_MISSES] += len(miss)
    if prefill:
        st.stats[ST_PREFILL] += 1
        miss.sort(key = lambda m: m[0])
    else:
        # cnt desc, hist desc, id asc (the kernel's float32 hist compares the same way here as
        # long as counts stay below 2^24, which the decay keeps them)
        miss.sort(key = lambda m: (-int(cnt[m[0]]), -np.float32(m[1]), m[0]))
    miss = [m[0] for m in miss]

    nn = int(nodes["n"])
    nflex = 0
    for e in miss:
        if not int(st.reg_home[hbase + e]):
            continue
        nflex += any(cpuloc(st, L, q, e) >= 0 for q in range(nn))
    if gpu_max >= 0:
        quota = min(gpu_max, nflex)
    else:
        quota = nflex - int(np.rint(np.float32(nflex) * np.float32(cpu_frac)))
    quota = max(quota, 0)

    load = [0] * MAX_NODES
    cost = [c if c > 0 else 1.0 for c in nodes.get("cost", [1.0] * nn)]
    cls = int(st.reg_class[L])
    lo = int(st.ctl[CTL_CLASS_LO + 2 * cls]) if cls >= 0 else 0
    rng = int(st.ctl[CTL_CLASS_HI + 2 * cls]) - lo if cls >= 0 else 0
    hand = int(st.ctl[CTL_HAND + cls]) if cls >= 0 else 0
    if rng > 0 and (hand < 0 or hand >= rng):
        hand = 0
    gcount = 0
    flex_gpu = 0

    for e in miss:
        c = int(cnt[e])
        mapped = int(st.reg_home[hbase + e]) != 0
        node = pick_node(st, L, e, c, nn, node_policy, pref_node, load, cost)
        if not mapped:
            to_gpu = False
        elif node < 0:
            to_gpu = True
        elif prefill:
            to_gpu = c >= prefill_gpu_min
        else:
            to_gpu = flex_gpu < quota
            flex_gpu += int(to_gpu)

        ln = LANE_NONE
        if to_gpu:
            ln = LANE_ZC
            if admit and not prefill and rng > 0 and gcount < st.gmax:
                victim = -1
                for _ in range(min(2 * rng, SCAN_MAX)):
                    s = lo + hand
                    hand = 0 if hand + 1 == rng else hand + 1
                    p = int(st.pin[s])
                    if p < 0 or p == ep:
                        continue
                    if st.refb[s]:
                        st.refb[s] = 0
                        continue
                    victim = s
                    break
                if victim >= 0:
                    install(st, L, e, victim)
                    st.refb[victim] = 1
                    st.pin[victim] = ep
                    st.gjobs[3 * gcount: 3 * gcount + 3] = (
                        int(st.slot_base[victim]), int(st.reg_home[hbase + e]), int(st.reg_bytes[L]))
                    gcount += 1
                    ln = LANE_GPU
                    st.stats[ST_ADMITS] += 1
                else:
                    st.stats[ST_NO_VICTIM] += 1
            if ln == LANE_ZC:
                st.stats[ST_ZC] += 1
        elif node >= 0:
            ln = node
            load[node] += c
            st.stats[ST_CPU] += 1
        else:
            st.stats[ST_NO_LANE] += 1
        lane[e] = ln

    if cls >= 0:
        st.ctl[CTL_HAND + cls] = hand
    st.ctl[CTL_GCOUNT] = gcount

    valid = (sel >= 0) & (sel < E)
    lanes_per_pick = np.where(valid, lane[np.clip(sel, 0, EMAX - 1)], LANE_NONE)
    gpu = (lanes_per_pick == LANE_GPU) | (lanes_per_pick == LANE_ZC)
    sel_gpu[:n] = np.where(gpu, sel, E)
    if lane_out is not None:
        lane_out[:E] = lane[:E]

    any_node = [False] * nn
    for q in range(nn):
        if prefill:
            if cpu_sel is None:
                continue
            hs = cpu_sel[q * n: (q + 1) * n]
        else:
            hs = _arr(int(nodes["h_sel"][q]), ctypes.c_int32, n)
        v = np.array([cpuloc(st, L, q, int(e)) if lanes_per_pick[i] == q else -1
                      for i, e in enumerate(sel)], dtype = np.int32)
        hs[:] = v
        any_node[q] = bool((v >= 0).any())
    if prefill:
        return lane

    rows = y.shape[0] if y is not None else 0
    for q in range(nn):
        _arr(int(nodes["dev_count"][q]), ctypes.c_int32, int(nodes["slot"][q]) + 1)[int(nodes["slot"][q])] = int(any_node[q])
        if not any_node[q]:
            continue
        _arr(int(nodes["h_w"][q]), ctypes.c_uint16, n)[:] = w.reshape(-1).view(np.uint16)
        hx = _arr(int(nodes["h_x"][q]), ctypes.c_uint16, rows * hi).reshape(rows, hi)
        hx[:] = 0
        hx[:, :y.shape[1]] = y.view(np.uint16)
    return lane


def gather(st: RefState):
    """Mirror of moe_hybrid_gather_kernel"""
    for j in range(int(st.ctl[CTL_GCOUNT])):
        dst, src, nbytes = (int(v) for v in st.gjobs[3 * j: 3 * j + 3])
        ctypes.memmove(dst, src, nbytes)


def host_ops(st: RefState, ops: np.ndarray):
    """Mirror of moe_hybrid_host_ops_kernel"""
    for op, L, e, s in ops.reshape(-1, 4).tolist():
        if op == OP_EVICT:
            if 0 <= s < st.scap:
                evict(st, s)
        elif op == OP_DISABLE:
            if 0 <= s < st.scap:
                evict(st, s)
                st.pin[s] = -1
                st.refb[s] = 0
        elif op == OP_ENABLE:
            if 0 <= s < st.scap:
                st.owner[s] = -1
                st.refb[s] = 0
                st.pin[s] = 0
        elif op == OP_INSTALL:
            if 0 <= s < st.scap and 0 <= L < st.lcap and 0 <= e < EMAX and st.pin[s] >= 0 \
                    and st.slotof[L * EMAX + e] < 0:
                install(st, L, e, s)
                st.refb[s] = 0
        elif op == OP_RESET_LAYER:
            if 0 <= L < st.lcap:
                for x in range(min(e, EMAX)):
                    s2 = int(st.slotof[L * EMAX + x])
                    if s2 >= 0:
                        evict(st, s2)
                    else:
                        repoint(st, L, x, home(st, L, x))

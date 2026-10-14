"""
Synthetic two-socket benchmark for how CPU-lane experts are split between NUMA nodes. One worker
process per node (confined to that node's CPUs, so the weights are first-touched in node-local
memory) runs the real EXL3 CPU expert kernel on random trellis weights; the parent drives both in
lockstep and times each token as the slower node's time, which is what the GPU waits for.

Strategies, interleaved token by token so background load hits all of them equally:
  shard      expert e runs on node e mod 2 (EXL3_NUMA_MEMORY_POLICY=shard, the default)
  replicate  every node holds every expert; each expert goes to the node with fewer rows so far
             (EXL3_NUMA_MEMORY_POLICY=replicate, numa_adaptive with equal costs)
  tp         every expert is split along the intermediate dimension: each node computes half of
             every expert's intermediate rows and a partial down projection (not implemented in
             serving; this measures what it would buy)
  one_node   all experts on node 0 only (reference)

    python eval/numa_split_bench.py --node0 1-51 --node1 65-115 --cpu_experts 2,4,6,8

Defaults follow RED-SNOW / GLM-5.3-Flash (hidden 4096, intermediate 2048, 288 experts, gate/up K 6,
down K 4). --cpu_experts is the number of distinct experts per row that miss the GPU cache. With
--rows R > 1 (MTP verify) each row draws its own experts. Per-layer times; x42 for the model.
"""

import argparse
import multiprocessing as mp
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def parse_cpus(s: str) -> list[int]:
    out = []
    for part in s.split(","):
        a, _, b = part.partition("-")
        out += range(int(a), int(b or a) + 1)
    return out


def worker(cpus, args, conn, start, done):
    os.sched_setaffinity(0, cpus)
    import torch
    torch.set_num_threads(1)
    from exllamav3.ext import exllamav3_ext as ext

    E, H, I = args.experts, args.hidden, args.interm
    swz = int(any(ext.exl3_moe_cpu_swizzle_group(float(K)) for K in (args.k_gate_up, args.k_down)))
    gen = torch.Generator().manual_seed(len(cpus))

    def mats(k, n, K):
        t, u, v = [], [], []
        for _ in range(E):
            t.append(torch.randint(-32768, 32767, (k // 16, n // 16, 16 * K), dtype = torch.int16, generator = gen))
            u.append((torch.randn(k, generator = gen) * 0.015).half())
            v.append(torch.randn(n, generator = gen).half())
        return t, u, v

    def layer(interm):
        g, u, d = mats(H, interm, args.k_gate_up), mats(H, interm, args.k_gate_up), mats(interm, H, args.k_down)
        return ext.exl3_moe_cpu_make_layer(*g, *u, *d, [], [], [], 0, 1.0, swz)

    full = layer(I)
    half = layer(I // 2)
    nt = args.threads or len(cpus)
    conn.send("ready")
    plan = conn.recv()                      # list of (kind, rows, [expert lists per row]) per step
    xs = {}
    for step in plan:
        start.wait()
        kind, rows, sel = step
        if sel and any(sel):
            k = max(len(s) for s in sel)
            if (rows, k) not in xs:
                xs[(rows, k)] = (torch.randn(rows, H).half(), torch.full((rows, k), 1.0 / k, dtype = torch.half),
                                 torch.empty(rows, H))
            x, w, out = xs[(rows, k)]
            # The kernel takes a fixed expert count per row; the parent builds rows that way
            s = torch.tensor(sel, dtype = torch.long)
            ext.exl3_moe_cpu_forward(half if kind == "tp" else full, x, s, w, out, nt)
        done.wait()
    ext.exl3_moe_cpu_free_layer(full)
    ext.exl3_moe_cpu_free_layer(half)


def split_rows(sel_rows, owner):
    """Per node: each row's experts owned by that node (lists may differ in length)"""
    out = [[], []]
    for row in sel_rows:
        parts = [[e for e in row if owner(e) == n] for n in (0, 1)]
        for n in (0, 1):
            out[n].append(parts[n])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--node0", default = "1-51")
    ap.add_argument("--node1", default = "65-115")
    ap.add_argument("--threads", type = int, default = 0, help = "pool size per node (0: all CPUs given)")
    ap.add_argument("--experts", type = int, default = 288)
    ap.add_argument("--hidden", type = int, default = 4096)
    ap.add_argument("--interm", type = int, default = 2048)
    ap.add_argument("--k_gate_up", type = int, default = 6)
    ap.add_argument("--k_down", type = int, default = 4)
    ap.add_argument("--cpu_experts", default = "4,8")
    ap.add_argument("--rows", default = "1", help = "rows > 1: shard is an upper bound (see below)")
    ap.add_argument("--iters", type = int, default = 300)
    ap.add_argument("--seed", type = int, default = 0)
    args = ap.parse_args()

    import random
    rnd = random.Random(args.seed)
    ctx = mp.get_context("spawn")
    start, done = ctx.Barrier(3), ctx.Barrier(3)
    pipes, procs = [], []
    for cpus in (parse_cpus(args.node0), parse_cpus(args.node1)):
        a, b = ctx.Pipe()
        p = ctx.Process(target = worker, args = (cpus, args, b, start, done), daemon = True)
        p.start()
        pipes.append(a)
        procs.append(p)
    t0 = time.time()
    for a in pipes:
        assert a.recv() == "ready"
    print(f" -- workers ready in {time.time() - t0:.0f} s (node0 {args.node0}, node1 {args.node1})", flush = True)

    strategies = ["shard", "replicate", "tp", "one_node"]
    # One kernel call per node per step. With rows > 1 under shard, a node's rows are topped up
    # to its busiest row's count with other experts of its shard (the kernel takes a fixed count
    # per row), which slightly overstates shard's cost; rows = 1 is exact
    plans = [[], []]
    keys = []
    for ce in [int(x) for x in args.cpu_experts.split(",")]:
        for rows in [int(x) for x in args.rows.split(",")]:
            for it in range(args.iters + 10):
                draw = [rnd.sample(range(args.experts), ce) for _ in range(rows)]
                for st in strategies:
                    if st == "tp":
                        per = [draw, draw]
                    elif st == "one_node":
                        per = [draw, []]
                    elif st == "shard":
                        parts = split_rows(draw, lambda e: e % 2)
                        per = []
                        for n in (0, 1):
                            m = max(len(r) for r in parts[n])
                            rows_n = []
                            for r in parts[n]:
                                pool = [e for e in range(n, args.experts, 2) if e not in r]
                                rows_n.append(r + rnd.sample(pool, m - len(r)))
                            per.append(rows_n if m else [])
                    else:   # replicate: balanced, ceil/floor halves of each row
                        h = (ce + 1) // 2
                        per = [[r[:h] for r in draw], [r[h:] for r in draw] if ce > h else []]
                    for n in (0, 1):
                        plans[n].append((st, rows, per[n]))
                    keys.append((st, ce, rows, it >= 10))
    for a, pl in zip(pipes, plans):
        a.send(pl)
    times = {}
    for key in keys:
        start.wait()
        t = time.perf_counter()
        done.wait()
        dt = time.perf_counter() - t
        if key[3]:
            times.setdefault(key[:3], []).append(dt)
    for p in procs:
        p.join()

    print(f"\n{'experts/row':>11s} {'rows':>4s}  " + "  ".join(f"{s:>10s}" for s in strategies) + "   (median us/layer)")
    for ce in [int(x) for x in args.cpu_experts.split(",")]:
        for rows in [int(x) for x in args.rows.split(",")]:
            med = [statistics.median(times[(s, ce, rows)]) * 1e6 for s in strategies]
            base = med[0]
            print(f"{ce:11d} {rows:4d}  " + "  ".join(f"{m:10.1f}" for m in med) +
                  "   vs shard: " + " ".join(f"{s} {base / m:4.2f}x" for s, m in zip(strategies[1:], med[1:])))


if __name__ == "__main__":
    main()

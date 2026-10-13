"""
Synthetic CPU MoE layer benchmark: one routed-expert layer with random EXL3 trellis weights, timed
on the CPU kernel alone (no model, no GPU). Seconds instead of a full hybrid_perf run, for
questions about the CPU side such as thread counts, SMT and NUMA placement.

    python eval/moe_cpu_layer_bench.py --cpus 64-127,192-255 --threads 63,96,126
    EXL3_MOE_CPU_PIN=1 python eval/moe_cpu_layer_bench.py --cpus 0-63 --threads 32,64

--cpus confines the process (and so the pool and the weights' first-touch pages) to those CPUs,
e.g. one NUMA node. Defaults follow RED-SNOW / GLM-5.3-Flash: hidden 4096, expert intermediate
2048, 288 experts, gate/up K 6, down K 4. Decode routes `--decode_experts` random experts per
token (about the CPU share of top-8 after GPU cache hits); prefill routes top-8 for `--rows`
tokens. Per-layer times; multiply by the MoE layer count for a model-level estimate.
"""

import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def parse_cpus(s: str) -> list[int]:
    out = []
    for part in s.split(","):
        a, _, b = part.partition("-")
        out += range(int(a), int(b or a) + 1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpus", type = str, default = None, help = "CPU list to confine to, e.g. 64-127,192-255")
    ap.add_argument("--threads", type = str, default = "0", help = "Comma-separated pool sizes (0: all allowed CPUs)")
    ap.add_argument("--experts", type = int, default = 288)
    ap.add_argument("--hidden", type = int, default = 4096)
    ap.add_argument("--interm", type = int, default = 2048)
    ap.add_argument("--k_gate_up", type = int, default = 6)
    ap.add_argument("--k_down", type = int, default = 4)
    ap.add_argument("--decode_experts", type = str, default = "4,8")
    ap.add_argument("--rows", type = str, default = "512,2048")
    ap.add_argument("--topk", type = int, default = 8)
    ap.add_argument("--iters", type = int, default = 200, help = "Decode tokens timed per point")
    args = ap.parse_args()

    if args.cpus:
        os.sched_setaffinity(0, parse_cpus(args.cpus))
    torch.set_num_threads(1)
    from exllamav3.ext import exllamav3_ext as ext

    allowed = len(os.sched_getaffinity(0))
    E, H, I = args.experts, args.hidden, args.interm
    swz = int(any(ext.exl3_moe_cpu_swizzle_group(float(K)) for K in (args.k_gate_up, args.k_down)))
    gen = torch.Generator().manual_seed(0)

    def mats(k, n, K):
        # Random bits are valid packed data in either layout; only the byte count matters here
        t, u, v = [], [], []
        for _ in range(E):
            t.append(torch.randint(-32768, 32767, (k // 16, n // 16, 16 * K), dtype = torch.int16, generator = gen))
            u.append((torch.randn(k, generator = gen) * 0.015).half())
            v.append(torch.randn(n, generator = gen).half())
        return t, u, v

    t0 = time.time()
    g = mats(H, I, args.k_gate_up)
    u = mats(H, I, args.k_gate_up)
    d = mats(I, H, args.k_down)
    handle = ext.exl3_moe_cpu_make_layer(*g, *u, *d, [], [], [], 0, 1.0, swz)
    gb = sum(x.numel() * 2 for m in (g, u, d) for x in m[0]) / 1e9
    print(f" -- layer: {E} experts, {gb:.2f} GB trellis, built in {time.time() - t0:.1f} s, "
          f"{allowed} CPUs allowed, tier amx={ext.exl3_moe_cpu_has_amx()}, "
          f"pin={os.environ.get('EXL3_MOE_CPU_PIN', '0')}")
    del g, u, d

    try:
        for threads in [int(x) for x in args.threads.split(",")]:
            nt = threads or allowed
            for de in [int(x) for x in args.decode_experts.split(",")]:
                x = torch.randn(1, H, generator = gen).half()
                w = torch.full((1, de), 1.0 / de, dtype = torch.half)
                out = torch.empty(1, H)
                sels = [torch.randperm(E, generator = gen)[:de].view(1, de) for _ in range(args.iters + 20)]
                for s in sels[:20]:
                    ext.exl3_moe_cpu_forward(handle, x, s, w, out, nt)
                t0 = time.perf_counter()
                for s in sels[20:]:
                    ext.exl3_moe_cpu_forward(handle, x, s, w, out, nt)
                dt = (time.perf_counter() - t0) / args.iters
                print(f" threads {nt:4d}  decode  {de:2d} experts/token : {dt * 1e6:8.1f} us/layer, "
                      f"{dt * 1e6 / de:7.1f} us/expert")
            for rows in [int(x) for x in args.rows.split(",")]:
                x = torch.randn(rows, H, generator = gen).half()
                sel = torch.stack([torch.randperm(E, generator = gen)[:args.topk] for _ in range(rows)])
                w = torch.full((rows, args.topk), 1.0 / args.topk, dtype = torch.half)
                out = torch.empty(rows, H)
                ext.exl3_moe_cpu_forward(handle, x, sel, w, out, nt)
                reps = 3
                t0 = time.perf_counter()
                for _ in range(reps):
                    ext.exl3_moe_cpu_forward(handle, x, sel, w, out, nt)
                dt = (time.perf_counter() - t0) / reps
                print(f" threads {nt:4d}  prefill {rows:5d} rows         : {dt * 1e3:8.2f} ms/layer, "
                      f"{rows / dt:9.0f} tok/s/layer")
    finally:
        ext.exl3_moe_cpu_free_layer(handle)


if __name__ == "__main__":
    main()

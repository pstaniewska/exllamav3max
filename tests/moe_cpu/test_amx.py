"""
CPU expert kernels, AMX tier (cpu/moe_mul1.cpp, amx_tiles): the tile product computes the VNNI kernel's integer
sum exactly and shares its epilogue, so whole-layer outputs must be bit-identical to the vector tier it builds on
(vbmi, or vnni on a CPU without VBMI), at every rate, layout, chunk size and thread count. The AMX tier also takes
larger chunks (up to 16 tokens per expert), so the cases cover chunks that end mid-tile, wide (two-row) tokens,
and passes of every row count down to 1 (EXL3_MOE_CPU_AMX_MIN_ROWS = 1).

Needs an AMX-INT8 CPU and a kernel that grants the tile state (Linux 5.16+); skipped elsewhere.
"""
import pytest
import torch

from testlib.exl3 import rand_experts
from testlib.moe_cpu import cpu_runtime, run_per_tier, supported_tiers, swizzle   # noqa: F401 (fixture)

pytestmark = [pytest.mark.nogpu, pytest.mark.cpu_flags("avx2"), pytest.mark.usefixtures("cpu_runtime")]

RATES = (1, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6, 7, 8)
E, H, I, TOPK = 3, 512, 256, 2
TOKENS = (1, 3, 4, 7, 16, 17, 33, 100)


def layer_outputs() -> dict:
    """{(K, gated, spiky, tokens, threads): output} at this process's tier, layout per the tier's rule"""
    from exllamav3.ext import exllamav3_ext as ext
    gen = torch.Generator().manual_seed(4321)
    results = {}
    for K in RATES:
        for gated in (True, False):
            ex = rand_experts(E, H, I, K, gen)
            sw = lambda p: [swizzle(t[0]) for t in ex[p]]
            swz = int(any(not torch.equal(a.flatten(), t[0].flatten()) for a, t in zip(sw("u"), ex["u"])))
            sc = lambda p, i: [t[i] for t in ex[p]]
            h = ext.exl3_moe_cpu_make_layer(
                sw("g") if gated else [], sc("g", 1) if gated else [], sc("g", 2) if gated else [],
                sw("u"), sc("u", 1), sc("u", 2), sw("d"), sc("d", 1), sc("d", 2),
                [], [], [], 0 if gated else 2, 0.0, swz)
            try:
                for spiky in (False, True):
                    for tokens in TOKENS:
                        x = torch.randn(tokens, H, generator = gen) * 0.05
                        if spiky:
                            # A large shared component in a few dimensions: most of the row rounds to zero
                            # in int8, so these tokens go wide (two GEMV rows each)
                            x[:, :3] += 40.0
                        x = x.half()
                        # Every token on the same experts: one expert's chunks run 16 (AMX) or 4 rows wide
                        sel = torch.randperm(E, generator = gen)[:TOPK].unsqueeze(0).repeat(tokens, 1).contiguous()
                        w = torch.rand(tokens, TOPK, generator = gen)
                        w = (w / w.sum(-1, keepdim = True)).half().contiguous()
                        for th in (1, 3):
                            out = torch.zeros(tokens, H, dtype = torch.float32)
                            ext.exl3_moe_cpu_forward(h, x, sel, w, out, th)
                            results[(K, gated, spiky, tokens, th)] = out.clone()
            finally:
                ext.exl3_moe_cpu_free_layer(h)
    return results


@pytest.mark.parametrize("min_rows", (None, 1))
def test_amx_matches_vector_tier_bit_for_bit(min_rows):
    tiers = supported_tiers()
    if "amx" not in tiers:
        pytest.skip("no AMX-INT8 (or the tile state is not granted) on this machine")
    base = "vbmi" if "vbmi" in tiers else "vnni"
    env = {} if min_rows is None else {"EXL3_MOE_CPU_AMX_MIN_ROWS": min_rows}
    outs = run_per_tier(layer_outputs, tiers = [base, "amx"], env = env)
    ref, got = outs[base], outs["amx"]
    assert ref.keys() == got.keys()
    for key, r in ref.items():
        g = got[key]
        assert torch.isfinite(g).all() and r.abs().max() > 0, key
        assert torch.equal(g, r), \
            f"amx differs from {base} at (K, gated, spiky, tokens, threads) = {key}: " \
            f"max abs diff {(g - r).abs().max().item():.3e}"

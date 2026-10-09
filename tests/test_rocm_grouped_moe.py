"""
ROCm coverage for the unified grouped MoE path at the Qwen3-235B-class decode shape
(hidden = 2560, top_k = 10, K3/mul1 routed experts, gated silu).

dev has no fork-specific grouped kernel: the shape runs through the unified fused MoE
(quant/exl3_moe.cu, RDNA inner rocm/quant/exl3_moe_inner_rdna.cuh) with the deterministic
slot + exl3_moe_gather accumulation that modules/block_sparse_mlp.py issues for the fused
tier (run_fused / gather, block_sparse_mlp.py:1221-1290). These tests drive
ext.exl3_moe + ext.exl3_moe_gather directly with the argument layout the module builds
(expert-sorted token/weight arrays, sentinel bucket for out-of-range ids, (concurrency,
rows, dim) temp workspaces) and check the result against a per-token torch oracle built
from ext.reconstruct / ext.hgemm / ext.had_r_128 (the reference contract salvaged from the
fork's test_hip_grouped_moe_k3.py and test_hip_missing_binding_fallbacks.py).

The fork's routing reference contracts (selected experts / routing weights vs the torch
top-k + softmax + per-expert-scale reference) run here against dev's native routing_std
kernel as the input contract for the oracle.
"""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest
import torch

if not (torch.version.hip and torch.cuda.is_available()):
    pytest.skip("ROCm tests", allow_module_level = True)

from exllamav3.ext import exllamav3_ext as ext
from exllamav3.modules import block_sparse_mlp_routing as routing_module
from exllamav3.util.backend import MOE_FUSED_ROWS

device = "cuda:0"

HIDDEN = 2560
INTERMEDIATES = (640, 768)
TOP_K = 10
NUM_EXPERTS = 16
K = 3  # K3 / mul1
# Fused-tier row capacity: the module sizes its temp buffers (C, fused_rows, .) with
# fused_rows = MOE_FUSED_ROWS (512 on ROCm -- the RDNA kernel tiles the rows itself), and the
# kernel indexes temps by that capacity regardless of the per-call expert counts
TEMP_ROWS = MOE_FUSED_ROWS


def _projection(k, n, seed):
    """Trellis + sign vectors for one projection of every expert. 0x2492 is a finite K3/mul1
    trellis cycle; per-expert input/output signs make the matrices distinct without invalid
    procedural-codebook states (trellis layout: (k // 16, n // 16, 16 * K) int16)."""
    words = (k // 16) * (n // 16) * (16 * K)
    tensors, suhs, svhs = [], [], []
    for expert in range(NUM_EXPERTS):
        stream = torch.full((words,), 0x2492, dtype = torch.int16, device = device)
        tensors.append(stream.view(k // 16, n // 16, 16 * K))
        gen = torch.Generator(device = device).manual_seed(seed * 100 + expert)
        suhs.append((torch.randint(0, 2, (k,), generator = gen, device = device) * 2 - 1).half())
        svhs.append((torch.randint(0, 2, (n,), generator = gen, device = device) * 2 - 1).half())
    return tensors, suhs, svhs


@pytest.fixture(scope = "module", params = INTERMEDIATES, ids = lambda width: f"intermediate-{width}")
def synthetic_grouped_case(request):
    torch.manual_seed(1201 + request.param)
    gate = _projection(HIDDEN, request.param, 11)
    up = _projection(HIDDEN, request.param, 23)
    down = _projection(request.param, HIDDEN, 37)
    return request.param, gate, up, down


def _ptrs(tensors):
    # Pointer tables as MultiLinear builds them (modules/multilinear.py:32-34)
    return torch.tensor([t.data_ptr() for t in tensors], dtype = torch.long, device = device)


def _grouped_args(gate, up, down):
    args = []
    for projection in (gate, up, down):
        trellis, suhs, svhs = projection
        args += [_ptrs(trellis), _ptrs(suhs), _ptrs(svhs)]
    return args


def _linear_ref(x, projection, expert):
    trellis, suhs, svhs = projection
    k, n = x.shape[-1], svhs[expert].numel()
    xh = torch.empty_like(x)
    y = torch.empty((1, n), dtype = torch.float16, device = device)
    w = torch.empty((k, n), dtype = torch.float16, device = device)
    ext.had_r_128(x, xh, suhs[expert], None, 1.0)
    ext.reconstruct(w, trellis[expert], K, False, True)
    ext.hgemm(xh, w, y)
    ext.had_r_128(y, y, None, svhs[expert], 1.0)
    return y


def _oracle(x, selected, weights, gate, up, down):
    """Per-token reference: silu(gate) * up, down projection, weighted; assignments summed
    per token in expert-sorted order (duplicate experts retain routing-slot order)."""
    results = []
    for token in range(x.shape[0]):
        assignments = []
        token_x = x[token:token + 1]
        for slot in range(TOP_K):
            expert = int(selected[token, slot])
            g = _linear_ref(token_x, gate, expert)
            u = _linear_ref(token_x, up, expert)
            a = torch.nn.functional.silu(g.float()).half() * u
            d = _linear_ref(a, down, expert).float()
            assignments.append((expert, slot, d * weights[token, slot].float()))
        assignments.sort(key = lambda item: (item[0], item[1]))
        result = torch.zeros_like(assignments[0][2])
        for _, _, assignment in assignments:
            result = result + assignment
        results.append(result)
    return torch.cat(results, dim = 0)


def _run_unified(x, selected, weights, gate, up, down):
    """Mirror of the module's fused-tier dispatch (block_sparse_mlp.py:1150-1290):
    expert-sorted token/weight arrays, sentinel bucket for out-of-range ids, deterministic
    slot scratch (every assignment owns slot_base[e] + rank) and one gather per call."""
    rows = x.shape[0]
    flat_expert = selected.reshape(-1)
    flat_weight = weights.reshape(-1)
    # Out-of-range ids map to the sentinel bucket (num_experts), exactly as the module's
    # local-expert mapping does; the kernel never processes the sentinel bucket
    valid = (flat_expert >= 0) & (flat_expert < NUM_EXPERTS)
    flat_expert_local = torch.where(valid, flat_expert, torch.full_like(flat_expert, NUM_EXPERTS))
    flat_token = torch.arange(rows, device = device).repeat_interleave(TOP_K)

    order = flat_expert_local.argsort(stable = True)
    token_sorted = flat_token[order]
    weight_sorted = flat_weight[order]
    expert_count = torch.bincount(flat_expert_local, minlength = NUM_EXPERTS + 1)
    inv_order = torch.empty_like(order).scatter_(
        0, order, torch.arange(flat_expert_local.shape[0], device = device))

    counts = expert_count[:NUM_EXPERTS]
    fused = (counts > 0) & (counts <= TEMP_ROWS)
    assert bool(fused[counts > 0].all()), "test shapes must keep every real expert in the fused tier"
    slot_base = torch.cumsum(torch.where(fused, counts, torch.zeros_like(counts)), 0) - counts
    n_slots = int(counts[fused].sum())
    kind = fused.long()
    scratch = torch.empty((n_slots, HIDDEN), dtype = torch.float, device = device)
    output = torch.zeros((rows, HIDDEN), dtype = torch.float, device = device)

    concurrency = ext.exl3_moe_max_concurrency(0)
    temp_state_g = torch.empty((concurrency, TEMP_ROWS, HIDDEN), dtype = torch.half, device = device)
    temp_state_u = torch.empty((concurrency, TEMP_ROWS, HIDDEN), dtype = torch.half, device = device)
    intermediate = gate[2][0].numel()  # output-side signs of the gate projection = intermediate dim
    temp_intermediate_g = torch.empty((concurrency, TEMP_ROWS, intermediate), dtype = torch.half, device = device)
    temp_intermediate_u = torch.empty((concurrency, TEMP_ROWS, intermediate), dtype = torch.half, device = device)

    ext.exl3_moe(
        x, output, expert_count, token_sorted, weight_sorted,
        temp_state_g, temp_state_u, temp_intermediate_g, temp_intermediate_u,
        0,                  # act_function: MOE_ACT_SILU
        float(K), float(K), float(K),
        *_grouped_args(gate, up, down),
        False, True,        # gate: mcg / mul1
        False, True,        # up
        False, True,        # down
        0.0,                # act_limit (silu is unclamped)
        int(fused.sum()),   # num_active: counted, as the module's fused tier does
        scratch, slot_base, # deterministic slot accumulation
        1, TEMP_ROWS, 16,   # count_lo, count_hi, m_tile
    )
    ext.exl3_moe_gather(
        output, scratch, flat_expert_local, inv_order,
        expert_count[:NUM_EXPERTS].cumsum(0) - counts, slot_base, kind, weight_sorted)
    return output


def test_unified_moe_bindings_exist():
    for name in ("exl3_moe", "exl3_moe_gather", "exl3_moe_max_concurrency"):
        assert hasattr(ext, name), f"missing binding: {name}"


@pytest.mark.parametrize(
    "ids",
    [
        pytest.param([9, 2, 15, 1, 12, 7, 4, 0, 11, 5], id = "shuffled"),
        pytest.param([7, 2, 7, 1, 2, 7, 4, 1, 9, 2], id = "duplicates"),
    ],
)
@pytest.mark.parametrize("rows", [1, 3, 5, 8, 12, 16])
@torch.inference_mode()
def test_unified_moe_matches_reconstruction_oracle(synthetic_grouped_case, ids, rows):
    _, gate, up, down = synthetic_grouped_case
    x = torch.randn((rows, HIDDEN), dtype = torch.float16, device = device) * 1e-3
    selected = torch.stack([
        torch.tensor(ids[row:] + ids[:row], dtype = torch.long, device = device)
        for row in range(rows)
    ])
    weights = torch.rand((rows, TOP_K), dtype = torch.float16, device = device)
    weights /= weights.sum(dim = -1, keepdim = True)
    expected = _oracle(x, selected, weights, gate, up, down)
    actual = _run_unified(x, selected, weights, gate, up, down)
    assert torch.isfinite(actual).all()
    # Relative contract as upstream's own MoE tests use (test_moe_coop.py: err <= 3e-3 * scale):
    # the fp16 pipeline's noise floor scales with output magnitude (Hadamard DC rows), so an
    # absolute atol alone is effectively a relative check at ~1e-5 and can never pass at scale
    scale = expected.abs().max()
    torch.testing.assert_close(actual, expected, rtol = 0, atol = max(0.04, 3e-3 * scale))


@torch.inference_mode()
def test_unified_moe_is_deterministic(synthetic_grouped_case):
    """Deterministic slot + gather accumulation must be bit-reproducible run to run."""
    _, gate, up, down = synthetic_grouped_case
    rows = 16
    x = torch.randn((rows, HIDDEN), dtype = torch.float16, device = device) * 1e-3
    ids = [7, 2, 7, 1, 2, 7, 4, 1, 9, 2]
    selected = torch.stack([
        torch.tensor(ids[row:] + ids[:row], dtype = torch.long, device = device)
        for row in range(rows)
    ])
    weights = torch.rand((rows, TOP_K), dtype = torch.float16, device = device)
    weights /= weights.sum(dim = -1, keepdim = True)
    first = _run_unified(x, selected, weights, gate, up, down).clone()
    second = _run_unified(x, selected, weights, gate, up, down).clone()
    torch.testing.assert_close(first, second, rtol = 0, atol = 0)


@torch.inference_mode()
def test_unified_moe_output_rows_are_isolated(synthetic_grouped_case):
    _, gate, up, down = synthetic_grouped_case
    rows = 8
    x = torch.randn((rows, HIDDEN), dtype = torch.float16, device = device) * 1e-3
    selected = torch.tensor(
        [[(row * 3 + slot * 5) % NUM_EXPERTS for slot in range(TOP_K)] for row in range(rows)],
        dtype = torch.long, device = device,
    )
    weights = torch.rand((rows, TOP_K), dtype = torch.float16, device = device)
    weights /= weights.sum(dim = -1, keepdim = True)
    baseline = _run_unified(x, selected, weights, gate, up, down).clone()

    changed_x = x.clone()
    changed_selected = selected.clone()
    changed_weights = weights.clone()
    changed_row = rows - 1
    changed_x[changed_row].mul_(-3)
    changed_selected[changed_row] = changed_selected[changed_row].roll(3)
    changed_weights[changed_row] = changed_weights[changed_row].roll(1)
    changed = _run_unified(changed_x, changed_selected, changed_weights, gate, up, down).clone()

    assert not torch.equal(changed[changed_row], baseline[changed_row])
    keep = torch.tensor([row for row in range(rows) if row != changed_row], dtype = torch.long, device = device)
    torch.testing.assert_close(
        changed.index_select(0, keep), baseline.index_select(0, keep), rtol = 0, atol = 0)


@torch.inference_mode()
def test_unified_moe_concentrated_duplicate_slots(synthetic_grouped_case):
    """All slots of every token may target one expert (40 assignments, within the fused
    tier's per-expert row capacity) without cross-row loss or aliasing."""
    _, gate, up, down = synthetic_grouped_case
    rows = 4
    x_gen = torch.Generator(device = device).manual_seed(3201)
    w_gen = torch.Generator(device = device).manual_seed(3202)
    x = torch.randn((rows, HIDDEN), dtype = torch.float16, device = device, generator = x_gen) * 1e-2
    selected = torch.zeros((rows, TOP_K), dtype = torch.long, device = device)
    weights = torch.rand((rows, TOP_K), dtype = torch.float16, device = device, generator = w_gen)
    weights /= weights.sum(dim = -1, keepdim = True)
    expected = _oracle(x, selected, weights, gate, up, down)
    actual = _run_unified(x, selected, weights, gate, up, down)
    assert torch.isfinite(actual).all()
    scale = expected.abs().max()
    torch.testing.assert_close(actual, expected, rtol = 0, atol = max(0.04, 3e-3 * scale))


@torch.inference_mode()
def test_out_of_range_expert_ids_contribute_zero(synthetic_grouped_case):
    """Ids outside [0, num_experts) map to the sentinel bucket: they are neither computed nor
    gathered, so a fully invalid routing leaves the output at zero with clean workspaces."""
    _, gate, up, down = synthetic_grouped_case
    x = torch.randn((1, HIDDEN), dtype = torch.float16, device = device) * 1e-3
    weights = torch.full((1, TOP_K), 1 / TOP_K, dtype = torch.float16, device = device)
    invalid = torch.tensor(
        [[-1, NUM_EXPERTS, -99, NUM_EXPERTS + 1, -2, 999, -8, 77, -4, 1024]],
        dtype = torch.long, device = device,
    )
    actual = _run_unified(x, invalid, weights, gate, up, down)
    torch.testing.assert_close(actual, torch.zeros_like(actual), rtol = 0, atol = 0)


@torch.inference_mode()
def test_moe_argument_contracts_are_enforced(synthetic_grouped_case):
    """The TORCH_CHECK contracts dev's unified entry point enforces before launch."""
    _, gate, up, down = synthetic_grouped_case
    rows = 1
    x = torch.randn((rows, HIDDEN), dtype = torch.float16, device = device)
    selected = torch.arange(TOP_K, dtype = torch.long, device = device).view(1, -1)
    weights = torch.full((1, TOP_K), 1 / TOP_K, dtype = torch.float16, device = device)
    concurrency = ext.exl3_moe_max_concurrency(0)
    temps = (
        torch.empty((concurrency, TEMP_ROWS, HIDDEN), dtype = torch.half, device = device),
        torch.empty((concurrency, TEMP_ROWS, HIDDEN), dtype = torch.half, device = device),
        torch.empty((concurrency, TEMP_ROWS, 640), dtype = torch.half, device = device),
        torch.empty((concurrency, TEMP_ROWS, 640), dtype = torch.half, device = device),
    )
    output = torch.zeros((rows, HIDDEN), dtype = torch.float, device = device)
    counts = torch.bincount(selected.reshape(-1), minlength = NUM_EXPERTS + 1)
    ptr_args = tuple(_grouped_args(gate, up, down))
    mul1 = (False, True, False, True, False, True)

    def call(hidden, codebooks, scratch = None, fused_base = None):
        ext.exl3_moe(
            hidden, output, counts, selected.reshape(-1), weights.reshape(-1),
            *temps, 0, 3.0, 3.0, 3.0, *ptr_args, *codebooks,
            0.0, int((counts > 0).sum()), scratch, fused_base, 1, TEMP_ROWS, 16)

    # fp32 hidden state rejected
    with pytest.raises(RuntimeError):
        call(x.float(), mul1)
    # Mixed codebooks across gate/up/down rejected (down loses mul1)
    with pytest.raises(RuntimeError):
        call(x, (False, True, False, True, False, False))
    # output_scratch without fused_base rejected
    with pytest.raises(RuntimeError):
        call(x, mul1, scratch = torch.zeros((rows * TOP_K, HIDDEN), dtype = torch.float, device = device))


# ---------------------------------------------------------------------------
# Routing contract (salvaged from the fork's test_hip_missing_binding_fallbacks.py,
# minus the monkeypatch scaffolding): dev's native routing_std kernel against the torch
# top-k / softmax / per-expert-scale reference — the input contract of the oracle above.
# ---------------------------------------------------------------------------

def _routing_cfg(num_experts = NUM_EXPERTS, top_k = TOP_K):
    gen = torch.Generator(device = device).manual_seed(42)
    # Tie-free gate: distinct, well-spaced logits per row so the selected set is unambiguous.
    # Layout matches the module's cfg.gate_tensor = routing_gate.inner.weight: (hidden, experts)
    gate = (torch.rand((HIDDEN, num_experts), generator = gen, device = device) * 0.02 - 0.01).half()
    return routing_module.RoutingCFG(
        gate_tensor = gate,
        gate_tensor_t = None,
        num_experts = num_experts,
        num_experts_per_tok = top_k,
        router_logits_bsz1 = torch.empty((1, num_experts), dtype = torch.half, device = device),
        routing_weights_bsz1 = torch.empty((1, top_k), dtype = torch.half, device = device),
        selected_experts_bsz1 = torch.empty((1, top_k), dtype = torch.long, device = device),
        e_score_correction_bias = None,
        e_score_bias_h = None,
        routed_scaling_factor = None,
        n_group = None,
        topk_group = None,
        per_expert_scale = (torch.rand(num_experts, generator = gen, device = device) * 0.5 + 0.75).bfloat16().to(device),
    )


@torch.inference_mode()
def test_routing_std_matches_torch_reference():
    rows = 8
    cfg = _routing_cfg()
    y = torch.randn((rows, HIDDEN), dtype = torch.float16, device = device) * 0.1
    selected, weights = routing_module.routing_std(rows, cfg, y, {})

    logits = (y @ cfg.gate_tensor).float()  # (rows, E): gate is (H, E)
    top_v, expected_selected = torch.topk(logits, TOP_K, dim = -1)
    expected_weights = torch.softmax(top_v, dim = -1)
    expected_weights *= cfg.per_expert_scale.float()[expected_selected]
    i1, o1 = selected.sort(dim = 1)
    i2, o2 = expected_selected.sort(dim = 1)
    assert torch.equal(i1, i2), "routing_std selected a different expert set"
    err = (weights.float().gather(1, o1) - expected_weights.half().float().gather(1, o2)).abs().max().item()
    assert err < 4e-3, f"routing_std weight error {err}"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))

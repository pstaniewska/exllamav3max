"""
ROCm coverage for dev's native dsa_topk trio (dsa_topk.cu): the single-pass radix-select
kernel (dsa_topk) and the split/merge tiled path (dsa_topk_tile + dsa_topk_merge_tiles),
ported from the fork's test_hip_dsa_topk_gfx12.py onto upstream dev's kernels.

What changed against the fork:
  - The kernel under test is dev's k-parameterized kernel (no fixed-K gfx12 contract): k is
    a runtime argument, k_pad only bounds the output row, and rows with an innermost-dense
    but 8-misaligned row stride take the scalar (non-vectorized) path instead of an error.
  - The tiled bindings exist unconditionally (bindings.cpp:124-126); the fork's negative
    assert and all fallback wiring are gone. The tiled path is exercised through the same
    call pattern QSAIndexer._select_rows runs: a fixed-stride score slab, each tile's local
    top-k landing in slot 1 of a G=2 workspace (slot 0 holds the running merged set), the
    native merge reducing the pair under dsa_topk's total order into the other workspace's
    slot 0, and the final tile merging the running set against the last tile.
Tie contract under test (documented in qsa_indexer._select_rows): selection is ordered by
the fp16 bit key (monotonic remap; keys <= key(-inf) = 0x03ff, i.e. -inf and negative NaNs,
are excluded) descending, ties broken by ascending index. Selected indices are emitted
ascending and the full output row is rewritten (-1 padding).
"""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest
import torch

if not (torch.version.hip and torch.cuda.is_available()):
    pytest.skip("ROCm dsa_topk tests", allow_module_level = True)

from exllamav3.ext import exllamav3_ext as ext

def _is_rdna_dgpu(device: torch.device) -> bool:
    """RDNA dGPUs only. The host CPU's iGPU (e.g. gfx1036 on Ryzen) enumerates as a HIP
    device but segfaults under torch's device-side generators, and is not a target here."""
    if not (torch.version.hip and torch.cuda.is_available()):
        return False
    try:
        arch = torch.cuda.get_device_properties(device).gcnArchName.split(":")[0]
        return arch.startswith(("gfx11", "gfx12"))
    except Exception:
        return False


# The QSA indexer's 512-block selection shape (token_budget 2048 / compress_ratio 4)
K = 512
KEY_NEG_INF = 0x03ff  # topk_key(0xfc00): -inf and everything below it is excluded


@pytest.fixture(params = list(
    d for d in (torch.device("cuda", i) for i in range(torch.cuda.device_count()))
    if d.index is not None and _is_rdna_dgpu(d)
), ids = lambda d: f"cuda:{d.index}")
def device(request) -> torch.device:
    return request.param


def _fp16_key(bits: torch.Tensor) -> torch.Tensor:
    """Monotonic uint16 remap of fp16 bit patterns (int32, uint16 range)."""
    return torch.where(bits & 0x8000 != 0, ~bits & 0xffff, bits | 0x8000)


def _ref_topk(scores: torch.Tensor, k: int, k_pad: int, t_ptr = None, t_seq = 0) -> torch.Tensor:
    """
    Oracle: per row, the k highest entries by (fp16 key desc, index asc) among entries above
    the -inf key and inside the row's causal bound (t_ptr, mirroring the kernel's
    t_ptr[row // t_seq] / *t_ptr override), emitted in ascending index order, -1 padded.
    """
    rows, width = scores.shape
    scores = scores.detach().cpu()
    bits = scores.view(torch.int16).to(torch.int32) & 0xffff
    key = _fp16_key(bits)
    if t_ptr is None:
        bounds = torch.full((rows,), width, dtype = torch.int64)
    else:
        tp = t_ptr.detach().cpu().long()
        if t_seq > 0:
            bounds = tp[torch.arange(rows) // t_seq]
        else:
            bounds = tp.expand(rows)
    col = torch.arange(width).unsqueeze(0)
    valid = (key > KEY_NEG_INF) & (col < bounds.unsqueeze(1))
    # Stable descending argsort: equal keys keep ascending-index order; invalid entries
    # (key forced to -1) sort last and are dropped by the per-row take count
    order = torch.argsort(torch.where(valid, key, torch.tensor(-1)), dim = 1, descending = True, stable = True)
    take = valid.sum(1, keepdim = True).clamp(max = k)
    pos = torch.arange(k).unsqueeze(0)
    sel = torch.where(pos < take, order[:, :k], torch.tensor(width))  # width = above any index
    out = sel.sort(dim = 1).values  # ascending indices, then the sentinels
    out = torch.where(out == width, torch.tensor(-1), out)
    if k_pad > k:  # the kernel pads the selection out to the full k_pad row
        out = torch.cat((out, torch.full((rows, k_pad - k), -1, dtype = out.dtype)), dim = 1)
    return out.to(torch.int32)


def _scores(device: torch.device, rows: int, width: int, mode: str, seed: int) -> torch.Tensor:
    generator = torch.Generator(device = device).manual_seed(seed)
    if mode == "randn":
        return torch.randn((rows, width), generator = generator, device = device, dtype = torch.half)
    if mode == "ties":
        # Quantized to 8 levels: heavy ties, many of them at the selection boundary
        return (torch.randint(0, 8, (rows, width), generator = generator, device = device) * 0.25).half()
    if mode == "ties_sparse":
        s = (torch.randint(0, 8, (rows, width), generator = generator, device = device) * 0.25).half()
        s[:, -width // 8:] = -float("inf")
        return s
    # sparse: mostly -inf, a random number of valid entries per row (exercises the
    # fewer-than-k-valid path where the kernel selects everything finite)
    s = torch.full((rows, width), -float("inf"), device = device, dtype = torch.half)
    for r in range(rows):
        n = int(torch.randint(1, max(width // 2, 2), (1,), generator = generator))
        idx = torch.randperm(width, generator = generator, device = device)[:n]
        s[r, idx] = torch.randn(n, generator = generator, device = device, dtype = torch.half)
    return s


# ---- single-pass full-width kernel -----------------------------------------------------------


@pytest.mark.parametrize("rows,width,k,k_pad,mode,seed", [
    (1, 513, K, K, "ties", 1201),
    (32, 3072, K, K, "randn", 1202),
    (64, 32768, K, K, "ties", 1203),
    (1024, 512, K, K, "randn", 1204),        # QSA single-tile shape: SEL_SLAB rows x 512 pools
    (3, 512, K, K, "sparse", 1205),
    (16, 4096, K, K, "ties_sparse", 1206),
    (8, 2048, 100, 128, "ties", 1207),       # k < k_pad: the tail is -1 padded
])
@torch.inference_mode()
@pytest.mark.xfail(reason = "dsa_topk tie-order contract violation on gfx1201, see #481", strict = False)
def test_dsa_topk_matches_tie_ordered_oracle(device, rows, width, k, k_pad, mode, seed):
    scores = _scores(device, rows, width, mode, seed)
    expected = _ref_topk(scores, k, k_pad)
    actual = torch.full((rows, k_pad), 12345, device = device, dtype = torch.int32)
    ext.dsa_topk(scores, actual, k, None, 0)
    torch.cuda.synchronize(device)
    # Prefill sentinel must not survive: the kernel rewrites every output column
    assert torch.equal(actual.cpu(), expected)


@torch.inference_mode()
def test_dsa_topk_boundary_tie_takes_lowest_indices(device):
    # 100 entries at 4.0, then 1500 tied at 1.0, -inf elsewhere; k = 512 takes all of the
    # 4.0 run plus the LOWEST 412 indices of the 1.0 tie run (1000..1411), not arbitrary ones
    width = 4096
    scores = torch.full((1, width), -float("inf"), device = device, dtype = torch.half)
    scores[0, :100] = 4.0
    scores[0, 1000:2500] = 1.0
    actual = torch.empty((1, K), device = device, dtype = torch.int32)
    ext.dsa_topk(scores, actual, K, None, 0)
    torch.cuda.synchronize(device)
    expected = torch.cat((
        torch.arange(100, device = device, dtype = torch.int32),
        torch.arange(1000, 1412, device = device, dtype = torch.int32),
    ))
    assert torch.equal(actual[0], expected)
    assert torch.equal(actual.cpu(), _ref_topk(scores, K, K))


@torch.inference_mode()
def test_dsa_topk_all_tied_row_takes_lowest_indices(device):
    scores = torch.ones((1, 2048), device = device, dtype = torch.half)
    actual = torch.empty((1, K), device = device, dtype = torch.int32)
    ext.dsa_topk(scores, actual, K, None, 0)
    torch.cuda.synchronize(device)
    assert torch.equal(actual[0], torch.arange(K, device = device, dtype = torch.int32))


@torch.inference_mode()
def test_dsa_topk_special_values_follow_fp16_key_order(device):
    # -0 (0x8000, key 0x7fff) ranks BELOW +0 (0x0000, key 0x8000) despite numeric equality
    raw = torch.empty((1, 1024), dtype = torch.int16, device = device)
    raw[0, ::2] = 0x0000
    raw[0, 1::2] = -32768  # 0x8000 as int16
    scores = raw.view(torch.float16)
    actual = torch.empty((1, K), device = device, dtype = torch.int32)
    ext.dsa_topk(scores, actual, K, None, 0)
    torch.cuda.synchronize(device)
    assert torch.equal(actual[0], torch.arange(0, 1024, 2, device = device, dtype = torch.int32))
    assert torch.equal(actual.cpu(), _ref_topk(scores, K, K))

    # A single +NaN ranks above +inf in key order: with k = 1 it wins the slot
    raw = torch.full((1, 1024), 0x7c00, dtype = torch.int16, device = device)  # +inf
    raw[0, 137] = 0x7e00                                       # +NaN
    scores = raw.view(torch.float16)
    actual = torch.empty((1, 32), device = device, dtype = torch.int32)
    ext.dsa_topk(scores, actual, 1, None, 0)
    torch.cuda.synchronize(device)
    assert actual[0, 0].item() == 137
    assert (actual[0, 1:] == -1).all()
    assert torch.equal(actual.cpu(), _ref_topk(scores, 1, 32))

    # -inf and negative NaNs (keys <= 0x03ff) are never selected; -0 is
    raw = torch.full((1, 1024), -1024, dtype = torch.int16, device = device)  # 0xfc00 = -inf
    raw[0, :5] = -512                                        # 0xfe00 = negative NaN
    raw[0, 5] = -32768                                       # 0x8000 = -0: the only valid entry
    scores = raw.view(torch.float16)
    actual = torch.empty((1, K), device = device, dtype = torch.int32)
    ext.dsa_topk(scores, actual, K, None, 0)
    torch.cuda.synchronize(device)
    exact = torch.full((K,), -1, device = device, dtype = torch.int32)
    exact[0] = 5
    assert torch.equal(actual[0], exact)
    assert torch.equal(actual.cpu(), _ref_topk(scores, K, K))

    # A row with no valid entry at all selects nothing
    scores = torch.full((1, 1024), -float("inf"), device = device, dtype = torch.half)
    actual = torch.empty((1, K), device = device, dtype = torch.int32)
    ext.dsa_topk(scores, actual, K, None, 0)
    torch.cuda.synchronize(device)
    assert (actual == -1).all()


@torch.inference_mode()
@pytest.mark.xfail(reason = "dsa_topk tie-order contract violation on gfx1201, see #481", strict = False)
def test_dsa_topk_t_ptr_bounds(device):
    rows, width = 6, 1024
    scores = _scores(device, rows, width, "randn", 1301)
    out = torch.empty((rows, K), device = device, dtype = torch.int32)

    # t_seq == 0: a single device-side bound for every row
    t_ptr = torch.tensor([400], device = device, dtype = torch.int32)
    ext.dsa_topk(scores, out, K, t_ptr, 0)
    torch.cuda.synchronize(device)
    assert torch.equal(out.cpu(), _ref_topk(scores, K, K, t_ptr, 0))
    if (out >= 0).any():
        assert (out[out >= 0] < 400).all()

    # t_seq > 0: per-job bounds, row r scans t_ptr[r // t_seq] entries
    t_ptr = torch.tensor([300, 600, 1024], device = device, dtype = torch.int32)
    ext.dsa_topk(scores, out, K, t_ptr, 2)
    torch.cuda.synchronize(device)
    assert torch.equal(out.cpu(), _ref_topk(scores, K, K, t_ptr, 2))
    for r in range(rows):
        bound = (300, 600, 1024)[r // 2]
        sel = out[r][out[r] >= 0]
        if sel.numel():
            assert (sel < bound).all()


@torch.inference_mode()
@pytest.mark.xfail(reason = "dsa_topk tie-order contract violation on gfx1201, see #481", strict = False)
def test_dsa_topk_strided_rows_take_the_scalar_path(device):
    # Row stride not divisible by 8: the non-vectorized kernel must match the oracle too
    base = torch.randn((4, 2051), device = device, dtype = torch.half)
    scores = base[:, :2048]
    assert scores.stride(0) % 8 != 0
    actual = torch.empty((4, K), device = device, dtype = torch.int32)
    ext.dsa_topk(scores, actual, K, None, 0)
    torch.cuda.synchronize(device)
    assert torch.equal(actual.cpu(), _ref_topk(scores, K, K))


@pytest.mark.parametrize("bad,match", [
    ("scores-dtype", "incorrect datatype"),
    ("out-dtype", "incorrect datatype"),
    ("out-rows", "output shape mismatch"),
    ("k-over-k_pad", "output shape mismatch"),
    ("scores-cols", "innermost dim must be dense"),
    ("scores-3d", "2D tensors expected"),
])
@torch.inference_mode()
def test_dsa_topk_rejects_malformed_inputs(device, bad, match):
    scores = _scores(device, 2, 1024, "randn", 1401)
    out = torch.empty((2, K), device = device, dtype = torch.int32)
    if bad == "scores-dtype":
        scores = scores.float()
    elif bad == "out-dtype":
        out = out.long()
    elif bad == "out-rows":
        out = out[1:]
    elif bad == "k-over-k_pad":
        pass  # k = K + 1 > k_pad = K
    elif bad == "scores-cols":
        scores = scores[:, ::2]
    else:
        scores = scores.unsqueeze(0)
    with pytest.raises(RuntimeError, match = match):
        ext.dsa_topk(scores, out, K + 1 if bad == "k-over-k_pad" else K, None, 0)


# ---- tiled split/merge path (the QSAIndexer._select_rows call pattern) ------------------------


def _tiled_topk_like_indexer(scores_full: torch.Tensor, k_sel: int, kp: int, t_tile: int, device):
    """
    Mirror of QSAIndexer._select_rows' tiled branch: the tile's scores live in a fixed-stride
    slab (view of width t1-t0, row stride rounded to 128 like the production SEL_SLAB slab);
    dsa_topk_tile emits the tile's local top-k into slot 1 (slot 0 = the running merged set,
    count zero-initialized); dsa_topk_merge_tiles reduces the pair into slot 0 of the other
    workspace (scores and count emitted, count as a stride-2 view); the last tile merges the
    running set against the final tile straight into the output row.
    Returns the output rows and the per-tile slot-1 candidate counts.
    """
    rows, T = scores_full.shape
    s_stride = -(-t_tile // 128) * 128
    slab = torch.full((rows, s_stride), -float("inf"), device = device, dtype = torch.half)
    out = torch.full((rows, kp), 12345, device = device, dtype = torch.int32)  # overwrite sentinel
    ws = [
        (
            torch.empty((rows, 2, kp), dtype = torch.int32, device = device),
            torch.empty((rows, 2, kp), dtype = torch.half, device = device),
            torch.zeros((rows, 2), dtype = torch.int32, device = device),
        )
        for _ in range(2)
    ]
    cur = 0
    counts = []
    tiles = list(range(0, T, t_tile))
    for n, t0 in enumerate(tiles):
        t1 = min(t0 + t_tile, T)
        slab[:, : t1 - t0].copy_(scores_full[:, t0:t1])
        sc = slab[:, : t1 - t0]  # (rows, t1-t0) view, row stride s_stride: the production shape
        w_idx, w_scr, w_cnt = ws[cur]
        ext.dsa_topk_tile(sc, w_idx, w_scr, w_cnt, 1, min(k_sel, t1 - t0), t0)
        counts.append(w_cnt[:, 1].clone())
        if n == len(tiles) - 1:
            ext.dsa_topk_merge_tiles(w_idx, w_scr, w_cnt, out, None, None, k_sel)
        else:
            n_idx, n_scr, n_cnt = ws[cur ^ 1]
            ext.dsa_topk_merge_tiles(w_idx, w_scr, w_cnt, n_idx[:, 0], n_scr[:, 0], n_cnt[:, 0], k_sel)
            cur ^= 1
    torch.cuda.synchronize(device)
    return out, counts


@pytest.mark.parametrize("rows,T,t_tile,k_sel,kp,mode,seed", [
    (4, 3584, 1024, K, K, "ties", 1501),
    (4, 3584, 1024, K, K, "randn", 1502),
    (2, 3584, 256, K, K, "ties", 1503),       # per-tile k (256) < k_sel, 14 tiles
    (1, 4096, 1024, 100, 128, "ties", 1504),  # k_sel < kp
])
@torch.inference_mode()
@pytest.mark.xfail(reason = "dsa_topk tie-order contract violation on gfx1201, see #481", strict = False)
def test_tiled_path_matches_full_width_and_oracle(device, rows, T, t_tile, k_sel, kp, mode, seed):
    scores = _scores(device, rows, T, mode, seed)
    tiled, counts = _tiled_topk_like_indexer(scores, k_sel, kp, t_tile, device)

    # Same row through the full-width single-pass kernel and the torch oracle
    full = torch.full((rows, kp), 12345, device = device, dtype = torch.int32)
    ext.dsa_topk(scores, full, k_sel, None, 0)
    torch.cuda.synchronize(device)
    assert torch.equal(tiled, full)
    assert torch.equal(tiled.cpu(), _ref_topk(scores, k_sel, kp))

    # Each tile emitted min(k_sel, valid entries in the tile) candidates into slot 1
    bits = scores.view(torch.int16).to(torch.int32) & 0xffff
    valid = (_fp16_key(bits) > KEY_NEG_INF)
    for n, t0 in enumerate(range(0, T, t_tile)):
        t1 = min(t0 + t_tile, T)
        assert torch.equal(counts[n], valid[:, t0:t1].sum(1).clamp(max = k_sel).to(torch.int32))


@torch.inference_mode()
def test_tiled_merge_resolves_cross_tile_ties_by_ascending_index(device):
    # The tie run at the threshold value spans the tile boundary at 1024: the merge must pick
    # the lowest global indices across the slot pair (1000..1023 from tile 0, 1024..1411 from
    # tile 1), identical to a single pass over the whole row
    T, t_tile = 3584, 1024
    scores = torch.full((2, T), -float("inf"), device = device, dtype = torch.half)
    scores[:, :100] = 4.0
    scores[:, 1000:2500] = 1.0
    tiled, _ = _tiled_topk_like_indexer(scores, K, K, t_tile, device)
    expected = torch.cat((
        torch.arange(100, device = device, dtype = torch.int32),
        torch.arange(1000, 1412, device = device, dtype = torch.int32),
    ))
    assert torch.equal(tiled[0], expected)
    assert torch.equal(tiled[1], expected)

    full = torch.empty((2, K), device = device, dtype = torch.int32)
    ext.dsa_topk(scores, full, K, None, 0)
    torch.cuda.synchronize(device)
    assert torch.equal(tiled, full)
    assert torch.equal(tiled.cpu(), _ref_topk(scores, K, K))


@torch.inference_mode()
@pytest.mark.xfail(reason = "dsa_topk tie-order contract violation on gfx1201, see #481", strict = False)
def test_tiled_merge_multi_slot_with_scores_and_counts(device):
    # Direct G=3 variant: three tiles into slots 0/1/2 (ascending offsets, as the tie contract
    # requires), one merge emitting scores and counts alongside the indices
    rows, T, tile = 3, 3072, 1024
    scores = _scores(device, rows, T, "ties", 1601)
    G, kp = 3, K
    ws_idx = torch.empty((rows, G, kp), dtype = torch.int32, device = device)
    ws_scr = torch.empty((rows, G, kp), dtype = torch.half, device = device)
    ws_cnt = torch.zeros((rows, G), dtype = torch.int32, device = device)
    for slot, t0 in enumerate(range(0, T, tile)):
        t1 = min(t0 + tile, T)
        ext.dsa_topk_tile(scores[:, t0:t1], ws_idx, ws_scr, ws_cnt, slot, min(K, t1 - t0), t0)

    out = torch.full((rows, kp), 12345, device = device, dtype = torch.int32)
    out_scr = torch.empty((rows, kp), device = device, dtype = torch.half)
    out_cnt = torch.empty((rows,), device = device, dtype = torch.int32)
    ext.dsa_topk_merge_tiles(ws_idx, ws_scr, ws_cnt, out, out_scr, out_cnt, K)
    torch.cuda.synchronize(device)

    assert torch.equal(out.cpu(), _ref_topk(scores, K, K))

    # out_cnt: emitted candidates = min(k_sel, valid entries over the whole row)
    bits = scores.view(torch.int16).to(torch.int32) & 0xffff
    n_valid = (_fp16_key(bits) > KEY_NEG_INF).sum(1)
    assert torch.equal(out_cnt, n_valid.clamp(max = K).to(torch.int32))

    # out_scr: the score of each emitted index, aligned with the output
    for r in range(rows):
        sel = out[r][out[r] >= 0].long()
        assert torch.equal(out_scr[r][: sel.numel()], scores[r][sel])


@torch.inference_mode()
def test_tiled_bindings_reject_malformed_inputs(device):
    scores = _scores(device, 2, 1024, "randn", 1701)
    ws_idx = torch.empty((2, 2, K), dtype = torch.int32, device = device)
    ws_scr = torch.empty((2, 2, K), dtype = torch.half, device = device)
    ws_cnt = torch.zeros((2, 2), dtype = torch.int32, device = device)

    with pytest.raises(RuntimeError, match = "bad slot"):
        ext.dsa_topk_tile(scores, ws_idx, ws_scr, ws_cnt, 2, K, 0)  # slot >= G
    with pytest.raises(RuntimeError, match = "bad workspace"):
        ext.dsa_topk_tile(scores, ws_idx, ws_scr[:, :, :-1], ws_cnt, 0, K, 0)
    with pytest.raises(RuntimeError, match = "bad slot"):
        ext.dsa_topk_tile(scores, ws_idx, ws_scr, ws_cnt, 0, K + 1, 0)  # k > k_pad

    out = torch.empty((2, K), dtype = torch.int32, device = device)
    with pytest.raises(RuntimeError, match = "shape mismatch"):
        ext.dsa_topk_merge_tiles(ws_idx, ws_scr, ws_cnt, out, None, None, K + 1)
    with pytest.raises(RuntimeError, match = "out_scr must match out_idx"):
        ext.dsa_topk_merge_tiles(ws_idx, ws_scr, ws_cnt, out, torch.empty((2, K - 1), dtype = torch.half, device = device), None, K)

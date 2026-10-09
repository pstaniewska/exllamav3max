"""
ROCm coverage for dev's native hyperconnection kernels (hc_mix.cu / hc_mix_tiled.cu):
hc_mix, hc_head, hc_mix (full mHC), hc_apply (residual update, including dev's weighted-copy
form), gr_mix (GatedResidual decode mix, with and without a pre-weighted stream copy) and
gr_mix_tiled (the tiled deterministic prefill form dev added over the fork).

Every kernel is checked against a plain torch reference of its documented math. The fork's
wave32/arch gating and hc_mix_supported plumbing are gone: dev binds all of these
unconditionally on both backends and has no hc_mix_supported binding.

Contract facts relied on (hc_mix.cu / hc_mix.cuh / hc_mix_tiled.cu):
  - H is fixed at 4. hc_mix: fn is (2H + H^2, H*D) or (H, H*D) for the head form; collapsed
    is fp16 or fp32. hc_apply updates x in place, optionally also emitting xw = x_new * wn.
  - gr_mix: fn (M, H*D) half with M = LR (+ H with post); upt is the (H, D/4, LR, 4)
    repacked layout; w is (H*D) half; dots is (R, M + 1, H) with row M holding the raw
    per-stream sums of squares; wstreams (optional) replaces streams in the dots only.
  - gr_mix_tiled: H = 4, D % 128 == 0, LR % 64 == 0, proj padded to Mpad % 64 == 0 <= 512;
    proj/up arrive as (2, N, K) int8 + (N) fp32 row scales from det_quant_weight; workspaces
    sized from gr_mix_tiled_slices().
"""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch
import torch.nn.functional as F

if not (torch.version.hip and torch.cuda.is_available()):
    pytest.skip("ROCm tests", allow_module_level = True)

from exllamav3.ext import exllamav3_ext as ext

device = "cuda:0"
H = 4


def _require_bindings():
    for name in ("hc_mix", "hc_head", "hc_mix_fused", "hc_mix_num_chunks",
                 "hc_apply", "gr_mix", "gr_mix_tiled", "gr_mix_tiled_slices"):
        assert hasattr(ext, name), f"missing native binding: {name}"


# ---------------------------------------------------------------------------------------------
# hc_mix / hc_head
# ---------------------------------------------------------------------------------------------

def _hc_mix_reference(streams, fn, base, scale, rms_eps, hc_eps, iters):
    rows = streams.shape[0]
    flat = streams.flatten(1)
    normed = flat * torch.rsqrt(flat.square().mean(-1, keepdim = True) + rms_eps)
    mix = F.linear(normed, fn.float())
    pre = torch.sigmoid(mix[:, :H] * scale[0] + base[:H]) + hc_eps
    post = 2 * torch.sigmoid(mix[:, H:2 * H] * scale[1] + base[H:2 * H])
    comb = torch.softmax((mix[:, 2 * H:] * scale[2] + base[2 * H:]).view(rows, H, H), dim = -1) + hc_eps
    comb = comb / (comb.sum(dim = -2, keepdim = True) + hc_eps)
    for _ in range(iters - 1):
        comb = comb / (comb.sum(dim = -1, keepdim = True) + hc_eps)
        comb = comb / (comb.sum(dim = -2, keepdim = True) + hc_eps)
    return post, comb, pre


@pytest.mark.parametrize("rows", [1, 7, 32])
@pytest.mark.parametrize("fn_dtype", [torch.float16, torch.float32], ids = ["fn-half", "fn-float"])
@pytest.mark.parametrize("out_dtype", [torch.float16, torch.float32], ids = ["half", "float"])
@torch.inference_mode()
def test_hc_mix_matches_reference(rows, fn_dtype, out_dtype):
    _require_bindings()
    torch.manual_seed(1201 + rows + 10 * int(fn_dtype == torch.float16) + 100 * int(out_dtype == torch.float16))
    d = 2560   # real dimension: several partials chunks, the decode-time shape
    m = 2 * H + H * H
    streams = torch.randn((rows, H, d), device = device) * 0.25
    fn = torch.randn((m, H * d), device = device, dtype = fn_dtype) * 0.05
    base = torch.randn((m,), device = device) * 0.1
    scale = torch.randn((3,), device = device) * 0.2
    rms_eps, hc_eps, iters = 1e-5, 1e-5, 4
    expected_post, expected_comb, pre = _hc_mix_reference(streams, fn, base, scale, rms_eps, hc_eps, iters)
    expected_collapsed = (pre.unsqueeze(-1) * streams).sum(1).to(out_dtype)

    chunks = ext.hc_mix_num_chunks(rows, H * d)
    partials = torch.empty((rows, chunks, m + 1), device = device)
    post = torch.empty((rows, H), device = device)
    comb = torch.empty((rows, H, H), device = device)
    collapsed = torch.empty((rows, d), dtype = out_dtype, device = device)

    ext.hc_mix(streams, fn, base, scale, rms_eps, hc_eps, iters, partials, post, comb, collapsed)

    torch.testing.assert_close(post, expected_post, rtol = 2e-4, atol = 2e-4)
    torch.testing.assert_close(comb, expected_comb, rtol = 2e-4, atol = 2e-4)
    torch.testing.assert_close(collapsed, expected_collapsed, rtol = 2e-3, atol = 2e-3)


@pytest.mark.parametrize("rows", [1, 7])
@pytest.mark.parametrize("out_dtype", [torch.float16, torch.float32], ids = ["half", "float"])
@torch.inference_mode()
def test_hc_head_matches_reference(rows, out_dtype):
    _require_bindings()
    torch.manual_seed(2201 + rows)
    d = 64
    streams = torch.randn((rows, H, d), device = device) * 0.25
    fn = (torch.randn((H, H * d), device = device) * 0.05).half()
    base = torch.randn((H,), device = device) * 0.1
    scale = torch.randn((1,), device = device) * 0.2
    rms_eps, hc_eps = 1e-5, 1e-5
    flat = streams.flatten(1)
    normed = flat * torch.rsqrt(flat.square().mean(-1, keepdim = True) + rms_eps)
    pre = torch.sigmoid(F.linear(normed, fn.float()) * scale + base) + hc_eps
    expected = (pre.unsqueeze(-1) * streams).sum(1).to(out_dtype)

    chunks = ext.hc_mix_num_chunks(rows, H * d)
    partials = torch.empty((rows, chunks, H + 1), device = device)
    collapsed = torch.empty((rows, d), dtype = out_dtype, device = device)

    ext.hc_head(streams, fn, base, scale, rms_eps, hc_eps, partials, collapsed)

    torch.testing.assert_close(collapsed, expected, rtol = 2e-3, atol = 2e-3)


# ---------------------------------------------------------------------------------------------
# hc_apply: x[h] <- post[h] * y + (comb @ x)[h], in place; the weighted-copy form also emits
# xw[h] = x_new[h] * wn[h]
# ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("rows", [1, 7])
@pytest.mark.parametrize("y_dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("with_comb", [False, True])
@torch.inference_mode()
def test_hc_apply_matches_reference(rows, y_dtype, with_comb):
    _require_bindings()
    torch.manual_seed(3201 + rows + int(with_comb))
    d = 68
    x = torch.randn((rows, H, d), device = device)
    original = x.clone()
    y = torch.randn((rows, d), device = device, dtype = y_dtype)
    post = torch.randn((rows, H), device = device)
    comb = torch.randn((rows, H, H), device = device) if with_comb else None
    if comb is None:
        expected = original + post.unsqueeze(-1) * y.float().unsqueeze(1)
    else:
        expected = post.unsqueeze(-1) * y.float().unsqueeze(1) + torch.matmul(
            comb.transpose(-1, -2), original
        )
    ptr = x.data_ptr()

    result = ext.hc_apply(x, y, post, comb, None, None)

    assert result is None
    assert x.data_ptr() == ptr
    torch.testing.assert_close(x, expected, rtol = 2e-5, atol = 2e-5)


@torch.inference_mode()
def test_hc_apply_weighted_copy_form():
    _require_bindings()
    torch.manual_seed(4201)
    rows, d = 3, 128
    x = torch.randn((rows, H, d), device = device)
    original = x.clone()
    y = torch.randn((rows, d), device = device, dtype = torch.half)
    post = torch.randn((rows, H), device = device)
    wn = (torch.randn((H * d,), device = device) * 0.2).half()
    xw = torch.empty((rows, H, d), device = device)

    ext.hc_apply(x, y, post, None, wn, xw)

    expected = original + post.unsqueeze(-1) * y.float().unsqueeze(1)
    torch.testing.assert_close(x, expected, rtol = 2e-5, atol = 2e-5)
    torch.testing.assert_close(xw, expected * wn.float().view(1, H, d), rtol = 2e-5, atol = 2e-5)


# ---------------------------------------------------------------------------------------------
# gr_mix: GatedResidual decode mix
#
#   rmr[h]      = 1 / sqrt(mean_d streams[h, d]^2 + eps)
#   dots[j, h]  = <dot_input[h, :], fn[j, h * D:(h + 1) * D]>   (dot_input = wstreams or
#                 streams * w); dots row M = raw per-stream sums of squares
#   t[i]        = silu(sum_h rmr[h] * dots[i, h] / H)
#   post[h]     = 2 sigmoid(sum_h rmr[h] * dots[LR + h, h] / H)
#   g[h, d]     = sigmoid(sum_i up[h, d, i] * t[i])
#   mixed[d]    = sum_h sigmoid(g[h, d]) * streams[h, d] * rmr[h] * w[h * D + d] / H
# ---------------------------------------------------------------------------------------------

def _gr_mix_reference(streams, fn, upt, w, rms_eps, has_post, wstreams = None):
    rows, _, d = streams.shape
    lr = upt.shape[2]
    m = lr + (H if has_post else 0)
    rmr = torch.rsqrt(streams.square().mean(-1) + rms_eps)                      # (R, H)
    dot_src = wstreams if wstreams is not None else streams * w.float().view(1, H, d)
    dots = torch.einsum("rhd,jhd->rjh", dot_src, fn.view(m, H, d).float())       # (R, M, H)
    t = F.silu((rmr.unsqueeze(1) * dots[:, :lr, :]).sum(-1) / H)                 # (R, LR)
    up = upt.permute(0, 1, 3, 2).reshape(H, d, lr).float()                       # (H, D, LR)
    g = torch.sigmoid(torch.einsum("ri,hdi->rhd", t, up))                        # (R, H, D)
    post = None
    if has_post:
        post_dots = (rmr.unsqueeze(1) * dots[:, lr:, :]).sum(-1) / H             # (R, H)
        post = 2 * torch.sigmoid(post_dots)
    coef = rmr.unsqueeze(-1) * w.float().view(H, d)                               # (R, H, D)
    mixed = (g * streams * coef).sum(1) / H
    return post, mixed


def _gr_mix_args(rows, d, lr, *, has_post = True, mixed_dtype = torch.float16):
    m = lr + (H if has_post else 0)
    streams = torch.randn((rows, H, d), device = device) * 0.25
    fn = (torch.randn((m, H * d), device = device) * 0.05).half()
    upt = (torch.randn((H, d // 4, lr, 4), device = device) * 0.08).half()
    w = (torch.randn((H * d,), device = device) * 0.1).half()
    dots = torch.empty((rows, m + 1, H), device = device)
    post = torch.empty((rows, H), device = device) if has_post else None
    mixed = torch.empty((rows, d), dtype = mixed_dtype, device = device)
    return streams, fn, upt, w, dots, post, mixed


@pytest.mark.parametrize("rows", [1, 7])
@pytest.mark.parametrize("has_post", [False, True], ids = ["final", "site"])
@torch.inference_mode()
def test_gr_mix_matches_reference(rows, has_post):
    _require_bindings()
    torch.manual_seed(5201 + rows + int(has_post))
    d, lr = 2560, 320
    streams, fn, upt, w, dots, post, mixed = _gr_mix_args(rows, d, lr, has_post = has_post)

    ext.gr_mix(streams, None, fn, upt, w, 1e-5, dots, post, mixed)

    expected_post, expected_mixed = _gr_mix_reference(streams, fn, upt, w, 1e-5, has_post)
    if has_post:
        torch.testing.assert_close(post, expected_post, rtol = 2e-4, atol = 2e-4)
    assert mixed.dtype == torch.float16
    torch.testing.assert_close(mixed, expected_mixed.half(), rtol = 2e-3, atol = 2e-3)


# With a pre-weighted stream copy (emitted by the previous site's hc_apply) the dots read the
# copy while the mixed output still scales the raw streams
@torch.inference_mode()
def test_gr_mix_uses_weighted_stream_copy():
    _require_bindings()
    torch.manual_seed(6201)
    rows, d, lr = 2, 512, 128
    streams, fn, upt, w, dots, post, mixed = _gr_mix_args(rows, d, lr, has_post = True)
    wstreams = streams * w.float().view(1, H, d)

    ext.gr_mix(streams, wstreams, fn, upt, w, 1e-5, dots, post, mixed)

    expected_post, expected_mixed = _gr_mix_reference(
        streams, fn, upt, w, 1e-5, True, wstreams = wstreams
    )
    torch.testing.assert_close(post, expected_post, rtol = 2e-4, atol = 2e-4)
    torch.testing.assert_close(mixed, expected_mixed.half(), rtol = 2e-3, atol = 2e-3)


@torch.inference_mode()
def test_gr_mix_float_output():
    _require_bindings()
    torch.manual_seed(6401)
    rows, d, lr = 2, 256, 64
    streams, fn, upt, w, dots, post, mixed = _gr_mix_args(rows, d, lr, has_post = True,
                                                         mixed_dtype = torch.float32)
    ext.gr_mix(streams, None, fn, upt, w, 1e-5, dots, post, mixed)
    expected_post, expected_mixed = _gr_mix_reference(streams, fn, upt, w, 1e-5, True)
    assert mixed.dtype == torch.float32
    torch.testing.assert_close(post, expected_post, rtol = 2e-4, atol = 2e-4)
    torch.testing.assert_close(mixed, expected_mixed, rtol = 2e-3, atol = 2e-3)


# ---------------------------------------------------------------------------------------------
# gr_mix_tiled: the deterministic int8 prefill form. The reference dequantizes the int8
# tables (q = hi * 128 + lo, value = q * scale, det_quant_split / det_gemm.cuh DET_QMAX
# scheme) and runs the same math in fp32; the kernel rounds the normed operand of the output
# gate to half, as the decode-path kernels see it
# ---------------------------------------------------------------------------------------------

def _dequant_i8(w_i8, w_s):
    hi = w_i8[0].float()
    lo = w_i8[1].float()
    return (hi * 128.0 + lo) * w_s.view(-1, 1)


@pytest.mark.parametrize("rows", [64, 100])
@torch.inference_mode()
def test_gr_mix_tiled_matches_reference(rows):
    _require_bindings()
    assert hasattr(ext, "det_quant_weight"), "missing binding: det_quant_weight"
    torch.manual_seed(7201 + rows)
    d, lr = 1024, 320
    use_post = True
    m = lr + (H if use_post else 0)
    mpad = (m + 63) // 64 * 64
    hd = H * d

    streams = torch.randn((rows, H, d), device = device) * 3.0
    w = (torch.randn((hd,), device = device) * 0.1).half()
    proj_h = torch.zeros((mpad, hd), device = device, dtype = torch.half)
    proj_h[:m] = torch.randn((m, hd), device = device) * (1.0 / hd ** 0.5)
    up_h = (torch.randn((hd, lr), device = device) * (1.0 / lr ** 0.5)).half()

    proj_i8 = torch.empty((2, mpad, hd), dtype = torch.int8, device = device)
    proj_sb = torch.empty((mpad,), dtype = torch.float, device = device)
    ext.det_quant_weight(proj_h, proj_i8, proj_sb)
    up_i8 = torch.empty((2, hd, lr), dtype = torch.int8, device = device)
    up_sb = torch.empty((hd,), dtype = torch.float, device = device)
    ext.det_quant_weight(up_h, up_i8, up_sb)

    slices = ext.gr_mix_tiled_slices(rows, d, mpad)
    assert slices > 0 and slices % H == 0
    rpad = (rows + 63) // 64 * 64
    dm_part = torch.empty((slices, rpad, mpad), dtype = torch.float, device = device)
    ss_part = torch.empty((slices, rpad), dtype = torch.float, device = device)
    rmr = torch.empty((rows, H), dtype = torch.float, device = device)
    t_i8 = torch.empty((2, rows, lr), dtype = torch.int8, device = device)
    t_s = torch.empty((rows, lr // 64), dtype = torch.float, device = device)
    post = torch.empty((rows, H), dtype = torch.float, device = device)
    mixed = torch.empty((rows, d), dtype = torch.half, device = device)

    ext.gr_mix_tiled(streams, w, proj_i8, proj_sb, up_i8, up_sb, 1e-6, m,
                     dm_part, ss_part, rmr, t_i8, t_s, post, mixed)

    # Reference over the dequantized tables
    proj_dq = _dequant_i8(proj_i8, proj_sb)[:m]                                  # (M, HD)
    up_dq = _dequant_i8(up_i8, up_sb)                                            # (HD, LR)
    rmr_ref = torch.rsqrt(streams.square().mean(-1, keepdim = True) + 1e-6)      # (R, H, 1)
    normed = streams * rmr_ref * w.float().view(1, H, d)
    dm = (normed.reshape(rows, hd) @ proj_dq.T) / H                              # (R, M); dm carries the /H
    t = F.silu(dm[:, :lr])
    post_ref = 2 * torch.sigmoid(dm[:, lr:])  # single /H total — the kernel applies it once (hc_mix_tiled.cu:305-311)
    g = torch.sigmoid(t @ up_dq.T).view(rows, H, d)
    mixed_ref = (g * normed.half().float()).sum(1) / H

    torch.testing.assert_close(post, post_ref, rtol = 2e-3, atol = 2e-3)
    torch.testing.assert_close(mixed, mixed_ref.half(), rtol = 3e-3, atol = 3e-3)

    # Deterministic run to run: the property the TP replicated-routing design relies on
    mixed_a, post_a = mixed.clone(), post.clone()
    ext.gr_mix_tiled(streams, w, proj_i8, proj_sb, up_i8, up_sb, 1e-6, m,
                     dm_part, ss_part, rmr, t_i8, t_s, post, mixed)
    assert torch.equal(mixed, mixed_a)
    assert torch.equal(post, post_a)


# ---------------------------------------------------------------------------------------------
# Host-side validation
# ---------------------------------------------------------------------------------------------

@torch.inference_mode()
def test_hc_mix_rejects_bad_fn_rows():
    _require_bindings()
    rows, d = 2, 64
    streams = torch.randn((rows, H, d), device = device)
    fn = torch.randn((H + 1, H * d), device = device)
    base = torch.randn((H + 1,), device = device)
    scale = torch.randn((3,), device = device)
    partials = torch.empty((rows, 1, H + 2), device = device)
    post = torch.empty((rows, H), device = device)
    comb = torch.empty((rows, H, H), device = device)
    collapsed = torch.empty((rows, d), dtype = torch.half, device = device)
    with pytest.raises(RuntimeError, match = "fn rows must be H or 2H"):
        ext.hc_mix(streams, fn, base, scale, 1e-5, 1e-5, 4, partials, post, comb, collapsed)


@torch.inference_mode()
def test_hc_mix_rejects_non_contiguous_streams():
    _require_bindings()
    rows, d = 2, 64
    streams = torch.randn((rows, d, H), device = device).permute(0, 2, 1)
    assert not streams.is_contiguous()
    fn = torch.randn((2 * H + H * H, H * d), device = device)
    base = torch.randn((2 * H + H * H,), device = device)
    scale = torch.randn((3,), device = device)
    partials = torch.empty((rows, 1, 2 * H + H * H + 1), device = device)
    post = torch.empty((rows, H), device = device)
    comb = torch.empty((rows, H, H), device = device)
    collapsed = torch.empty((rows, d), dtype = torch.half, device = device)
    with pytest.raises(RuntimeError, match = "contiguous inputs required"):
        ext.hc_mix(streams, fn, base, scale, 1e-5, 1e-5, 4, partials, post, comb, collapsed)


@torch.inference_mode()
def test_hc_apply_rejects_bad_y_shape():
    _require_bindings()
    x = torch.randn((2, H, 64), device = device)
    y = torch.randn((2, 63), device = device)
    post = torch.randn((2, H), device = device)
    with pytest.raises(RuntimeError, match = "y shape"):
        ext.hc_apply(x, y, post, None, None, None)


@torch.inference_mode()
def test_hc_apply_rejects_comb_with_weighted_copy():
    _require_bindings()
    x = torch.randn((2, H, 64), device = device)
    y = torch.randn((2, 64), device = device)
    post = torch.randn((2, H), device = device)
    comb = torch.randn((2, H, H), device = device)
    wn = torch.randn((H * 64,), device = device).half()
    xw = torch.empty_like(x)
    with pytest.raises(RuntimeError, match = "xw is for the comb-less"):
        ext.hc_apply(x, y, post, comb, wn, xw)


@torch.inference_mode()
def test_gr_mix_rejects_bad_fn_rows():
    _require_bindings()
    rows, d, lr = 2, 256, 64
    streams, fn, upt, w, dots, post, mixed = _gr_mix_args(rows, d, lr, has_post = True)
    fn_bad = fn[:-1].contiguous()
    with pytest.raises(RuntimeError, match = "fn rows must be LR"):
        ext.gr_mix(streams, None, fn_bad, upt, w, 1e-5, dots, post, mixed)


@torch.inference_mode()
def test_gr_mix_rejects_bad_upt_layout():
    _require_bindings()
    rows, d, lr = 2, 256, 64
    streams, fn, upt, w, dots, post, mixed = _gr_mix_args(rows, d, lr, has_post = True)
    upt_bad = upt.reshape(H, d // 4, 4, lr).contiguous()
    with pytest.raises(RuntimeError, match = "repacked layout"):
        ext.gr_mix(streams, None, fn, upt_bad, w, 1e-5, dots, post, mixed)


@torch.inference_mode()
def test_gr_mix_rejects_bad_dots_shape():
    _require_bindings()
    rows, d, lr = 2, 256, 64
    streams, fn, upt, w, dots, post, mixed = _gr_mix_args(rows, d, lr, has_post = True)
    dots_bad = dots[:, :-1].contiguous()
    with pytest.raises(RuntimeError, match = "dots shape"):
        ext.gr_mix(streams, None, fn, upt, w, 1e-5, dots_bad, post, mixed)


@torch.inference_mode()
def test_gr_mix_rejects_bad_wstreams_shape():
    _require_bindings()
    rows, d, lr = 2, 256, 64
    streams, fn, upt, w, dots, post, mixed = _gr_mix_args(rows, d, lr, has_post = True)
    wstreams = torch.randn((rows + 1, H, d), device = device)
    with pytest.raises(RuntimeError, match = "wstreams shape"):
        ext.gr_mix(streams, wstreams, fn, upt, w, 1e-5, dots, post, mixed)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))

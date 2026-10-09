"""
ROCm coverage for the CPU expert tier's GPU-side hand-off contracts and the CPU expert
forward, adapted from the fork's test_hip_moe_cpu_offload.py onto dev's interface.

dev's binding surface (cpu/moe_handoff.h, bindings.cpp:191-198) keeps the fork's
exl3_moe_flag_write / exl3_moe_flag_wait names and adds exl3_moe_cpu_set_memops(bool):
the flag ops prefer stream memory operations (hipStreamWriteValue32 / hipStreamWaitValue32
on HIP) and fall back to the flag kernels when the memop call fails or the toggle is off
(moe_handoff.cu:117-170). The fork's finding that the memop path cannot allocate signal
memory on gfx1201 means the kernel fallback is the operative wait on RDNA4 — and only the
kernel wait has a bounded timeout (~30 s, MOE_WAIT_TIMEOUT_NS) that sets the abort flag;
the memop wait relies on the host-side watchdog writing satisfying values into the flags
(moe_handoff.cu:56-60, model/moe_cpu_host.py:1119). Tests therefore run the contracts
under both toggle states, and the full-timeout-budget test forces the kernel path
explicitly via exl3_moe_cpu_set_memops(False).

The full-timeout tests opt in via EXL3_ROCM_SLOW_TESTS (the fork's EXL3_HIP_SLOW_TESTS).

The CPU expert forward (cpu/moe_mul1.cpp) is compared against the unified fused MoE kernel
and a per-expert reconstruct reference on synthetic K3/mul1 experts — no checkpoint needed
(the fork needed a real model for this check; dev's unified path makes it self-contained).
"""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ctypes
import time
from contextlib import nullcontext
from multiprocessing import shared_memory

import numpy as np
import pytest
import torch

if not (torch.version.hip and torch.cuda.is_available()):
    pytest.skip("ROCm tests", allow_module_level = True)

from exllamav3.ext import exllamav3_ext as ext
from exllamav3.util.backend import MOE_FUSED_ROWS

device = "cuda:0"

FLAG_REQUIRED_BINDINGS = (
    "cuda_host_register", "cuda_host_get_device_pointer", "cuda_host_unregister",
    "exl3_moe_flag_write", "exl3_moe_flag_wait", "exl3_moe_cpu_set_memops",
)
for _name in FLAG_REQUIRED_BINDINGS:
    if not hasattr(ext, _name):
        pytest.skip(f"{_name} binding is unavailable", allow_module_level = True)

_slow_ok = os.environ.get("EXL3_ROCM_SLOW_TESTS", "0") != "0"

# Test-local layout inside the registered region: abort word, four 64-byte flag slots,
# one payload block. The flag ops take arbitrary device-visible u32 addresses, so the
# layout only has to stay consistent within this file (dev's real ctrl-block offsets live
# in cpu/moe_handoff.h and are managed by model/moe_cpu_host.py).
ABORT_OFFSET = 0
FLAGS_OFFSET = 4096
FLAG_STRIDE = 64
PAYLOAD_OFFSET = 8192
PAYLOAD_WORDS = 64 * 2560 * 2 // 4
REGION_BYTES = PAYLOAD_OFFSET + PAYLOAD_WORDS * 4 + 4096

HIP_HOST_REGISTER_PORTABLE = 0x01
HIP_HOST_REGISTER_MAPPED = 0x02


class kernel_flag_path:
    """Force the kernel fallback (EXL3_MOE_MEMOPS=0 equivalent) and restore the toggle.
    exl3_moe_cpu_set_memops is process-wide (moe_handoff.cu:117-119)."""

    def __enter__(self):
        ext.exl3_moe_cpu_set_memops(False)

    def __exit__(self, *exc):
        ext.exl3_moe_cpu_set_memops(True)


@pytest.fixture(scope = "module")
def registered_region():
    """One registered shared-memory region with host and device-visible bases."""
    torch.zeros(1, device = device)
    shm = shared_memory.SharedMemory(create = True, size = REGION_BYTES)
    u8 = np.frombuffer(shm.buf, dtype = np.uint8)
    u8[:] = 0
    base = u8.ctypes.data
    rc = ext.cuda_host_register(base, REGION_BYTES,
                                HIP_HOST_REGISTER_PORTABLE | HIP_HOST_REGISTER_MAPPED)
    assert rc is None, f"cuda_host_register failed: {rc}"
    gpu_base = ext.cuda_host_get_device_pointer(base)
    u32 = np.frombuffer(shm.buf, dtype = np.uint32)
    region = {"host": base, "device": gpu_base, "u32": u32}
    try:
        yield region
    finally:
        ext.cuda_host_unregister(base)
        # shm.close() fails while any export of shm.buf exists, so drop every numpy view
        # (and the dict holding them) before closing
        region.clear()
        del u32, u8
        shm.close()
        shm.unlink()


def _flag_device(region, index):
    return region["device"] + FLAGS_OFFSET + index * FLAG_STRIDE


def _abort_device(region):
    return region["device"] + ABORT_OFFSET


def _pattern_u32(seq: int, words: int) -> np.ndarray:
    i = np.arange(words, dtype = np.uint64)
    seq_v = np.full(words, seq & 0xFFFFFFFFFFFFFFFF, dtype = np.uint64)
    v = seq_v * np.uint64(0x9E3779B97F4A7C15) + i * np.uint64(0xBF58476D1CE4E5B9)
    v ^= v >> np.uint64(30)
    v *= np.uint64(0x94D049BB133111EB)
    v ^= v >> np.uint64(31)
    return (v & np.uint64(0xFFFFFFFF)).astype(np.uint32)


def _reduce(words: np.ndarray) -> tuple[int, int]:
    return int(words.astype(np.uint64).sum()) & 0xFFFFFFFF, int(np.bitwise_xor.reduce(words))


def _payload_host_view(region):
    return np.ctypeslib.as_array(
        ctypes.cast(region["host"] + PAYLOAD_OFFSET, ctypes.POINTER(ctypes.c_uint32)),
        shape = (PAYLOAD_WORDS,),
    )


@pytest.mark.parametrize("use_kernel_path", [False, True], ids = ["flag-path", "kernel-path"])
@torch.inference_mode()
def test_flag_write_wait_round_trip_advances_sequence(registered_region, use_kernel_path):
    """Write/wait pairs advance a sequence on the registered region, whichever flag path the
    toggle selects: memops where the runtime accepts them, the launchable flag kernels
    otherwise (a failed memop call latches the kernel fallback process-wide,
    moe_handoff.cu:136/160)."""
    with kernel_flag_path() if use_kernel_path else nullcontext():
        flag = _flag_device(registered_region, 0)
        abort = _abort_device(registered_region)
        for seq in range(5000, 5010):
            ext.exl3_moe_flag_write(flag, seq)
            ext.exl3_moe_flag_wait(flag, seq, abort)
            torch.cuda.synchronize()
        assert int(registered_region["u32"][FLAGS_OFFSET // 4]) == 5009
        assert int(registered_region["u32"][ABORT_OFFSET // 4]) == 0


@pytest.mark.parametrize("use_kernel_path", [False, True], ids = ["flag-path", "kernel-path"])
@torch.inference_mode()
def test_cpu_publish_gpu_wait_acquires_payload(registered_region, use_kernel_path):
    with kernel_flag_path() if use_kernel_path else nullcontext():
        flag_word = (FLAGS_OFFSET + 1 * FLAG_STRIDE) // 4
        payload = _payload_host_view(registered_region)
        for k in range(3):
            seq = 6000 + k
            payload[:] = _pattern_u32(seq, PAYLOAD_WORDS)
            registered_region["u32"][flag_word] = 0
            registered_region["u32"][flag_word] = seq   # release-store on the publishing side
            ext.exl3_moe_flag_wait(_flag_device(registered_region, 1), seq,
                                   _abort_device(registered_region))
            torch.cuda.synchronize()
            assert int(registered_region["u32"][ABORT_OFFSET // 4]) == 0
            assert int(registered_region["u32"][flag_word]) == seq


@torch.inference_mode()
def test_gpu_publish_cpu_consume_after_d2h_copy(registered_region):
    """Stream-ordered D2H copy into the mapped payload, then a flag write: the CPU observes
    the payload only after the flag, mirroring the collect side of the hand-off."""
    payload_dev = torch.empty(PAYLOAD_WORDS, dtype = torch.int32, device = device)
    payload_host = torch.frombuffer(registered_region["u32"].base, dtype = torch.int32,
                                    count = PAYLOAD_WORDS, offset = PAYLOAD_OFFSET)
    flag_word = (FLAGS_OFFSET + 2 * FLAG_STRIDE) // 4
    for k in range(3):
        seq = 7000 + k
        pattern = _pattern_u32(seq, PAYLOAD_WORDS)
        expected = _reduce(pattern)
        registered_region["u32"][flag_word] = 0
        payload_dev.copy_(torch.from_numpy(pattern.astype(np.int32)))
        payload_host.copy_(payload_dev)      # stream-ordered D2H into the mapped payload
        ext.exl3_moe_flag_write(_flag_device(registered_region, 2), seq)
        torch.cuda.synchronize()
        deadline = time.time() + 5.0
        while np.int32(int(registered_region["u32"][flag_word]) - seq) < 0 and time.time() < deadline:
            time.sleep(0.0005)
        assert np.int32(int(registered_region["u32"][flag_word]) - seq) >= 0, \
            "CPU never observed the published flag"
        assert _reduce(_payload_host_view(registered_region).copy()) == expected


@torch.inference_mode()
def test_host_write_unblocks_pending_wait(registered_region):
    """The watchdog pattern: a pending wait (kernel or front-end memop) is unblocked by the
    host writing a satisfying value into the flag — no abort, no timeout."""
    flag_word = (FLAGS_OFFSET + 3 * FLAG_STRIDE) // 4
    registered_region["u32"][flag_word] = 0
    ext.exl3_moe_flag_wait(_flag_device(registered_region, 3), 987654,
                           _abort_device(registered_region))
    time.sleep(0.05)               # the wait is resident on the stream by now
    registered_region["u32"][flag_word] = 987654   # watchdog path: host satisfies the value
    done = torch.cuda.Event()
    done.record()
    deadline = time.time() + 10.0
    while not done.query() and time.time() < deadline:
        time.sleep(0.0005)
    assert done.query(), "host write did not unblock the pending wait"
    assert int(registered_region["u32"][ABORT_OFFSET // 4]) == 0


@pytest.mark.skipif(not _slow_ok, reason = "needs the full ~30 s wait budget (EXL3_ROCM_SLOW_TESTS=1)")
@torch.inference_mode()
def test_unsatisfiable_kernel_wait_is_bounded_and_sets_abort(registered_region):
    """Only the kernel wait carries the timeout/abort contract; the memop wait has none
    (moe_handoff.cu:56-60), so this runs with the toggle forced off. A timed-out wait sets
    the abort flag instead of hanging the stream, and a satisfied wait on the same stream
    proceeds normally afterwards."""
    with kernel_flag_path():
        registered_region["u32"][ABORT_OFFSET // 4] = 0
        start = time.perf_counter()
        ext.exl3_moe_flag_wait(_flag_device(registered_region, 0), 0x7ABCDEF0,
                               _abort_device(registered_region))
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        assert 25.0 <= elapsed <= 45.0, f"wait returned after {elapsed:.1f}s"
        assert int(registered_region["u32"][ABORT_OFFSET // 4]) == 1
        # A satisfied wait on the same stream proceeds after the abort path completed
        registered_region["u32"][ABORT_OFFSET // 4] = 0
        ext.exl3_moe_flag_write(_flag_device(registered_region, 0), 0x7ABCDEF0)
        start = time.perf_counter()
        ext.exl3_moe_flag_wait(_flag_device(registered_region, 0), 0x7ABCDEF0,
                               _abort_device(registered_region))
        torch.cuda.synchronize()
        assert time.perf_counter() - start < 1.0


# ---------------------------------------------------------------------------
# CPU expert forward vs the unified GPU MoE path (synthetic K3/mul1 experts)
# ---------------------------------------------------------------------------

N_EXPERTS = 4
HIDDEN, INTERM, TOP_K = 2560, 640, 10
K = 3


def _projection(k, n, seed):
    """CPU-resident trellis + sign vectors for one projection of every expert (unswizzled;
    0x2492 is a finite K3/mul1 trellis cycle)."""
    words = (k // 16) * (n // 16) * (16 * K)
    tensors, suhs, svhs = [], [], []
    for expert in range(N_EXPERTS):
        stream = torch.full((words,), 0x2492, dtype = torch.int16)
        tensors.append(stream.view(k // 16, n // 16, 16 * K))
        gen = torch.Generator().manual_seed(seed * 100 + expert)
        suhs.append((torch.randint(0, 2, (k,), generator = gen) * 2 - 1).half())
        svhs.append((torch.randint(0, 2, (n,), generator = gen) * 2 - 1).half())
    return tensors, suhs, svhs


def _ptr_table(tensors, dev):
    return torch.tensor([t.data_ptr() for t in tensors], dtype = torch.long, device = dev)


@pytest.fixture(scope = "module")
def k3_expert_layer():
    """Synthetic K3/mul1 experts registered for the CPU worker, plus device copies and
    pointer tables for the GPU-side unified MoE call."""
    for name in ("exl3_moe_cpu_make_layer", "exl3_moe_cpu_forward", "exl3_moe"):
        assert hasattr(ext, name), f"missing binding: {name}"
    if not ext.exl3_moe_cpu_has_avx2():
        pytest.skip("CPU expert kernel needs AVX2")
    projections = [
        _projection(HIDDEN, INTERM, 11),   # gate: (hidden -> interm)
        _projection(HIDDEN, INTERM, 23),   # up
        _projection(INTERM, HIDDEN, 37),   # down
    ]
    groups = []
    for trellis, suhs, svhs in projections:
        groups += [trellis, suhs, svhs]
    handle = ext.exl3_moe_cpu_make_layer(*groups, [], [], [], 0, 0.0, 0)

    # Pointer tables alias the device-resident copies, so those must outlive every kernel
    # call that uses the tables; building them here keeps them alive for the fixture's scope.
    # alive order: (gate, up, down) x (trellis, suh, svh) — the reconstruct reference below
    # indexes it directly
    alive, tables = [], []
    for idx in range(3):
        for part in (0, 1, 2):
            moved = [t.to(device) for t in projections[idx][part]]
            alive.append(moved)
            tables.append(_ptr_table(moved, device))
    torch.cuda.synchronize()
    return handle, alive, tables


def _gpu_unified_forward(y, selected, weights, tables, intermediate):
    """The unified fused MoE path with deterministic slot accumulation, as the module
    dispatches it (see test_rocm_grouped_moe.py for the argument layout)."""
    rows = y.shape[0]
    flat_expert = selected.reshape(-1)
    flat_weight = weights.reshape(-1)
    valid = (flat_expert >= 0) & (flat_expert < N_EXPERTS)
    flat_expert_local = torch.where(valid, flat_expert, torch.full_like(flat_expert, N_EXPERTS))
    flat_token = torch.arange(rows, device = device).repeat_interleave(TOP_K)
    order = flat_expert_local.argsort(stable = True)
    token_sorted = flat_token[order]
    weight_sorted = flat_weight[order]
    expert_count = torch.bincount(flat_expert_local, minlength = N_EXPERTS + 1)
    inv_order = torch.empty_like(order).scatter_(
        0, order, torch.arange(flat_expert_local.shape[0], device = device))
    counts = expert_count[:N_EXPERTS]
    slot_base = torch.cumsum(counts, 0) - counts
    scratch = torch.empty((rows * TOP_K, HIDDEN), dtype = torch.float, device = device)
    output = torch.zeros((rows, HIDDEN), dtype = torch.float, device = device)
    concurrency = ext.exl3_moe_max_concurrency(0)
    # Temp capacity must be (C, fused_rows, dim) as the module sizes it (the RDNA kernel
    # indexes temps by row capacity, not by the per-call counts), and num_active must be the
    # counted fused-expert total as the module's tier split passes (see test_rocm_grouped_moe)
    R = MOE_FUSED_ROWS
    temp_state_g = torch.empty((concurrency, R, HIDDEN), dtype = torch.half, device = device)
    temp_state_u = torch.empty((concurrency, R, HIDDEN), dtype = torch.half, device = device)
    temp_intermediate_g = torch.empty((concurrency, R, intermediate), dtype = torch.half, device = device)
    temp_intermediate_u = torch.empty((concurrency, R, intermediate), dtype = torch.half, device = device)
    ext.exl3_moe(
        y, output, expert_count, token_sorted, weight_sorted,
        temp_state_g, temp_state_u, temp_intermediate_g, temp_intermediate_u,
        0, float(K), float(K), float(K),
        *tables,
        False, True, False, True, False, True,
        0.0, int((counts > 0).sum()), scratch, slot_base, 1, R, 16,
    )
    ext.exl3_moe_gather(
        output, scratch, flat_expert_local, inv_order,
        expert_count[:N_EXPERTS].cumsum(0) - counts, slot_base, (counts > 0).long(), weight_sorted)
    return output


@torch.inference_mode()
def test_cpu_expert_forward_matches_unified_kernel(k3_expert_layer):
    """The CPU mul1 kernel quantizes activations to int8, so CPU-vs-GPU tolerance is the
    int8-activation approximation envelope; the GPU path must match the fp16 reconstruct
    reference tightly, and both paths must be deterministic."""
    handle, alive, tables = k3_expert_layer
    generator = torch.Generator().manual_seed(20260907)
    for rows in (1, 4, 16):
        # 0.01 keeps the up-projection intermediates inside the unified kernel's fp16-partial
        # dynamic range (fp16 partial sums through global memory, exl3_moe_inner_rdna.cuh);
        # at y*0.5 the structured trellis overflows to inf → NaN on the GPU path, while the
        # CPU kernel's int8 activations + per-row scales survive any range
        y = torch.randn(rows, HIDDEN, generator = generator, dtype = torch.half) * 0.01
        selected = torch.randint(0, N_EXPERTS, (rows, TOP_K), generator = generator,
                                 dtype = torch.long)
        if rows >= 4:
            selected[1, 0] = -1            # GPU-resident-masked sentinel (split semantics)
            selected[2, 3] = selected[0, 0]   # duplicate id
        weights = torch.rand(rows, TOP_K, generator = generator, dtype = torch.half) * 0.4 + 0.05

        out_cpu = torch.empty(rows, HIDDEN, dtype = torch.float)
        ext.exl3_moe_cpu_forward(handle, y.clone(), selected.clone(), weights, out_cpu, 8)
        out_cpu2 = torch.empty(rows, HIDDEN, dtype = torch.float)
        ext.exl3_moe_cpu_forward(handle, y.clone(), selected.clone(), weights, out_cpu2, 8)
        assert torch.equal(out_cpu, out_cpu2), "CPU forward is not deterministic"

        output = _gpu_unified_forward(y.to(device), selected.to(device),
                                      weights.to(device), tables, INTERM)
        torch.cuda.synchronize()

        rel_rms = ((out_cpu - output.cpu()).pow(2).mean().sqrt()
                   / (output.cpu().pow(2).mean().sqrt() + 1e-9)).item()
        # Measured CPU-vs-GPU envelope at this data: ~2.7-3.8% rel RMS — the CPU int8-activation
        # quantization dominates (per-row scales on structured-trellis magnitudes); the fork's
        # 0.02 was calibrated for its own fp32-intermediate fused kernel
        assert rel_rms <= 0.04, f"rows={rows}: CPU/GPU rel RMS {rel_rms:.4f} exceeds the int8 envelope"

        reference = torch.zeros(rows, HIDDEN, dtype = torch.float, device = device)
        expert_ids = sorted({int(v) for row in selected.tolist() for v in row if 0 <= v < N_EXPERTS})
        for expert in expert_ids:
            slots = (selected == expert).nonzero()[:, 0].to(device)
            xg = y.to(device).index_select(0, slots)
            wseg = weights.to(device)[selected == expert].float().unsqueeze(1)

            def projection(x_in, trellis, suh, svh):
                k, n = trellis.shape[0] * 16, trellis.shape[1] * 16
                xh = torch.empty_like(x_in)
                ext.had_r_128(x_in, xh, suh, None, 1.0)
                w = torch.empty(k * n, dtype = torch.half, device = device).view(k, n)
                ext.reconstruct(w, trellis, K, False, True)
                o = torch.empty(x_in.shape[0], n, dtype = torch.half, device = device)
                ext.hgemm(xh, w, o)
                ext.had_r_128(o, o, None, svh, 1.0)
                return o

            gate = projection(xg, alive[0][expert], alive[1][expert], alive[2][expert])
            up = projection(xg, alive[3][expert], alive[4][expert], alive[5][expert])
            act = (torch.nn.functional.silu(gate.float()) * up.float()).half()
            ah = torch.empty_like(act)
            ext.had_r_128(act, ah, alive[7][expert], None, 1.0)
            down_trellis = alive[6][expert]
            kd, nd = down_trellis.shape[0] * 16, down_trellis.shape[1] * 16
            wd = torch.empty(kd * nd, dtype = torch.half, device = device).view(kd, nd)
            ext.reconstruct(wd, down_trellis, K, False, True)
            od = torch.empty(act.shape[0], nd, dtype = torch.half, device = device)
            ext.hgemm(ah, wd, od)
            ext.had_r_128(od, od, None, alive[8][expert], 1.0)
            reference.index_add_(0, slots, od.float()[:, :HIDDEN] * wseg)
        # Relative contract: outputs at this data scale reach ~4.5e3, and the unified kernel
        # matches the reconstruct reference to ~5e-4 relative — an absolute 0.02 is a ~5e-6
        # relative check, below any fp16 pipeline's noise floor (same artifact as F5)
        ref_scale = reference.abs().max()
        assert (output - reference).abs().max().item() <= max(0.02, 3e-3 * ref_scale), \
            "unified kernel diverged from the reconstruct reference"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))

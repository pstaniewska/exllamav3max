"""
ROCm coverage for the compiled rewind/cache bindings (gdn.cu / bindings.cpp).

In dev these bindings are bound unconditionally on both backends, so this test only checks
presence and the state-rewind copy semantics (no portable/ROCm-only framing, no fallback
simulation).
"""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest
import torch

if not (torch.version.hip and torch.cuda.is_available()):
    pytest.skip("ROCm tests", allow_module_level = True)

from exllamav3.ext import exllamav3_ext as ext

device = "cuda:0"


def test_rewind_and_cache_bindings_exist():
    for name in (
        "ConvRewindJob", "StateRewindJob", "batched_conv_rewind",
        "batched_state_rewind", "dspark_write_rows",
    ):
        assert hasattr(ext, name), f"missing binding: {name}"


@torch.inference_mode()
def test_batched_state_rewind_copies_state():
    # StateRewindJob(src, dst, num_elements) is a flat fp32 copy with num_elements % 4 == 0
    state = torch.arange(4 * 8, device = device, dtype = torch.float32).reshape(4, 8)
    expected = state[3].clone()
    job = ext.StateRewindJob(state[3].data_ptr(), state[0].data_ptr(), state[0].numel())
    ext.batched_state_rewind([job], 0)
    torch.cuda.synchronize()

    torch.testing.assert_close(state[0], expected)


@torch.inference_mode()
def test_batched_state_rewind_copies_multiple_jobs():
    src = torch.randn(2, 16, device = device, dtype = torch.float32)
    dst = torch.zeros_like(src)
    jobs = [
        ext.StateRewindJob(src[i].data_ptr(), dst[i].data_ptr(), src[i].numel())
        for i in range(2)
    ]
    ext.batched_state_rewind(jobs, torch.cuda.current_device())
    torch.cuda.synchronize()

    torch.testing.assert_close(dst, src)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))

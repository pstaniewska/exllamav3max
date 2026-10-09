"""
ROCm coverage for the fused shared-expert sigmoid gate (activation.cu).

Dev exposes both entry points natively on both backends:
  - add_sigmoid_gate(x, y, z):        z += x * sigmoid(y),        all fp32
  - add_sigmoid_gate_proj(x, y, z, w): z += x * sigmoid(y @ w),   x/z fp32, y/w fp16
Dev checks dtypes and the (dim, 1) gate shape but does not validate contiguity, device
placement, or z's shape — rejection tests here cover only the contracts dev actually
enforces.
"""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest
import torch

if not (torch.version.hip and torch.cuda.is_available()):
    pytest.skip("ROCm tests", allow_module_level = True)

from exllamav3.ext import exllamav3_ext as ext

device = "cuda:0"


def _case(shape, *, offset = False):
    def make(shape, dtype):
        if not offset:
            return torch.randn(shape, dtype = dtype, device = device)
        numel = torch.Size(shape).numel()
        storage = torch.empty(numel + 1, dtype = dtype, device = device)
        view = storage[1:].view(shape)
        view.normal_()
        assert view.is_contiguous()
        return view

    x = make(shape, torch.float32)
    y = make(shape, torch.float16)
    z = make(shape, torch.float32)
    w = make((shape[-1], 1), torch.float16)
    return x, y, z, w


@pytest.mark.parametrize("shape", [(1, 17), (3, 2, 2560)])
@pytest.mark.parametrize("offset", [False, True], ids = ["base", "offset"])
@torch.inference_mode()
def test_add_sigmoid_gate_proj_matches_reference_and_updates_z_in_place(shape, offset):
    x, y, z, w = _case(shape, offset = offset)
    identity = id(z)
    ptr = z.data_ptr()
    expected = z.clone() + x * torch.sigmoid(y.float().matmul(w.float()))

    result = ext.add_sigmoid_gate_proj(x, y, z, w)

    assert result is None
    assert id(z) == identity and z.data_ptr() == ptr
    torch.testing.assert_close(z, expected, rtol = 5e-5, atol = 3e-5)


@torch.inference_mode()
def test_add_sigmoid_gate_proj_uses_current_stream():
    x = torch.ones((2, 2560), dtype = torch.float32, device = device)
    z = torch.zeros_like(x)
    y = torch.zeros((2, 2560), dtype = torch.float16, device = device)
    w = torch.zeros((2560, 1), dtype = torch.float16, device = device)
    source_y = torch.full_like(y, 0.25)
    source_w = torch.full_like(w, 1 / 2560)
    stream = torch.cuda.Stream(device = device)

    with torch.cuda.stream(stream):
        y.copy_(source_y)
        w.copy_(source_w)
        ext.add_sigmoid_gate_proj(x, y, z, w)
    stream.synchronize()

    expected_gate = torch.sigmoid(torch.tensor(0.25, device = device))
    torch.testing.assert_close(z, torch.full_like(z, expected_gate), rtol = 2e-5, atol = 2e-5)


@torch.inference_mode()
def test_add_sigmoid_gate_matches_reference():
    # The 3-arg form applies a per-group broadcast gate (y.size(-1) == 1), not elementwise logits
    x = torch.randn(4, 256, dtype = torch.float32, device = device)
    y = torch.randn(4, 1, dtype = torch.float32, device = device)
    z = torch.randn(4, 256, dtype = torch.float32, device = device)
    expected = z + x * torch.sigmoid(y)

    assert ext.add_sigmoid_gate(x, y, z) is None
    torch.testing.assert_close(z, expected, rtol = 1e-5, atol = 1e-5)


@pytest.mark.parametrize(
    "bad_arg,dtype",
    [("x", torch.float16), ("z", torch.float16), ("y", torch.float32), ("w", torch.float32)],
)
def test_add_sigmoid_gate_proj_rejects_invalid_dtypes(bad_arg, dtype):
    tensors = list(_case((2, 32)))
    tensors["xyzw".index(bad_arg)] = tensors["xyzw".index(bad_arg)].to(dtype)
    with pytest.raises(RuntimeError, match = "datatype|dtype"):
        ext.add_sigmoid_gate_proj(*tensors)


@pytest.mark.parametrize(
    "shapes",
    [
        ((2, 32), (2, 32), (2, 32), (32,)),      # w must be 2-D
        ((2, 32), (2, 32), (2, 32), (32, 2)),    # gate must have size(-1) == 1
        ((2, 32), (2, 16), (2, 32), (16, 1)),    # y.size(-1) must match x.size(-1)
        ((2, 32), (2, 32), (2, 32), (16, 1)),    # w.size(-2) must match x.size(-1)
    ],
)
def test_add_sigmoid_gate_proj_rejects_invalid_shapes(shapes):
    dtypes = (torch.float32, torch.float16, torch.float32, torch.float16)
    tensors = [torch.empty(shape, dtype = dtype, device = device) for shape, dtype in zip(shapes, dtypes)]
    with pytest.raises(RuntimeError, match = "gate must|incompatible shapes"):
        ext.add_sigmoid_gate_proj(*tensors)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))

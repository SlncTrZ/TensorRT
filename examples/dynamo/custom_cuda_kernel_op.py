"""
.. _custom_cuda_kernel_op:

Hand-Written Custom Plugin via ``torch_tensorrt.kernels.custom_cuda_kernel_op``
====================================================================================

This example demonstrates a shape-changing (non-pointwise) CUDA kernel that
duplicates each input element into two output elements:
    y[2*i] = x[i], y[2*i + 1] = x[i]

Because the output size (``2 * n``) does not match any of the geometries built
into the declarative :func:`torch_tensorrt.kernels.cuda_kernel_op`
DSL, we drop down one layer to
:func:`torch_tensorrt.kernels.custom_cuda_kernel_op` and hand-write
``eager_fn`` / ``aot_fn`` directly.

For kernels that *do* fit the DSL (pointwise, ND-grid, reduction) start from
the simpler declarative path shown in
``cuda_kernel_op.py``.
"""

import tensorrt.plugin as trtp
import torch
from cuda.core import Device as _Device
from cuda.core import LaunchConfig as _LaunchConfig
from cuda.core import Program as _Program
from cuda.core import ProgramOptions as _ProgramOptions
from cuda.core import launch as _cuda_launch

import torch_tensorrt
import torch_tensorrt.kernels as ttk

CU_REPEAT2 = """
extern "C" __global__ void repeat2_kernel(
        const float* __restrict__ x, const int n, float* __restrict__ y) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const float v = x[i];
        y[2 * i] = v;
        y[2 * i + 1] = v;
    }
}
"""

_device = _Device()
_device.set_current()
_opts = _ProgramOptions(
    std="c++17", arch=f"sm_{_device.arch}", include_path=["/usr/local/cuda/include"]
)
_program = _Program(CU_REPEAT2, code_type="c++", options=_opts)
_module = _program.compile("ptx", name_expressions=("repeat2_kernel",))
_kernel = _module.get_kernel("repeat2_kernel")


class _PTStream:
    def __cuda_stream__(self):
        return (0, torch.cuda.current_stream().cuda_stream)


def _eager_repeat2(x: torch.Tensor) -> torch.Tensor:
    if x.dtype != torch.float32:
        raise ValueError("This example expects float32 input")
    flat = x.contiguous().view(-1)
    n = int(flat.numel())
    y = torch.empty((n * 2,), device=x.device, dtype=x.dtype)
    block = 256
    grid = max(1, (n + block - 1) // block)
    stream = _device.create_stream(_PTStream())
    _cuda_launch(
        stream,
        _LaunchConfig(grid=(grid,), block=(block,)),
        _kernel,
        flat.data_ptr(),
        n,
        y.data_ptr(),
    )
    return y


def _aot_repeat2(inputs, outputs, tactic):
    n = inputs[0].shape_expr.numel()
    params = trtp.KernelLaunchParams()
    params.grid_x = trtp.cdiv(n, 256)
    params.block_x = 256
    params.shared_mem = 0
    extra = trtp.SymIntExprs(1)
    extra[0] = trtp.SymInt32(n)
    return params, extra


def _repeat2_meta(x: torch.Tensor) -> torch.Tensor:
    return torch.empty((x.numel() * 2,), device=x.device, dtype=x.dtype)


ttk.custom_cuda_kernel_op(
    op_name="kern_ex::repeat2",
    kernel_source=CU_REPEAT2,
    kernel_name="repeat2_kernel",
    meta_fn=_repeat2_meta,
    eager_fn=_eager_repeat2,
    aot_fn=_aot_repeat2,
    supports_dynamic_shapes=True,
)


class Repeat2Model(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.ops.kern_ex.repeat2(x)


if __name__ == "__main__":
    x = torch.randn(1024, device="cuda", dtype=torch.float32)
    ref = torch.repeat_interleave(x, 2, dim=0)

    model = Repeat2Model().cuda().eval()
    eager_out = model(x)
    print(
        "Eager result matches repeat_interleave:",
        torch.allclose(eager_out, ref, atol=1e-4),
    )

    print("Compiling with Torch-TensorRT...")
    with torch_tensorrt.logging.debug():
        trt_model = torch_tensorrt.compile(
            model,
            inputs=[x],
            enabled_precisions={torch.float32},
            min_block_size=1,
        )

    with torch.no_grad():
        for _ in range(5):
            out = trt_model(x)
            assert torch.allclose(out, ref, atol=1e-2, rtol=1e-2), "Mismatch!"

    print("TRT inference successful - results match repeat_interleave")

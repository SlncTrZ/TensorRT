"""
torch_tensorrt.kernels  (experimental)
=======================================
High-level decorators for registering custom CUDA C++ kernels — compiled at
runtime with NVRTC via **cuda-python** — as TensorRT Quick Deployable Plugins
(QDP). Tensor-only declarative kernels use AOT plugin launches when available;
kernels with ``ScalarInput`` use TensorRT's QDP JIT path so runtime scalar
attributes can be forwarded by value.

The module offers two registration paths.  Pick ``cuda_kernel_op`` first; drop
down to ``custom_cuda_kernel_op`` only when your kernel doesn't fit the Elementwise /
Reduction conventions.

``cuda_kernel_op`` *(recommended starting point)*
    Fully declarative.  Describe inputs, outputs, extras, and launch geometry
    via :class:`KernelSpec` dataclasses; the meta / eager / converter functions and
    the PyTorch schema are all derived for you.  Covers pointwise, ND-grid,
    and reduction kernels out of the box.

``custom_cuda_kernel_op``
    One-shot function when you want to hand-write ``meta_fn`` / ``eager_fn`` /
    ``aot_fn`` directly.  Use this for shape-changing kernels, multi-output
    kernels, or anything outside the declarative DSL.  All three callables are
    passed as arguments; nothing is decorated.

``custom_plugin`` / ``cuda_python``
    Lower-level building blocks used internally by ``custom_cuda_kernel_op``.  Use
    ``cuda_python`` to construct a reusable :class:`CudaPythonSpec`, then pass
    it to ``custom_plugin`` to register.

``ptx_op``
    Register a kernel from pre-compiled PTX bytes.  Skips NVRTC entirely;
    you supply ``meta_fn`` / ``eager_fn`` / ``aot_fn`` directly.  Useful when
    the PTX comes from an external compiler (Triton, a cached NVRTC output,
    etc.).

Minimal example — ``cuda_kernel_op`` (derives meta/eager/aot/schema)::

    import torch, torch_tensorrt
    import torch_tensorrt.kernels as ttk

    cu_code = \"\"\"
    extern "C" __global__ void my_relu(const float* x, int n, float* y) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n) y[i] = x[i] > 0.f ? x[i] : 0.f;
    }
    \"\"\"

    ttk.cuda_kernel_op(
        "myns::relu",
        ttk.KernelSpec(
            kernel_source=cu_code,
            kernel_name="my_relu",
            inputs=[ttk.InputDecl("x")],
            outputs=[ttk.OutputDecl("y", shape=ttk.SameAs(0))],
            extras=[ttk.Numel("x")],
            geometry=ttk.Elementwise(block=(256,), layout="flat"),
        ),
        supports_dynamic_shapes=True,
    )

    class M(torch.nn.Module):
        def forward(self, x): return torch.ops.myns.relu(x)

    trt = torch_tensorrt.compile(
        M().cuda().eval(),
        inputs=[torch.randn(1024, device="cuda")],
        enabled_precisions={torch.float32},
        min_block_size=1,
    )

For kernels outside the Elementwise / Reduction shape conventions, drop down
to :func:`custom_cuda_kernel_op` and supply ``aot_fn`` / ``eager_fn`` directly.
"""

from torch_tensorrt.kernels._custom_plugin import (
    cuda_python,
    custom_cuda_kernel_op,
    custom_plugin,
    ptx_op,
)
from torch_tensorrt.kernels._kernel_plugin import cuda_kernel_op
from torch_tensorrt.kernels._kernel_spec import (
    Custom,
    DimSize,
    Elementwise,
    InputDecl,
    KernelSpec,
    Numel,
    OutputDecl,
    ReduceDims,
    Reduction,
    SameAs,
    ScalarInput,
)
from torch_tensorrt.kernels._specs import CudaPythonSpec

__all__ = [
    "CudaPythonSpec",
    "Custom",
    "DimSize",
    "Elementwise",
    "InputDecl",
    "KernelSpec",
    "Numel",
    "OutputDecl",
    "ReduceDims",
    "Reduction",
    "SameAs",
    "ScalarInput",
    "custom_cuda_kernel_op",
    "cuda_python",
    "custom_plugin",
    "cuda_kernel_op",
    "ptx_op",
]

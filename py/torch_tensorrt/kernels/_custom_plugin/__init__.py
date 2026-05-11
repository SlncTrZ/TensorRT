from __future__ import annotations

import logging
from typing import Any, Callable, List, Optional

from torch_tensorrt._features import ENABLED_FEATURES
from torch_tensorrt.dynamo.conversion._ConverterRegistry import ConverterPriority
from torch_tensorrt.kernels._specs import CudaPythonSpec, _default_cuda_include_paths

_LOGGER = logging.getLogger(__name__)


def cuda_python(
    kernel_source: str,
    kernel_name: str,
    aot_fn: Optional[Callable[..., Any]] = None,
    eager_fn: Optional[Callable[..., Any]] = None,
    include_paths: Optional[List[str]] = None,
    compile_std: str = "c++17",
    arch_override: Optional[str] = None,
) -> CudaPythonSpec:
    """Create a :class:`CudaPythonSpec` for a CUDA C++ kernel compiled with NVRTC.

    Args:
        kernel_source: Raw CUDA C++ source string containing the ``__global__`` kernel.
        kernel_name: Name of the ``extern "C" __global__`` function to invoke.
        aot_fn: Callable ``(inputs, outputs, tactic) -> (KernelLaunchParams, SymExprs | None)``
            that returns launch parameters for the TensorRT AOT implementation.
            May be set later by assigning to ``spec.aot_fn``.
        eager_fn: Optional callable used as the CUDA device implementation of the
            PyTorch custom op.  Required when ``register_torch_op=True`` in
            :func:`custom_plugin`.  Signature must match the op schema.
        include_paths: Extra ``#include`` search paths passed to NVRTC.
            Defaults to ``$CUDA_HOME/include`` (or ``$CUDA_PATH/include``),
            falling back to ``["/usr/local/cuda/include"]``.
        compile_std: C++ standard flag forwarded to NVRTC (default ``"c++17"``).
        arch_override: Override the GPU architecture string (e.g. ``"sm_86"``).
            When ``None`` the current device's arch is used.

    Returns:
        A :class:`CudaPythonSpec` instance ready for use with :func:`custom_plugin`.
    """
    if not ENABLED_FEATURES.qdp_plugin:
        raise RuntimeError(
            "TensorRT QDP plugins are not available. "
            "Requires TensorRT >= 10.7.0 (and not 10.14.x)."
        )
    return CudaPythonSpec(
        kernel_source=kernel_source,
        kernel_name=kernel_name,
        aot_fn=aot_fn,
        eager_fn=eager_fn,
        include_paths=(
            include_paths
            if include_paths is not None
            else _default_cuda_include_paths()
        ),
        compile_std=compile_std,
        arch_override=arch_override,
    )


def custom_plugin(
    op_name: str,
    spec: CudaPythonSpec,
    supports_dynamic_shapes: bool = False,
    requires_output_allocator: bool = False,
    priority: ConverterPriority = ConverterPriority.STANDARD,
    capability_validator: Optional[Callable[..., Any]] = None,
    schema: Optional[str] = None,
) -> Callable[..., Any]:
    """Decorator that registers a CUDA kernel as a TensorRT QDP plugin.

    The decorated function acts as the **meta / fake implementation** (shape and
    dtype inference for TorchDynamo tracing).  It must mirror the op's signature
    with proper type annotations and return ``torch.empty_*`` tensors of the
    correct shape.

    ``spec.eager_fn`` is registered as the CUDA device implementation of the
    generated PyTorch custom op, so it must accept the same positional arguments
    and return the same outputs as the meta function.

    ``spec.aot_fn`` is registered as the TensorRT AOT implementation and must
    have the signature::

        def aot_fn(inputs: list[trtp.TensorDesc],
                   outputs: tuple[trtp.TensorDesc],
                   tactic: int) -> tuple[trtp.KernelLaunchParams, trtp.SymExprs | None]:

    Example::

        cu_code = \"\"\"
        extern "C" __global__ void my_sigmoid(const float* x, int n, float* y) {
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < n) y[i] = 1.0f / (1.0f + __expf(-x[i]));
        }
        \"\"\"

        def _eager(x: torch.Tensor) -> torch.Tensor:
            ...  # cuda-python launch

        def _aot(inputs, outputs, tactic):
            import tensorrt.plugin as trtp
            N = inputs[0].shape_expr.numel()
            p = trtp.KernelLaunchParams()
            p.grid_x, p.block_x, p.shared_mem = trtp.cdiv(N, 256), 256, 0
            extra = trtp.SymIntExprs(1)
            extra[0] = trtp.SymInt32(N)
            return p, extra

        spec = ttk.cuda_python(cu_code, "my_sigmoid", aot_fn=_aot, eager_fn=_eager)

        @ttk.custom_plugin("myns::sigmoid", spec, supports_dynamic_shapes=True)
        def _(x: torch.Tensor) -> torch.Tensor:
            return torch.empty_like(x)

    Args:
        op_name: Plugin id in ``"namespace::name"`` format.
        spec: :class:`CudaPythonSpec` with ``aot_fn`` and ``eager_fn`` populated.
        supports_dynamic_shapes: Pass ``True`` if the kernel handles dynamic shapes.
        requires_output_allocator: Set ``True`` for data-dependent output shapes.
        priority: Converter priority in the registry.
        capability_validator: Optional ``(Node, CompilationSettings) -> bool`` guard.
        schema: Optional explicit PyTorch op schema suffix in Torch schema syntax
            (for example, ``"(Tensor x) -> Tensor"``). When omitted, the schema
            is inferred from the decorated meta function's type hints.
    """
    from torch_tensorrt.kernels._custom_plugin._descriptor import (
        register_cuda_python_plugin,
    )

    def decorator(meta_fn: Callable[..., Any]) -> Callable[..., Any]:
        register_cuda_python_plugin(
            op_name=op_name,
            spec=spec,
            meta_fn=meta_fn,
            supports_dynamic_shapes=supports_dynamic_shapes,
            requires_output_allocator=requires_output_allocator,
            priority=priority,
            capability_validator=capability_validator,
            register_torch_op=True,
            schema=schema,
        )
        return meta_fn

    return decorator


def custom_cuda_kernel_op(
    op_name: str,
    kernel_source: str,
    kernel_name: str,
    meta_fn: Callable[..., Any],
    eager_fn: Callable[..., Any],
    aot_fn: Callable[..., Any],
    *,
    include_paths: Optional[List[str]] = None,
    compile_std: str = "c++17",
    arch_override: Optional[str] = None,
    supports_dynamic_shapes: bool = False,
    requires_output_allocator: bool = False,
    priority: ConverterPriority = ConverterPriority.STANDARD,
    capability_validator: Optional[Callable[..., Any]] = None,
    schema: Optional[str] = None,
) -> None:
    """Register a hand-written CUDA kernel as a TensorRT QDP plugin in one call.

    All three callables are passed as arguments:

    * ``meta_fn`` — fake/meta impl: shape + dtype inference for tracing.
    * ``eager_fn`` — CUDA device impl invoked when the op runs in PyTorch eager.
    * ``aot_fn`` — TensorRT AOT impl returning symbolic launch params + extras.

    Use this when your kernel doesn't fit the declarative
    :func:`cuda_kernel_op` DSL — e.g. shape-changing kernels, multi-output
    kernels, or kernels needing custom launch logic.

    Args:
        op_name: Plugin id in ``"namespace::name"`` format.
        kernel_source: Raw CUDA C++ source containing the ``__global__`` kernel.
        kernel_name: Name of the ``extern "C" __global__`` function to invoke.
        meta_fn: Fake/meta implementation. Must mirror the op signature with
            type hints and return ``torch.empty_*`` tensors of the right shape.
        eager_fn: CUDA implementation. Same positional signature as ``meta_fn``.
        aot_fn: ``(inputs, outputs, tactic) -> (KernelLaunchParams, SymExprs | None)``.
        schema: Optional explicit Torch schema (e.g. ``"(Tensor x) -> Tensor"``);
            inferred from ``meta_fn`` type hints when omitted.

    See :func:`cuda_python` / :func:`custom_plugin` for the lower-level building
    blocks if you need finer control over spec construction.
    """
    from torch_tensorrt.kernels._custom_plugin._descriptor import (
        register_cuda_python_plugin,
    )

    spec = cuda_python(
        kernel_source=kernel_source,
        kernel_name=kernel_name,
        aot_fn=aot_fn,
        eager_fn=eager_fn,
        include_paths=include_paths,
        compile_std=compile_std,
        arch_override=arch_override,
    )
    register_cuda_python_plugin(
        op_name=op_name,
        spec=spec,
        meta_fn=meta_fn,
        supports_dynamic_shapes=supports_dynamic_shapes,
        requires_output_allocator=requires_output_allocator,
        priority=priority,
        capability_validator=capability_validator,
        register_torch_op=True,
        schema=schema,
    )


def ptx_op(
    op_name: str,
    ptx: bytes,
    kernel_name: str,
    meta_fn: Callable[..., Any],
    eager_fn: Callable[..., Any],
    aot_fn: Callable[..., Any],
    *,
    supports_dynamic_shapes: bool = False,
    requires_output_allocator: bool = False,
    priority: ConverterPriority = ConverterPriority.STANDARD,
    capability_validator: Optional[Callable[..., Any]] = None,
    schema: Optional[str] = None,
) -> None:
    """Register a pre-compiled PTX kernel as a TensorRT QDP plugin."""
    if not ENABLED_FEATURES.qdp_plugin:
        raise RuntimeError(
            "TensorRT QDP plugins are not available. "
            "Requires TensorRT >= 10.7.0 (and not 10.14.x)."
        )

    from torch_tensorrt.kernels._custom_plugin._descriptor import (
        register_cuda_python_plugin,
    )

    spec = CudaPythonSpec(
        kernel_source="",
        kernel_name=kernel_name,
        aot_fn=aot_fn,
        eager_fn=eager_fn,
    )
    register_cuda_python_plugin(
        op_name=op_name,
        spec=spec,
        meta_fn=meta_fn,
        supports_dynamic_shapes=supports_dynamic_shapes,
        requires_output_allocator=requires_output_allocator,
        priority=priority,
        capability_validator=capability_validator,
        register_torch_op=True,
        schema=schema,
        precompiled_ptx=ptx,
    )

"""Tests for custom_cuda_kernel_op and its lower-level building blocks."""

import pytest
import torch

import torch_tensorrt
import torch_tensorrt.kernels as ttk
from torch_tensorrt.kernels._custom_plugin._descriptor import _infer_schema

from .conftest import (
    SIGMOID_SRC,
    make_eager_sigmoid,
    make_sigmoid_aot,
    skip_no_cuda,
    skip_no_qdp,
)

# ---- No-GPU: CudaPythonSpec construction ----


class TestCudaPythonSpec:
    def test_basic_construction(self):
        spec = ttk.cuda_python("// src", "my_k", aot_fn=lambda *a: None)
        assert spec.kernel_source == "// src"
        assert spec.kernel_name == "my_k"
        assert spec.compile_std == "c++17"
        assert "/usr/local/cuda/include" in spec.include_paths

    def test_overrides(self):
        spec = ttk.cuda_python(
            "// s",
            "k",
            include_paths=["/opt/cuda/include"],
            arch_override="sm_90",
        )
        assert spec.include_paths == ["/opt/cuda/include"]
        assert spec.arch_override == "sm_90"

    def test_aot_fn_settable_post_construction(self):
        spec = ttk.cuda_python("// s", "k")
        assert spec.aot_fn is None
        spec.aot_fn = lambda *a: None
        assert spec.aot_fn is not None


# ---- No-GPU: schema inference (small defs — needs real __annotations__) ----


def test_schema_single_tensor():
    def meta(x: torch.Tensor) -> torch.Tensor:
        return torch.empty_like(x)

    s = _infer_schema(meta)
    assert "Tensor x" in s and "-> Tensor" in s


def test_schema_two_tensors():
    def meta(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.empty_like(a)

    s = _infer_schema(meta)
    assert "Tensor a" in s and "Tensor b" in s


def test_schema_mixed_scalar():
    def meta(x: torch.Tensor, scale: float) -> torch.Tensor:
        return torch.empty_like(x)

    s = _infer_schema(meta)
    assert "Tensor x" in s and "float scale" in s


# ---- No-GPU: decorator plumbing ----


def test_custom_plugin_forwards_explicit_schema(monkeypatch):
    from torch_tensorrt.kernels._custom_plugin import _descriptor

    captured = {}
    monkeypatch.setattr(
        _descriptor,
        "register_cuda_python_plugin",
        lambda *a, **k: captured.update(k),
    )

    spec = ttk.cuda_python("// s", "k", aot_fn=lambda *a: None)

    @ttk.custom_plugin("ttk_test::schema_forward", spec, schema="(Tensor x) -> Tensor")
    def _meta(x: torch.Tensor) -> torch.Tensor:
        return torch.empty_like(x)

    assert captured["schema"] == "(Tensor x) -> Tensor"
    assert captured["register_torch_op"] is True


def test_custom_cuda_kernel_op_one_shot(monkeypatch):
    from torch_tensorrt.kernels._custom_plugin import _descriptor

    captured = {}
    monkeypatch.setattr(
        _descriptor,
        "register_cuda_python_plugin",
        lambda *a, **k: captured.update(k),
    )

    def _meta(x: torch.Tensor) -> torch.Tensor:
        return torch.empty_like(x)

    ttk.custom_cuda_kernel_op(
        op_name="ttk_test::one_shot",
        kernel_source="// s",
        kernel_name="k",
        meta_fn=_meta,
        eager_fn=lambda x: x,
        aot_fn=lambda *a: None,
        schema="(Tensor x) -> Tensor",
        supports_dynamic_shapes=True,
    )

    assert captured["op_name"] == "ttk_test::one_shot"
    assert captured["schema"] == "(Tensor x) -> Tensor"
    assert captured["supports_dynamic_shapes"] is True


def test_precompiled_ptx_skips_nvrtc(monkeypatch):
    """register_cuda_python_plugin(precompiled_ptx=...) must not call compile_to_ptx."""
    from torch_tensorrt.kernels._custom_plugin import _descriptor, _nvrtc
    from torch_tensorrt.kernels._specs import CudaPythonSpec

    def _fail(*a, **k):
        raise AssertionError(
            "compile_to_ptx must NOT run when precompiled_ptx is provided"
        )

    monkeypatch.setattr(_nvrtc, "compile_to_ptx", _fail)
    for name in (
        "_register_pytorch_op",
        "_register_aot_impl",
        "custom_op",
    ):
        monkeypatch.setattr(_descriptor, name, lambda *a, **k: None)

    spec = CudaPythonSpec(
        kernel_source="// ignored",
        kernel_name="k",
        aot_fn=lambda *a: None,
        eager_fn=lambda x: x,
    )

    def _meta(x: torch.Tensor) -> torch.Tensor:
        return torch.empty_like(x)

    _descriptor.register_cuda_python_plugin(
        op_name="ttk_test::ptx_reused",
        spec=spec,
        meta_fn=_meta,
        precompiled_ptx=b"fake-ptx",
    )


def test_register_pytorch_op_partial_failure_is_atomic(monkeypatch):
    """A failure inside _register_pytorch_op must not leave the op half-registered.

    Without atomic teardown, ``lib.define`` from the first attempt would leave
    ``torch.ops.<ns>.<name>`` populated, ``_torch_op_already_registered`` would
    short-circuit on retry, and the user would silently keep a broken state
    with no CUDA / fake impl.
    """
    from torch_tensorrt.kernels._custom_plugin import _descriptor

    op_name = "ttk_test::partial_recovery"

    def _meta(x: torch.Tensor) -> torch.Tensor:
        return torch.empty_like(x)

    real_register_fake = torch.library.register_fake
    call_count = {"n": 0}

    def _flaky_register_fake(op):
        def _inner(fn):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise RuntimeError("simulated upstream failure")
            return real_register_fake(op)(fn)

        return _inner

    monkeypatch.setattr(torch.library, "register_fake", _flaky_register_fake)

    # First attempt: simulated failure must propagate.
    with pytest.raises(RuntimeError, match="simulated upstream failure"):
        _descriptor._register_pytorch_op(op_name, _meta, eager_fn=None)

    # After failure, the op must look un-registered so a retry can recover.
    assert not _descriptor._torch_op_already_registered(op_name)

    # Second attempt: the rigged register_fake lets this one through.
    _descriptor._register_pytorch_op(op_name, _meta, eager_fn=None)
    assert _descriptor._torch_op_already_registered(op_name)


# ---- GPU: NVRTC compilation ----


@skip_no_cuda
@skip_no_qdp
class TestNVRTC:
    def test_compiles_to_ptx(self):
        from torch_tensorrt.kernels._custom_plugin._nvrtc import compile_to_ptx

        ptx, _, _ = compile_to_ptx(
            SIGMOID_SRC, "ttk_test_sigmoid", ["/usr/local/cuda/include"]
        )
        assert isinstance(ptx, bytes) and b"ttk_test_sigmoid" in ptx

    def test_invalid_source_raises(self):
        from torch_tensorrt.kernels._custom_plugin._nvrtc import compile_to_ptx

        with pytest.raises(Exception):
            compile_to_ptx(
                "this is not valid CUDA !!!###", "bad", ["/usr/local/cuda/include"]
            )

    def test_arch_override_respected(self):
        from torch_tensorrt.kernels._custom_plugin._nvrtc import compile_to_ptx

        arch = f"sm_{torch.cuda.get_device_capability()[0]}0"
        ptx, _, _ = compile_to_ptx(
            SIGMOID_SRC,
            "ttk_test_sigmoid",
            ["/usr/local/cuda/include"],
            arch_override=arch,
        )
        assert isinstance(ptx, bytes)


# ---- GPU: integration — register, eager, TRT compile w/ dynamic shapes ----


def _register_sigmoid(op_name: str):
    spec = ttk.cuda_python(
        SIGMOID_SRC,
        "ttk_test_sigmoid",
        aot_fn=make_sigmoid_aot(),
        eager_fn=make_eager_sigmoid(),
    )

    @ttk.custom_plugin(op_name, spec, supports_dynamic_shapes=True)
    def _meta(x: torch.Tensor) -> torch.Tensor:
        return torch.empty_like(x)


@skip_no_cuda
@skip_no_qdp
class TestIntegration:
    def test_register_and_eager(self):
        try:
            _register_sigmoid("ttk_test::sigmoid_eager")
        except Exception:
            pass
        x = torch.randn(1024, device="cuda")
        assert torch.allclose(
            torch.ops.ttk_test.sigmoid_eager(x), torch.sigmoid(x), atol=1e-4, rtol=1e-4
        )

    def test_trt_compile_dynamic_shapes(self):
        try:
            _register_sigmoid("ttk_test::sigmoid_dyn")
        except Exception:
            pass

        class M(torch.nn.Module):
            def forward(self, x):
                return torch.ops.ttk_test.sigmoid_dyn(x)

        inputs = [
            torch_tensorrt.Input(
                min_shape=(1, 128),
                opt_shape=(1, 512),
                max_shape=(1, 2048),
                dtype=torch.float32,
            )
        ]
        trt = torch_tensorrt.compile(
            M().cuda().eval(),
            inputs=inputs,
            enabled_precisions={torch.float32},
            min_block_size=1,
        )
        for size in [128, 512, 2048]:
            x = torch.randn(1, size, device="cuda")
            with torch.no_grad():
                assert torch.allclose(trt(x), torch.sigmoid(x), atol=1e-2, rtol=1e-2)


@skip_no_cuda
@skip_no_qdp
def test_schema_override_integration():
    """End-to-end: schema= overrides the inferred schema at real registration."""
    src = """
    extern "C" __global__ void schema_ov_noop(
            const float* x, int n, float alpha, float* y) {}
    """
    spec = ttk.cuda_python(
        src,
        "schema_ov_noop",
        aot_fn=make_sigmoid_aot(),
        eager_fn=lambda x, alpha: (
            alpha * x
        ),  # reference impl — doesn't touch the kernel
    )

    @ttk.custom_plugin(
        "ttk_test::schema_ov",
        spec,
        supports_dynamic_shapes=True,
        schema="(Tensor x, float alpha) -> Tensor",
    )
    def _meta(x, alpha):  # no hints — only schema= makes `float alpha` land
        return torch.empty_like(x)

    schemas = [
        str(s) for s in torch._C._jit_get_schemas_for_operator("ttk_test::schema_ov")
    ]
    assert any("float alpha" in s for s in schemas)

    x = torch.randn(32, device="cuda")
    assert torch.allclose(torch.ops.ttk_test.schema_ov(x, 2.5), 2.5 * x, atol=1e-5)

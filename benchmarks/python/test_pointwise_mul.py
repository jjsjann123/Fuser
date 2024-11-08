# SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
import pytest
from nvfuser import FusionDefinition, DataType
from nvfuser.pytorch_utils import torch_dtype_to_nvfuser_dtype
from .core import run_benchmark, clear_dynamo_cache
import torch
from .global_params import generate_input_sizes, FLOAT_DTYPES, PROMOTE_DTYPES


def pointwise_mul_fusion(
    fd: FusionDefinition,
    dtype: DataType,
) -> None:
    T0 = fd.define_tensor(
        shape=[-1, -1], contiguity=[True, True], dtype=dtype, is_cpu=False
    )
    if dtype in PROMOTE_DTYPES:
        T0 = fd.ops.cast(T0, dtype=DataType.Float)
    T2 = fd.ops.mul(T0, T0)
    if dtype in PROMOTE_DTYPES:
        T2 = fd.ops.cast(T2, dtype=dtype)
    fd.add_output(T2)


def pointwise_mul_fwd_fn(inputs: list):  # in_tensor
    return torch.mul(inputs[0], inputs[0])


@pytest.mark.parametrize("size", generate_input_sizes(dims=2))
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_pointwise_mul_nvf_benchmark(
    benchmark,
    size: tuple,
    dtype: torch.dtype,
    disable_validation: bool,
    disable_benchmarking: bool,
):
    inputs = [torch.randn(size, device="cuda", dtype=dtype)]

    with FusionDefinition() as fd:
        pointwise_mul_fusion(fd, torch_dtype_to_nvfuser_dtype(dtype))

    if not disable_validation:
        eager_output = torch.mul(inputs[0].to(torch.double), inputs[0].to(torch.double))
        fd.validate(inputs, [eager_output.to(dtype)])

    if not disable_benchmarking:
        run_benchmark(benchmark, fd.execute, inputs)


@pytest.mark.parametrize("executor", ["eager", "torchcompile"])
@pytest.mark.parametrize("size", generate_input_sizes(dims=2))
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_pointwise_mul_baseline_benchmark(
    benchmark,
    size: tuple,
    dtype: torch.dtype,
    executor: str,
):
    if executor == "torchcompile":
        clear_dynamo_cache()
    input = torch.randn(size, device="cuda", dtype=dtype)

    benchmark_fn = {
        "eager": pointwise_mul_fwd_fn,
        "torchcompile": torch.compile(pointwise_mul_fwd_fn),
    }
    # Inputs and outputs are same as nvFuser, no need for manual IOByte computation
    run_benchmark(
        benchmark,
        benchmark_fn[executor],
        [input],
    )

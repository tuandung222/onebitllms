# Copyright 2025 The Falcon-LLM Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Triton fast paths for llama.cpp-compatible fake quantizers."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from .llama_cpp_quant import Q8_BLOCK_SIZE, _quantize_ste


@triton.jit
def _round_away_from_zero_tl(x):
    return tl.where(x >= 0.0, tl.floor(x + 0.5), tl.ceil(x - 0.5))


@triton.jit
def _fake_quant_q8_0_kernel(
    x_ptr,
    y_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    block_id = tl.program_id(0)
    offsets = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offsets).to(tl.float32)

    amax = tl.max(tl.abs(x), axis=0)
    d_raw = amax / 127.0
    d = d_raw.to(tl.float16).to(tl.float32)
    inv_d = tl.where(d_raw != 0.0, 1.0 / d_raw, 0.0)

    q = _round_away_from_zero_tl(x * inv_d)
    q = tl.clamp(q, -128.0, 127.0)
    y = q * d
    tl.store(y_ptr + offsets, y)


@torch.compiler.disable
def fake_quant_q8_0_triton(x: torch.Tensor, *, use_ste: bool = True) -> torch.Tensor:
    """Fake quantize with Q8_0 using a Triton CUDA kernel.

    This is a fast path for CUDA QAT. It mirrors ``fake_quant_q8_0`` and keeps
    that PyTorch implementation as the reference source of truth.
    """
    if not x.is_cuda:
        raise ValueError("fake_quant_q8_0_triton requires a CUDA tensor")
    if x.shape[-1] % Q8_BLOCK_SIZE != 0:
        raise ValueError(f"last dimension {x.shape[-1]} must be divisible by block_size={Q8_BLOCK_SIZE}")

    x_contiguous = x.contiguous()
    y = torch.empty_like(x_contiguous)
    n_blocks = x_contiguous.numel() // Q8_BLOCK_SIZE
    _fake_quant_q8_0_kernel[(n_blocks,)](
        x_contiguous,
        y,
        BLOCK_SIZE=Q8_BLOCK_SIZE,
    )
    y = y.reshape(x.shape)
    return _quantize_ste(x, y.to(dtype=x.dtype)) if use_ste else y.to(dtype=x.dtype)

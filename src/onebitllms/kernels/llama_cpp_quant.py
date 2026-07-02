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
"""llama.cpp-compatible fake quantizers.

These functions simulate quantize-dequantize noise for QAT. They do not pack
GGUF bytes and they are not inference kernels. The formulas intentionally mirror
the deterministic llama.cpp reference quantizers for Q1_0, Q2_0, Q4_0, Q4_1,
and Q8_0 activation fake quantization.
"""

from __future__ import annotations

import torch


Q1_BLOCK_SIZE = 128
Q2_BLOCK_SIZE = 128
Q4_BLOCK_SIZE = 32
Q8_BLOCK_SIZE = 32


class _STEQuantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, x_quant: torch.Tensor) -> torch.Tensor:
        del ctx
        return x_quant

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        del ctx
        return grad_output, None


def _quantize_ste(x: torch.Tensor, x_quant: torch.Tensor) -> torch.Tensor:
    return _STEQuantize.apply(x, x_quant)


def _fp16_round(x: torch.Tensor) -> torch.Tensor:
    return x.to(torch.float16).to(torch.float32)


def _round_away_from_zero(x: torch.Tensor) -> torch.Tensor:
    """Match C/C++ std::round for deterministic ggml quantization formulas."""
    return torch.where(x >= 0, torch.floor(x + 0.5), torch.ceil(x - 0.5))


def _as_float_blocks(x: torch.Tensor, block_size: int) -> tuple[torch.Tensor, torch.Size]:
    if x.shape[-1] % block_size != 0:
        raise ValueError(f"last dimension {x.shape[-1]} must be divisible by block_size={block_size}")
    orig_shape = x.shape
    x_f32 = x.contiguous().to(torch.float32)
    return x_f32.view(-1, x.shape[-1] // block_size, block_size), orig_shape


def _restore_blocks(x_blocks: torch.Tensor, orig_shape: torch.Size, dtype: torch.dtype) -> torch.Tensor:
    return x_blocks.reshape(orig_shape).to(dtype)


def fake_quant_q1_0(x: torch.Tensor, *, use_ste: bool = True) -> torch.Tensor:
    """Fake quantize with the prism-llama-cpp Q1_0 quantize-dequantize rule.

    Per 128-value block:

    - compute ``d_raw = mean(abs(x))``;
    - store/decode ``d = fp16(d_raw)``;
    - store one sign bit per value, where ``x >= 0`` maps to ``+d``;
    - dequantize to ``+d`` or ``-d``.
    """
    xb, orig_shape = _as_float_blocks(x, Q1_BLOCK_SIZE)
    d = _fp16_round(xb.abs().mean(dim=-1, keepdim=True))
    sign = torch.where(xb >= 0.0, torch.ones_like(xb), -torch.ones_like(xb))
    xq = d * sign
    xq = _restore_blocks(xq, orig_shape, x.dtype)
    return _quantize_ste(x, xq) if use_ste else xq


def fake_quant_q2_0(x: torch.Tensor, *, use_ste: bool = True) -> torch.Tensor:
    """Fake quantize with the prism-llama-cpp Q2_0 quantize-dequantize rule.

    Per 128-value block:

    - compute ``d_raw = max(abs(x))``;
    - store/decode ``d = fp16(d_raw)``;
    - encode ``q = clamp(std::round(x / d_raw) + 1, 0, 3)``;
    - dequantize with ``(q - 1) * d``.
    """
    xb, orig_shape = _as_float_blocks(x, Q2_BLOCK_SIZE)
    amax = xb.abs().max(dim=-1, keepdim=True).values
    d = _fp16_round(amax)
    inv_d = torch.where(amax > 0.0, 1.0 / amax, torch.zeros_like(amax))

    q = _round_away_from_zero(xb * inv_d) + 1.0
    q = torch.clamp(q, 0.0, 3.0)
    xq = (q - 1.0) * d
    xq = _restore_blocks(xq, orig_shape, x.dtype)
    return _quantize_ste(x, xq) if use_ste else xq


def fake_quant_q4_0(x: torch.Tensor, *, use_ste: bool = True) -> torch.Tensor:
    """Fake quantize with the llama.cpp Q4_0 quantize-dequantize rule.

    Per 32-value block:

    - choose the signed value with largest absolute magnitude;
    - compute ``d_raw = signed_absmax / -8``;
    - select 4-bit codes with ``trunc(x / d_raw + 8.5)``;
    - dequantize with ``(q - 8) * fp16(d_raw)``.
    """
    xb, orig_shape = _as_float_blocks(x, Q4_BLOCK_SIZE)
    abs_x = xb.abs()
    max_idx = abs_x.argmax(dim=-1, keepdim=True)
    signed_absmax = xb.gather(dim=-1, index=max_idx)
    d_raw = signed_absmax / -8.0
    d = _fp16_round(d_raw)
    inv_d = torch.where(d_raw != 0, 1.0 / d_raw, torch.zeros_like(d_raw))

    q = torch.trunc(xb * inv_d + 8.5)
    q = torch.clamp(q, 0.0, 15.0)
    xq = (q - 8.0) * d
    xq = _restore_blocks(xq, orig_shape, x.dtype)
    return _quantize_ste(x, xq) if use_ste else xq


def fake_quant_q4_1(x: torch.Tensor, *, use_ste: bool = True) -> torch.Tensor:
    """Fake quantize with the llama.cpp Q4_1 quantize-dequantize rule.

    Per 32-value block:

    - compute block ``min`` and ``max``;
    - compute ``d_raw = (max - min) / 15``;
    - select 4-bit codes with ``trunc((x - min) / d_raw + 0.5)``;
    - dequantize with ``q * fp16(d_raw) + fp16(min)``.
    """
    xb, orig_shape = _as_float_blocks(x, Q4_BLOCK_SIZE)
    x_min = xb.min(dim=-1, keepdim=True).values
    x_max = xb.max(dim=-1, keepdim=True).values
    d_raw = (x_max - x_min) / 15.0
    d = _fp16_round(d_raw)
    m = _fp16_round(x_min)
    inv_d = torch.where(d_raw != 0, 1.0 / d_raw, torch.zeros_like(d_raw))

    q = torch.trunc((xb - x_min) * inv_d + 0.5)
    q = torch.clamp(q, 0.0, 15.0)
    xq = q * d + m
    xq = _restore_blocks(xq, orig_shape, x.dtype)
    return _quantize_ste(x, xq) if use_ste else xq


def fake_quant_q8_0_activation(x: torch.Tensor, *, use_ste: bool = True) -> torch.Tensor:
    """Fake quantize activations with the llama.cpp Q8_0 quantize-dequantize rule.

    Per 32-value block:

    - compute ``d_raw = max(abs(x)) / 127``;
    - store/decode ``d = fp16(d_raw)``;
    - encode signed int8 codes with C/C++ ``roundf(x / d_raw)`` semantics;
    - dequantize with ``q * d``.

    This is intended for transient QAT activation noise. It does not mean
    activations are stored in GGUF.
    """
    xb, orig_shape = _as_float_blocks(x, Q8_BLOCK_SIZE)
    d_raw = xb.abs().max(dim=-1, keepdim=True).values / 127.0
    d = _fp16_round(d_raw)
    inv_d = torch.where(d_raw != 0, 1.0 / d_raw, torch.zeros_like(d_raw))

    q = _round_away_from_zero(xb * inv_d)
    q = torch.clamp(q, -128.0, 127.0)
    xq = q * d
    xq = _restore_blocks(xq, orig_shape, x.dtype)
    return _quantize_ste(x, xq) if use_ste else xq

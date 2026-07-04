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
"""Linear wrappers for llama.cpp fake quantization-aware training."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from onebitllms.kernels import (
    fake_quant_q1_0,
    fake_quant_q2_0,
    fake_quant_q4_0,
    fake_quant_q4_1,
    fake_quant_q8_0,
    fake_quant_q8_0_activation,
    fake_quant_q8_1,
)


_WEIGHT_QUANTIZERS = {
    "Q1_0": fake_quant_q1_0,
    "Q2_0": fake_quant_q2_0,
    "Q4_0": fake_quant_q4_0,
    "Q4_1": fake_quant_q4_1,
    "Q8_0": fake_quant_q8_0,
    "Q8_1": fake_quant_q8_1,
}

_QUANT_BLOCK_SIZES = {
    "Q1_0": 128,
    "Q2_0": 128,
    "Q4_0": 32,
    "Q4_1": 32,
    "Q8_0": 32,
    "Q8_1": 32,
}

_ACTIVATION_QUANTIZERS = {
    "Q8_0": fake_quant_q8_0_activation,
}


class LlamaCppFakeQuantLinear(nn.Module):
    """nn.Linear equivalent with llama.cpp weight fake quantization.

    The layer keeps trainable floating-point weights, applies a fake
    quantize-dequantize transform in forward, and uses straight-through
    gradients for QAT. It does not pack GGUF bytes and is not a llama.cpp
    inference kernel.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        quant_type: str = "Q4_0",
        activation_quant: Optional[str] = None,
        accumulator_dtype: Optional[torch.dtype] = torch.float32,
    ) -> None:
        super().__init__()
        quant_key = quant_type.upper()
        if quant_key not in _WEIGHT_QUANTIZERS:
            allowed = ", ".join(sorted(_WEIGHT_QUANTIZERS))
            raise ValueError(f"unsupported llama.cpp fake quant type {quant_type!r}; allowed: {allowed}")
        block_size = _QUANT_BLOCK_SIZES[quant_key]
        if in_features % block_size != 0:
            raise ValueError(
                f"in_features={in_features} must be divisible by {block_size} for {quant_key} fake quantization"
            )
        activation_quant_key = activation_quant.upper() if activation_quant is not None else None
        if activation_quant_key is not None and activation_quant_key not in _ACTIVATION_QUANTIZERS:
            allowed = ", ".join(sorted(_ACTIVATION_QUANTIZERS))
            raise ValueError(
                f"unsupported llama.cpp activation fake quant type {activation_quant!r}; allowed: {allowed}"
            )
        if activation_quant_key == "Q8_0" and in_features % 32 != 0:
            raise ValueError(f"in_features={in_features} must be divisible by 32 for Q8_0 activation fake quantization")

        self.in_features = in_features
        self.out_features = out_features
        self.quant_type = quant_key
        self.activation_quant = activation_quant_key
        self.accumulator_dtype = accumulator_dtype

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / fan_in**0.5 if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    @classmethod
    def from_linear(
        cls,
        layer: nn.Linear,
        *,
        quant_type: str = "Q4_0",
        activation_quant: Optional[str] = None,
        accumulator_dtype: Optional[torch.dtype] = torch.float32,
    ) -> "LlamaCppFakeQuantLinear":
        wrapped = cls(
            layer.in_features,
            layer.out_features,
            bias=layer.bias is not None,
            quant_type=quant_type,
            activation_quant=activation_quant,
            accumulator_dtype=accumulator_dtype,
        )
        wrapped.weight.data.copy_(layer.weight.data)
        wrapped.weight.requires_grad_(layer.weight.requires_grad)
        if layer.bias is not None and wrapped.bias is not None:
            wrapped.bias.data.copy_(layer.bias.data)
            wrapped.bias.requires_grad_(layer.bias.requires_grad)
        wrapped.train(layer.training)
        return wrapped.to(device=layer.weight.device, dtype=layer.weight.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation_quant is not None:
            activation_quantizer = _ACTIVATION_QUANTIZERS[self.activation_quant]
            x = activation_quantizer(x)

        weight_quantizer = _WEIGHT_QUANTIZERS[self.quant_type]
        w = weight_quantizer(self.weight)

        if self.accumulator_dtype is not None:
            out = F.linear(
                x.to(self.accumulator_dtype),
                w.to(self.accumulator_dtype),
                self.bias.to(self.accumulator_dtype) if self.bias is not None else None,
            )
            return out.to(x.dtype)
        return F.linear(x, w, self.bias)

    def to_linear(self) -> nn.Linear:
        """Convert back to nn.Linear while preserving trained float weights."""
        layer = nn.Linear(
            self.in_features,
            self.out_features,
            bias=self.bias is not None,
            device=self.weight.device,
            dtype=self.weight.dtype,
        )
        layer.weight.data.copy_(self.weight.data)
        layer.weight.requires_grad_(self.weight.requires_grad)
        if self.bias is not None and layer.bias is not None:
            layer.bias.data.copy_(self.bias.data)
            layer.bias.requires_grad_(self.bias.requires_grad)
        layer.train(self.training)
        return layer

    def __repr__(self) -> str:
        return (
            f"LlamaCppFakeQuantLinear(in_features={self.in_features}, "
            f"out_features={self.out_features}, quant_type={self.quant_type}, "
            f"activation_quant={self.activation_quant})"
        )

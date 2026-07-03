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
from collections.abc import Iterable, Mapping
from typing import Optional

import torch
import torch.nn as nn

from onebitllms.layers import BitNetLinear, LlamaCppFakeQuantLinear


def _state_dict_keys(obj) -> tuple[str, ...]:
    if isinstance(obj, nn.Module):
        return tuple(obj.state_dict().keys())
    if isinstance(obj, Mapping):
        return tuple(obj.keys())
    if isinstance(obj, Iterable):
        return tuple(obj)
    raise TypeError(f"expected nn.Module, mapping, or iterable of keys, got {type(obj)!r}")


def assert_state_dict_keys_unchanged(before, after) -> bool:
    """Assert two model/key snapshots have identical state_dict key order.

    ``before`` and ``after`` may be modules, state_dict mappings, or iterables of
    key strings. The function returns True when keys match and raises
    AssertionError otherwise.
    """
    before_keys = _state_dict_keys(before)
    after_keys = _state_dict_keys(after)
    if before_keys != after_keys:
        before_set = set(before_keys)
        after_set = set(after_keys)
        missing = sorted(before_set - after_set)
        unexpected = sorted(after_set - before_set)
        raise AssertionError(
            "state_dict keys changed after fake quant module transformation; "
            f"missing={missing}, unexpected={unexpected}"
        )
    return True

def replace_linear_with_bitnet_linear(model, previous_dtype: Optional[torch.dtype] = None):
    """
    """
    # Recursively replace linear layers
    if previous_dtype is None:
        previous_dtype = torch.get_default_dtype()

        model_dtype = model.dtype
        torch.set_default_dtype(model_dtype)

        previous_dtype = model_dtype

    for name, module in model.named_children():
        if len(list(module.children())) > 0:
            replace_linear_with_bitnet_linear(module, previous_dtype=previous_dtype)
        
        # Replace nn.Linear layers, but skip 'lm_head'
        if name != 'lm_head' and isinstance(module, nn.Linear):
            in_features = module.in_features
            out_features = module.out_features
            bias = module.bias is not None

            with torch.device(module.weight.device):
                # Create a new instance of the custom linear layer
                new_layer = BitNetLinear(in_features, out_features, bias=bias)
                # Copy weights and biases
                with torch.no_grad():
                    new_layer.weight.copy_(module.weight)
                    if bias:
                        new_layer.bias.copy_(module.bias)
            
            # Replace the layer in the model
            setattr(model, name, new_layer)
    return model


def replace_linear_with_llama_cpp_fake_quant_linear(
    model,
    quant_type: str = "Q4_0",
    target_names: Optional[tuple[str, ...]] = None,
    skip_names: tuple[str, ...] = ("lm_head",),
    activation_quant: Optional[str] = None,
    accumulator_dtype: Optional[torch.dtype] = torch.float32,
    skip_if_not_divisible_by: Optional[int] = None,
):
    """Recursively replace nn.Linear layers with llama.cpp fake-quant layers.

    Args:
        model:
            Model containing ``nn.Linear`` modules.
        quant_type:
            ``"Q1_0"``, ``"Q2_0"``, ``"Q4_0"``, or ``"Q4_1"``.
        target_names:
            Optional name substrings. If provided, only child module names
            containing at least one target substring are replaced.
        skip_names:
            Child module names to keep untouched. Defaults to ``("lm_head",)``.
        activation_quant:
            Optional activation fake quantizer. Currently supports ``"Q8_0"``.
        accumulator_dtype:
            Optional dtype used for ``F.linear`` accumulation in the wrapper.
        skip_if_not_divisible_by:
            Skip linears whose input feature dimension is not divisible by the
            llama.cpp block size. Defaults to 128 for Q1_0/Q2_0 and 32 for
            Q4_0/Q4_1.

    Returns:
        The input model, modified in-place.
    """
    quant_key = quant_type.upper()
    activation_quant_key = activation_quant.upper() if activation_quant is not None else None
    if skip_if_not_divisible_by is None:
        block_sizes = {"Q1_0": 128, "Q2_0": 128, "Q4_0": 32, "Q4_1": 32}
        if quant_key not in block_sizes:
            allowed = ", ".join(sorted(block_sizes))
            raise ValueError(f"unsupported llama.cpp fake quant type {quant_type!r}; allowed: {allowed}")
        skip_if_not_divisible_by = block_sizes[quant_key]
    if activation_quant_key is not None:
        if activation_quant_key != "Q8_0":
            raise ValueError(f"unsupported llama.cpp activation fake quant type {activation_quant!r}; allowed: Q8_0")
        skip_if_not_divisible_by = max(skip_if_not_divisible_by, 32)

    for name, module in model.named_children():
        if len(list(module.children())) > 0:
            replace_linear_with_llama_cpp_fake_quant_linear(
                module,
                quant_type=quant_key,
                target_names=target_names,
                skip_names=skip_names,
                activation_quant=activation_quant_key,
                accumulator_dtype=accumulator_dtype,
                skip_if_not_divisible_by=skip_if_not_divisible_by,
            )

        if name in skip_names or not isinstance(module, nn.Linear):
            continue
        if target_names is not None and not any(target in name for target in target_names):
            continue
        if module.in_features % skip_if_not_divisible_by != 0:
            continue

        new_layer = LlamaCppFakeQuantLinear.from_linear(
            module,
            quant_type=quant_key,
            activation_quant=activation_quant_key,
            accumulator_dtype=accumulator_dtype,
        )
        setattr(model, name, new_layer)
    return model


def replace_llama_cpp_fake_quant_linear_with_linear(model):
    """Recursively convert llama.cpp fake-quant linear wrappers back to nn.Linear.

    Use this after QAT and before Hugging Face checkpoint saving or GGUF export
    when downstream code expects standard ``nn.Linear`` modules.
    """
    for name, module in model.named_children():
        if isinstance(module, LlamaCppFakeQuantLinear):
            setattr(model, name, module.to_linear())
        else:
            replace_llama_cpp_fake_quant_linear_with_linear(module)
    return model

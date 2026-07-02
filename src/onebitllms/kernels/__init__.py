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
from .llama_cpp_quant import (
    fake_quant_q1_0,
    fake_quant_q2_0,
    fake_quant_q4_0,
    fake_quant_q4_1,
    fake_quant_q8_0_activation,
)


def _missing_triton_kernel(*args, **kwargs):
    raise ModuleNotFoundError(
        "Triton is required for onebitllms BitNet CUDA kernels. Install triton "
        "in the training environment before calling activation_quant_triton or "
        "weight_quant_triton."
    )


try:
    from .activation_quant import activation_quant_triton
except ModuleNotFoundError as exc:
    if exc.name != "triton":
        raise
    activation_quant_triton = _missing_triton_kernel

try:
    from .weight_quant import weight_quant_triton
except ModuleNotFoundError as exc:
    if exc.name != "triton":
        raise
    weight_quant_triton = _missing_triton_kernel

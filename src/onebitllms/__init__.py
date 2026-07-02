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
from .kernels import (
    activation_quant_triton,
    fake_quant_q1_0,
    fake_quant_q2_0,
    fake_quant_q4_0,
    fake_quant_q4_1,
    weight_quant_triton,
)
from .layers import BitNetLinear, LlamaCppFakeQuantLinear
from .utils import (
    convert_to_bf16,
    quantize_to_1bit,
    replace_linear_with_bitnet_linear,
    replace_linear_with_llama_cpp_fake_quant_linear,
)

__version__ = "0.0.5.dev0"

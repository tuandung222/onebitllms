
<p align="center">
  <img src="assets/onebitllms-logo.png" alt="logo" width="400" >
</p>

> `onebitllms` is a lightweight python package that can be used to easily perform Large-Language Model (LLMs) fine-tuning following the training procedure from BitNet, to produce customized 1.58-bit LLMs.

## Requirements

Currently the library only works for NVIDIA CUDA compiled GPUs.

## Installation

Download the package through pip:

```bash
pip install onebitllms
```

or directly from source:

```bash
pip install git+https://github.com/tiiuae/onebitllms.git
```

## Getting started

### 1.58bit Fine-tuning

Simply use the `replace_linear_with_bitnet_linear` after loading the **pre-quantized** checkpoint, and use directly that model for fine-tuning (e.g. with `SFTTrainer` from `trl` library):

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from onebitllms import replace_linear_with_bitnet_linear

model_id = "tiiuae/Falcon-E-1B-Base"
revision = "prequantized"

tokenizer = AutoTokenizer.from_pretrained(
    model_id, revision=revision
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    revision=revision,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

model = replace_linear_with_bitnet_linear(model)

# Do the training here ...
```

After training your model, make sure to quantize the final checkpoint in 1-bit precision with the method `quantize_to_1bit`:

```python
from onebitllms import quantize_to_1bit

model_output_dir = ""
quantized_model_output_dir = ""

quantize_to_1bit(model_output_dir, quantized_model_output_dir)
```

Example scripts can be found in `examples/`. We use the following command on a 8xA10G 23GB instance and training took ~8.5 hours.

```bash
python examples/sft.py \
    --model_name_or_path tiiuae/Falcon-E-1B-Base \
    --model_revision prequantized \
    --torch_dtype bfloat16 \
    --learning_rate 0.0001 \
    --dataset_name trl-lib/Capybara \
    --per_device_train_batch_size 1 \
    --output_dir Falcon-E-Capybara \
    --logging_steps 1 \
    --save_strategy steps \
    --save_steps 100 \
    --packing \
    --gradient_accumulation_steps 16
```

Once your 1.58bit model is ready, we suggest you to deploy it with [`bitnet.cpp` package](https://github.com/microsoft/BitNet).

### Using quantization triton kernels

You can also inject the `BitNetLinear` classes into your pre-training framework:

```python
from onebitllms import BitNetLinear

# inject it in your model classes for pre-training ..
```

You can also use the triton kernels directly for a more fine-grained control over the operations:

```python
from onebitllms import activation_quant_triton, weight_quant_triton
```

The existing Triton kernels are for the BitNet training path:

- `weight_quant_triton`: ternary BitNet weight fake quantization.
- `activation_quant_triton`: row-wise int8 activation fake quantization.

They are not llama.cpp GGUF packing kernels. On GPU training setups, Triton is
the right direction for speed, but the quantization formula should be validated
first with a deterministic PyTorch reference.

### Sử dụng llama.cpp fake quantizers

Fork này cung cấp thêm các fake quantizer cho weight, bám theo công thức
quantize-dequantize xác định trong `prism-llama-cpp` / `ggml`:

```python
from onebitllms import (
    fake_quant_q1_0,
    fake_quant_q2_0,
    fake_quant_q4_0,
    fake_quant_q4_1,
    fake_quant_q8_0,
)

w_q1 = fake_quant_q1_0(weight)
w_q2 = fake_quant_q2_0(weight)
w_q4 = fake_quant_q4_0(weight)
w_q41 = fake_quant_q4_1(weight)
w_q8 = fake_quant_q8_0(weight)
```

Để thay layer trong model, có thể replace trực tiếp các `nn.Linear` tương thích:

```python
from onebitllms import replace_linear_with_llama_cpp_fake_quant_linear

model = replace_linear_with_llama_cpp_fake_quant_linear(model, quant_type="Q2_0")
```

Có thể bật thêm fake quant activation `Q8_0` tạm thời trong quá trình QAT:

```python
model = replace_linear_with_llama_cpp_fake_quant_linear(
    model,
    quant_type="Q4_0",
    activation_quant="Q8_0",
)
```

Sau QAT, hãy chuyển wrapper fake quant về lại `nn.Linear` chuẩn trước khi lưu
checkpoint Hugging Face hoặc export sang GGUF:

```python
from onebitllms import replace_llama_cpp_fake_quant_linear_with_linear

model = replace_llama_cpp_fake_quant_linear_with_linear(model)
model.save_pretrained(output_dir)
```

Các kiểu weight fake quant đang hỗ trợ:

| Type | Block size | Nguồn công thức | Ghi chú |
| --- | ---: | --- | --- |
| `Q1_0` | 128 | `prism-llama-cpp` `quantize_row_q1_0_ref` | 1 sign bit và fp16 scale theo block |
| `Q2_0` | 128 | `prism-llama-cpp` `quantize_row_q2_0_ref` | mã 2-bit với ngữ nghĩa `std::round` của C/C++ |
| `Q4_0` | 32 | llama.cpp `quantize_row_q4_0_ref` | block quantization theo signed absmax |
| `Q4_1` | 32 | llama.cpp `quantize_row_q4_1_ref` | block quantization affine theo min/max |
| `Q8_0` | 32 | llama.cpp `quantize_row_q8_0_ref` / `dequantize_row_q8_0` | target GGUF thật, dùng được với `llama-quantize Q8_0` |
| `Q8_1` | 32 | ggml `quantize_row_q8_1_ref` | format runtime/vector-dot; không phải target export model thông thường của `llama-quantize` |

Các kiểu activation fake quant đang hỗ trợ:

| Type | Block size | Nguồn công thức | Ghi chú |
| --- | ---: | --- | --- |
| `Q8_0` | 32 | llama.cpp `quantize_row_q8_0_ref` | nhiễu int8 activation tạm thời cho QAT |

Các layer này vẫn giữ weight trainable ở floating-point, áp dụng nhiễu
quantize-dequantize trong forward pass, và dùng straight-through gradient cho
QAT. Chúng không export file GGUF và không phải inference kernel. Activation
fake quantization là tùy chọn và không có nghĩa activation được lưu trong GGUF.
Luồng sử dụng dự kiến:

```text
QAT với PyTorch weight được fake quant
-> lưu HF checkpoint
-> convert HF checkpoint sang F16/BF16 GGUF
-> chạy llama-quantize với Q1_0/Q2_0/Q4_0/Q4_1/Q8_0
-> inference bằng llama.cpp / prism-llama-cpp
```

Với QAT 8-bit để chạy inference desktop, nên ưu tiên `quant_type="Q8_0"` vì
`prism-llama-cpp/tools/quantize` expose `Q8_0` như một mode quantize model.
`Q8_1` tồn tại trong ggml như block quantized dùng cho các đường vector-dot;
nhiễu QAT sau dequant cũng là `q * d` như Q8_0, nhưng nên xem là fake quantizer
experimental/runtime-style nếu export path chưa hỗ trợ tensor Q8_1 rõ ràng.

`Q8_K` cũng xuất hiện trong ggml như format K-quant/vector-dot nội bộ, nhưng
không được expose như mode `llama-quantize` thông thường trong fork này. Vì vậy
package chưa expose `Q8_K` như một QAT target.

Tài liệu chi tiết:

- [`docs/llama_cpp_q8_0_qat.md`](docs/llama_cpp_q8_0_qat.md): giải thích công thức `Q8_0`, khác biệt với `Q8_1/Q8_K`, và quy trình QAT -> GGUF.
- [`docs/testing_llama_cpp_fake_quant.md`](docs/testing_llama_cpp_fake_quant.md): kế hoạch kiểm thử nhiều lớp để kiểm tra fake quantizer không lệch khỏi llama.cpp.

### Revert back to bfloat16 format

From our experiments, the BitNet checkpoints are *universal*, meaning we can revert back to bfloat16 format with minimal performance degradation. You can use the method `convert_to_bf16` after training your model:

```python
from onebitllms import convert_to_bf16

model_output_dir = ""
quantized_model_output_dir = ""

convert_to_bf16(model_output_dir, quantized_model_output_dir)
```

## Common questions

*What is 1-bit fine-tuning?*

We term *1-bit fine-tuning* as simply doing continuous training from a **pre-quantized** BitNet compatible checkpoint. As of today, there are multiple ongoing work that explores fine-tuning existing checkpoints into BitNet format but this often leads to poor performance.

*What models can I fine-tune?*

To the best of our knowledge, as of today, only Falcon-Edge and recent Microsoft BitNet series models published their **pre-quantized** checkpoints. If in the future other models gets published together with their **pre-quantized** checkpoints, they should be compatible with `onebitllms` out-of-the-box.

*What else can I do with `onebitllms`?*

You can also use the `BitnetLinear` class exposed in this package and use it inside your pre-training / fine-tuning framework. In contrary to existing implementation, we use optimized triton kernels for computing the quantization errors making the pre-training and fine-tuning much faster than existing implementations. From our experiments, we estimate the overheads between non-BitNet and BitNet pre-training to be around ~20% (to be confirmed with more rigourous analysis).

*Is LoRA supported in `onebitllms`?*

LoRA is not supported with this package and remains a very interesting research question. Unlocking LoRA with `onebitllms` could open-up exciting opportunities such as being able to fine-tune a BitNet 7B on a free-tier Google Colab instance.

*Can `onebitllms` used for inference?*

As of today, BitNet models are extremely interesting for CPU deployment. We strongly encourage users to deploy their models with [`bitnet.cpp`](https://github.com/microsoft/BitNet/) package after fine-tuning it with `onebitllms`. If you want to run it on GPU, to the best of our knowledge, you can use HuggingFace's `transformers` native integration of BitNet models.

*Can I contribute to `onebitllms`?*

Of course. Contributions to enhance the codebase, introduce new features and example scripts are strongly encouraged.

## Citation

If you find this work useful for your research and work, please consider citing us, as well as citing the foundational work behind BitNet models:

```
@misc{tiionebitllms,
    title = {Falcon-E, a series of powerful, universal and fine-tunable 1.58bit language models.},
    author = {Falcon-LLM Team},
    month = {May},
    url={https://github.com/tiiuae/onebitllms}, 
    year = {2025}
}
```

```
@misc{wang2025bitnetcppefficientedgeinference,
      title={Bitnet.cpp: Efficient Edge Inference for Ternary LLMs}, 
      author={Jinheng Wang and Hansong Zhou and Ting Song and Shijie Cao and Yan Xia and Ting Cao and Jianyu Wei and Shuming Ma and Hongyu Wang and Furu Wei},
      year={2025},
      eprint={2502.11880},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2502.11880}, 
}
```

```
@misc{,
      title={1.58-Bit LLM: A New Era of Extreme Quantization}, 
      author={Mohamed Mekkouri and Marc Sun and Leandro von Werra and Thomas Wolf},
      year={2024},
}
```

```
@misc{ma2024era1bitllmslarge,
      title={The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits}, 
      author={Shuming Ma and Hongyu Wang and Lingxiao Ma and Lei Wang and Wenhui Wang and Shaohan Huang and Li Dong and Ruiping Wang and Jilong Xue and Furu Wei},
      year={2024},
      eprint={2402.17764},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2402.17764}, 
}
```

```
@misc{wang2023bitnetscaling1bittransformers,
      title={BitNet: Scaling 1-bit Transformers for Large Language Models}, 
      author={Hongyu Wang and Shuming Ma and Li Dong and Shaohan Huang and Huaijie Wang and Lingxiao Ma and Fan Yang and Ruiping Wang and Yi Wu and Furu Wei},
      year={2023},
      eprint={2310.11453},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2310.11453}, 
}
```

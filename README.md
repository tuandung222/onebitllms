<p align="center">
  <img src="assets/onebitllms-logo.png" alt="onebitllms logo" width="400">
</p>

# onebitllms fork: BitNet fine-tuning và llama.cpp-compatible QAT

Fork này mở rộng `onebitllms` theo hai hướng:

1. Giữ lại đường fine-tuning BitNet / 1.58-bit từ upstream.
2. Thêm các fake quantizer tương thích công thức `llama.cpp` / `prism-llama-cpp` để nghiên cứu QAT trước khi export sang GGUF và PTQ bằng `llama-quantize`.

Mục tiêu thực tế của fork là phục vụ nghiên cứu quy trình:

```text
QAT trong PyTorch
-> lưu Hugging Face checkpoint chuẩn
-> convert sang GGUF F16/BF16
-> PTQ bằng llama.cpp / prism-llama-cpp
-> inference trên desktop
```

Các layer fake quant trong fork này không phải inference kernel và không pack GGUF. Chúng giữ weight trainable ở floating-point, inject nhiễu quantize-dequantize trong forward pass, và dùng straight-through estimator cho backward.

## Tính năng chính

| Nhóm | Trạng thái | Ghi chú |
| --- | --- | --- |
| BitNet / 1.58-bit fine-tuning | Có | Kế thừa upstream `onebitllms`, dùng `BitNetLinear` và Triton kernels |
| Triton quant kernels | Có | Phục vụ đường BitNet training, không phải GGUF packing |
| llama.cpp fake quant weight | Có | `Q1_0`, `Q2_0`, `Q4_0`, `Q4_1`, `Q8_0`, `Q8_1` |
| llama.cpp fake quant activation | Có | `activation_quant="Q8_0"` như nhiễu tạm thời trong QAT |
| Triton fast path cho llama.cpp QAT | Experimental | Hiện có `Q8_0` CUDA fast path, default vẫn là PyTorch |
| Patch/unpatch `nn.Linear` | Có | Dùng để train bằng wrapper rồi export checkpoint chuẩn |
| GGUF export trực tiếp | Không | Dùng converter và `llama-quantize` của llama.cpp |
| Inference kernel | Không | Inference chạy bằng llama.cpp / prism-llama-cpp / bitnet.cpp |

## Cài đặt

Fork này nên được cài từ source:

```bash
git clone https://github.com/tuandung222/onebitllms.git
cd onebitllms
pip install -e .
```

Nếu muốn chạy test:

```bash
pip install -e ".[test]"
```

Yêu cầu chính:

- Python >= 3.9.
- PyTorch.
- `transformers`, `accelerate`, `safetensors`, `huggingface_hub`.
- GPU NVIDIA + Triton nếu dùng đường BitNet CUDA kernels.
- Local `prism-llama-cpp` hoặc `llama.cpp` nếu muốn kiểm thử alignment, convert GGUF, quantize và inference.

## Quick start: llama.cpp-compatible QAT

### 1. Fake quant trực tiếp một tensor

```python
import torch
from onebitllms import fake_quant_q4_0, fake_quant_q4_1, fake_quant_q8_0

weight = torch.randn(128, 256)

w_q4_0 = fake_quant_q4_0(weight)
w_q4_1 = fake_quant_q4_1(weight)
w_q8_0 = fake_quant_q8_0(weight)
```

### 2. Thay `nn.Linear` bằng fake quant wrapper

```python
from transformers import AutoModelForCausalLM
from onebitllms import replace_linear_with_llama_cpp_fake_quant_linear

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-1.7B",
    torch_dtype="auto",
    device_map="auto",
)

model = replace_linear_with_llama_cpp_fake_quant_linear(
    model,
    quant_type="Q4_1",
)
```

Wrapper sẽ thay các `nn.Linear` tương thích block size. Mặc định helper bỏ qua `lm_head`.

### 3. Bật fake quant activation Q8_0 nếu cần

```python
model = replace_linear_with_llama_cpp_fake_quant_linear(
    model,
    quant_type="Q8_0",
    activation_quant="Q8_0",
)
```

Lưu ý: activation fake quant chỉ inject nhiễu trong training. GGUF không lưu activation ở dạng quantized.

### 4. Unpatch trước khi lưu checkpoint

```python
from onebitllms import replace_llama_cpp_fake_quant_linear_with_linear

model = replace_llama_cpp_fake_quant_linear_with_linear(model)
model.save_pretrained("output-hf-checkpoint")
```

Không nên lưu checkpoint HF khi model vẫn còn `LlamaCppFakeQuantLinear`, vì converter của llama.cpp kỳ vọng cấu trúc module/weight chuẩn.

### 5. Dùng Triton fast path cho Q8_0

Default backend là `torch` để giữ tính portable và deterministic. Nếu train trên CUDA và đã cài Triton, có thể bật fast path cho `Q8_0`:

```python
model = replace_linear_with_llama_cpp_fake_quant_linear(
    model,
    quant_type="Q8_0",
    activation_quant="Q8_0",
    backend="auto",
)
```

Các backend:

- `backend="torch"`: luôn dùng PyTorch reference.
- `backend="auto"`: dùng Triton cho CUDA `Q8_0` nếu có thể, fallback về PyTorch khi không hỗ trợ.
- `backend="triton"`: bắt buộc dùng Triton, hiện chỉ hỗ trợ `Q8_0`.

Trước khi dùng `backend="triton"` cho training thật, hãy chạy script validation trên máy GPU:

```bash
PYTHONPATH=src python scripts/check_llama_cpp_q8_0_triton.py --benchmark
```

## Các kiểu llama.cpp fake quant đang hỗ trợ

| Type | Block size | Công thức | Có nên dùng làm target export? |
| --- | ---: | --- | --- |
| `Q1_0` | 128 | `prism-llama-cpp` `quantize_row_q1_0_ref` | Chỉ dùng nếu fork/export path hỗ trợ |
| `Q2_0` | 128 | `prism-llama-cpp` `quantize_row_q2_0_ref` | Chỉ dùng nếu fork/export path hỗ trợ |
| `Q4_0` | 32 | llama.cpp `quantize_row_q4_0_ref` | Có, nếu sẽ PTQ bằng `llama-quantize Q4_0` |
| `Q4_1` | 32 | llama.cpp `quantize_row_q4_1_ref` | Có, nếu sẽ PTQ bằng `llama-quantize Q4_1` |
| `Q8_0` | 32 | llama.cpp `quantize_row_q8_0_ref` / `dequantize_row_q8_0` | Có, đây là target 8-bit chính |
| `Q8_1` | 32 | ggml `quantize_row_q8_1_ref` | Không nên xem là target export thông thường |

`Q8_0` là lựa chọn 8-bit nên ưu tiên. Trong `prism-llama-cpp`, `Q8_0` xuất hiện trực tiếp trong `tools/quantize/quantize.cpp` như một mode quantize model. `Q8_1` có trong `ggml`, nhưng chủ yếu là format runtime/vector-dot và không phải lựa chọn `llama-quantize` thông thường. `Q8_K` cũng là format nội bộ/K-quant nên chưa được expose như QAT target trong fork này.

## Quy trình QAT -> GGUF -> inference

Luồng khuyến nghị:

```text
1. Load model HF.
2. Patch selected Linear layers bằng LlamaCppFakeQuantLinear.
3. Fine-tune/QAT trong PyTorch.
4. Unpatch về nn.Linear.
5. Save HF checkpoint.
6. Convert checkpoint sang GGUF F16/BF16 bằng llama.cpp.
7. Chạy llama-quantize với target khớp fake quant, ví dụ Q8_0.
8. Chạy llama-cli để smoke test inference.
9. Chạy eval cố định để so QAT+PTQ với PTQ-only.
```

Ví dụ phần PTQ sau khi đã có GGUF F16/BF16:

```bash
/path/to/prism-llama-cpp/build/bin/llama-quantize \
  model-f16.gguf \
  model-q8_0.gguf \
  Q8_0
```

Smoke test inference:

```bash
/path/to/prism-llama-cpp/build/bin/llama-cli \
  -m model-q8_0.gguf \
  -p "Explain quantization-aware training in one paragraph." \
  -n 64
```

## Kiểm thử và validation

Fork này có hai lớp test chính cho phần llama.cpp fake quant.

### Unit tests

Nếu có `pytest`:

```bash
PYTHONPATH=src python -m pytest tests/test_llama_cpp_fake_quant.py -q
```

Nếu môi trường chưa có `pytest`:

```bash
PYTHONPATH=src python - <<'PY'
import tests.test_llama_cpp_fake_quant as t

for name in sorted(n for n in dir(t) if n.startswith("test_")):
    getattr(t, name)()
    print(f"{name}: ok")
PY
```

Các test này kiểm tra:

- Công thức tensor khớp reference PyTorch.
- Rounding half-tie khớp C/C++ `roundf`.
- Forward/backward của `LlamaCppFakeQuantLinear`.
- STE gradient không bị đứt.
- Patch/unpatch giữ nguyên `state_dict` keys và values.

### Q8_0 alignment với gguf-py

Script alignment đối chiếu `fake_quant_q8_0` với implementation `Q8_0` trong `prism-llama-cpp/gguf-py`:

```bash
PYTHONPATH=src python scripts/check_llama_cpp_q8_0_alignment.py \
  --prism-llama-cpp /path/to/prism-llama-cpp
```

Điều kiện pass:

```text
summary: max_error=0 mismatches=0
```

Nếu có bất kỳ mismatch nào, không được xem `Q8_0` fake quant là tương thích công thức.

### Q8_0 Triton validation

Trên máy có CUDA/Triton:

```bash
PYTHONPATH=src python scripts/check_llama_cpp_q8_0_triton.py --benchmark
```

Trong môi trường CPU-only, có thể kiểm tra script không phá workflow bằng:

```bash
PYTHONPATH=src python scripts/check_llama_cpp_q8_0_triton.py --allow-missing-cuda
```

Điều kiện pass trên GPU vẫn là:

```text
summary: max_error=0 mismatches=0
```

## BitNet / 1.58-bit fine-tuning

Đường BitNet gốc của upstream vẫn được giữ lại. Ví dụ fine-tune từ checkpoint pre-quantized:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from onebitllms import replace_linear_with_bitnet_linear

model_id = "tiiuae/Falcon-E-1B-Base"
revision = "prequantized"

tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    revision=revision,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

model = replace_linear_with_bitnet_linear(model)
```

Sau khi train xong, quantize checkpoint về 1-bit:

```python
from onebitllms import quantize_to_1bit

quantize_to_1bit(
    "model-output-dir",
    "quantized-model-output-dir",
)
```

Có thể revert checkpoint BitNet về BF16:

```python
from onebitllms import convert_to_bf16

convert_to_bf16(
    "quantized-model-output-dir",
    "bf16-model-output-dir",
)
```

Inference BitNet nên chạy bằng [`bitnet.cpp`](https://github.com/microsoft/BitNet) hoặc integration phù hợp trong `transformers`.

## Triton kernels

Fork này vẫn expose các Triton kernels từ upstream:

```python
from onebitllms import activation_quant_triton, weight_quant_triton
```

Ý nghĩa:

- `weight_quant_triton`: ternary BitNet weight fake quantization.
- `activation_quant_triton`: row-wise int8 activation fake quantization cho đường BitNet.

Các kernel này không phải GGUF packing kernels. Nếu sau này tối ưu QAT trên GPU cho các kiểu `llama.cpp`, hướng đúng là thêm kernel Triton sau khi công thức PyTorch/reference đã được chứng minh khớp `ggml`.

## Cấu trúc repo

```text
src/onebitllms/
  kernels/
    llama_cpp_quant.py      # fake quantizers Q1/Q2/Q4/Q8 theo ggml
    activation_quant.py     # Triton activation kernel cho BitNet
    weight_quant.py         # Triton weight kernel cho BitNet
  layers/
    llama_cpp.py            # LlamaCppFakeQuantLinear
    bitnet.py               # BitNetLinear
  utils/
    monkey_patching.py      # patch/unpatch Linear layers
    quantization_utils.py   # BitNet checkpoint utilities

tests/
  test_llama_cpp_fake_quant.py
  test_kernels.py

docs/
  llama_cpp_q8_0_qat.md
  testing_llama_cpp_fake_quant.md

scripts/
  check_llama_cpp_q8_0_alignment.py
  check_llama_cpp_q8_0_triton.py
```

## Tài liệu chi tiết

- [Q8_0 QAT tương thích llama.cpp](docs/llama_cpp_q8_0_qat.md)
- [Kế hoạch kiểm thử llama.cpp fake quant](docs/testing_llama_cpp_fake_quant.md)

## Giới hạn cần nhớ

- Fake quant đúng công thức không đảm bảo checkpoint QAT sẽ tốt hơn PTQ-only. Cần eval chất lượng riêng.
- `activation_quant="Q8_0"` không có nghĩa activation được lưu trong GGUF.
- `Q8_1` và `Q8_K` không nên được quảng bá như target export GGUF thông thường trong fork hiện tại.
- Triton fast path hiện mới có cho `Q8_0`; các kiểu `Q4_0/Q4_1` vẫn dùng PyTorch reference.
- Trước khi export HF checkpoint sang GGUF, phải unpatch wrapper fake quant về `nn.Linear`.
- Nếu target là một kiểu quantize khác của llama.cpp, fake quant trong QAT nên khớp đúng công thức của target đó.

## FAQ

### Fork này khác upstream `tiiuae/onebitllms` ở đâu?

Fork này giữ đường BitNet của upstream và thêm đường nghiên cứu QAT tương thích `llama.cpp` / `prism-llama-cpp`.

### Có thể dùng fork này để inference trực tiếp không?

Không. Fork này phục vụ training/fine-tuning/fake quant. Inference nên chạy bằng `llama.cpp`, `prism-llama-cpp`, `bitnet.cpp`, hoặc backend inference phù hợp.

### QAT xong có cần `llama-quantize` nữa không?

Có. Fake quant trong QAT chỉ giúp model thấy nhiễu quantization trong training. Sau khi lưu checkpoint HF, vẫn cần convert sang GGUF và chạy `llama-quantize` để tạo file quantized thật.

### Nên chọn `Q8_0` hay `Q8_1` cho QAT 8-bit?

Nên chọn `Q8_0` nếu mục tiêu là desktop inference bằng GGUF, vì `Q8_0` là mode quantize model thật trong `llama-quantize`. `Q8_1` là format runtime/vector-dot trong `ggml`, không phải đường export thông thường.

### LoRA có được hỗ trợ không?

LoRA không phải mục tiêu chính của package hiện tại. Có thể nghiên cứu thêm, nhưng cần kiểm tra kỹ tương tác giữa LoRA adapter, patch/unpatch layer, và export checkpoint.

## Citation

Fork này dựa trên upstream `onebitllms` của Falcon-LLM Team. Nếu dùng cho nghiên cứu, hãy cite upstream và các công trình nền tảng về BitNet:

```bibtex
@misc{tiionebitllms,
    title = {Falcon-E, a series of powerful, universal and fine-tunable 1.58bit language models.},
    author = {Falcon-LLM Team},
    month = {May},
    url = {https://github.com/tiiuae/onebitllms},
    year = {2025}
}
```

```bibtex
@misc{wang2025bitnetcppefficientedgeinference,
    title = {Bitnet.cpp: Efficient Edge Inference for Ternary LLMs},
    author = {Jinheng Wang and Hansong Zhou and Ting Song and Shijie Cao and Yan Xia and Ting Cao and Jianyu Wei and Shuming Ma and Hongyu Wang and Furu Wei},
    year = {2025},
    eprint = {2502.11880},
    archivePrefix = {arXiv},
    primaryClass = {cs.LG},
    url = {https://arxiv.org/abs/2502.11880}
}
```

```bibtex
@misc{mekkouri2024bitllm,
    title = {1.58-Bit LLM: A New Era of Extreme Quantization},
    author = {Mohamed Mekkouri and Marc Sun and Leandro von Werra and Thomas Wolf},
    year = {2024}
}
```

```bibtex
@misc{ma2024era1bitllmslarge,
    title = {The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits},
    author = {Shuming Ma and Hongyu Wang and Lingxiao Ma and Lei Wang and Wenhui Wang and Shaohan Huang and Li Dong and Ruiping Wang and Jilong Xue and Furu Wei},
    year = {2024},
    eprint = {2402.17764},
    archivePrefix = {arXiv},
    primaryClass = {cs.CL},
    url = {https://arxiv.org/abs/2402.17764}
}
```

```bibtex
@misc{wang2023bitnetscaling1bittransformers,
    title = {BitNet: Scaling 1-bit Transformers for Large Language Models},
    author = {Hongyu Wang and Shuming Ma and Li Dong and Shaohan Huang and Huaijie Wang and Lingxiao Ma and Fan Yang and Ruiping Wang and Yi Wu and Furu Wei},
    year = {2023},
    eprint = {2310.11453},
    archivePrefix = {arXiv},
    primaryClass = {cs.CL},
    url = {https://arxiv.org/abs/2310.11453}
}
```

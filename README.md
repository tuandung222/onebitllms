<p align="center">
  <img src="assets/onebitllms-logo.png" alt="onebitllms logo" width="400">
</p>

# onebitllms fork: BitNet fine-tuning and llama.cpp-compatible QAT

This fork extends `onebitllms` in two directions:

1. It keeps the upstream BitNet / 1.58-bit fine-tuning path.
2. It adds fake quantizers that mirror `llama.cpp` / `prism-llama-cpp` quantization formulas for QAT research before GGUF export and PTQ with `llama-quantize`.

The intended workflow is:

```text
QAT in PyTorch
-> save a standard Hugging Face checkpoint
-> convert to F16/BF16 GGUF
-> PTQ with llama.cpp / prism-llama-cpp
-> inference on llama.cpp-compatible runtimes
```

The fake-quant layers in this fork are not inference kernels and do not pack GGUF bytes. They keep trainable weights in floating point, inject quantize-dequantize noise in the forward pass, and use a straight-through estimator for backward.

## Features

| Area | Status | Notes |
| --- | --- | --- |
| BitNet / 1.58-bit fine-tuning | Supported | Inherited from upstream `onebitllms`, using `BitNetLinear` and Triton kernels |
| Triton quant kernels | Supported | Used by the BitNet training path, not GGUF packing |
| llama.cpp fake-quant weights | Supported | `Q1_0`, `Q2_0`, `Q4_0`, `Q4_1`, `Q8_0`, `Q8_1` |
| llama.cpp fake-quant activations | Supported | `activation_quant="Q8_0"` as temporary QAT noise |
| Triton fast path for llama.cpp QAT | Experimental | Currently available for `Q8_0` CUDA only; the default backend is PyTorch |
| Patch/unpatch `nn.Linear` | Supported | Train with wrappers, then export a standard checkpoint |
| Direct GGUF export | Not supported | Use llama.cpp converters and `llama-quantize` |
| Inference kernel | Not supported | Run inference with llama.cpp / prism-llama-cpp / bitnet.cpp |

## Installation

Install this fork from source:

```bash
git clone https://github.com/tuandung222/onebitllms.git
cd onebitllms
pip install -e .
```

To run tests:

```bash
pip install -e ".[test]"
```

Main requirements:

- Python >= 3.9.
- PyTorch.
- `transformers`, `accelerate`, `safetensors`, `huggingface_hub`.
- NVIDIA GPU + Triton if you use the BitNet CUDA kernels.
- A local `prism-llama-cpp` or `llama.cpp` checkout if you want to run alignment tests, convert GGUF files, quantize, or run inference.

## Quick Start: llama.cpp-Compatible QAT

### 1. Fake-quantize a tensor directly

```python
import torch
from onebitllms import fake_quant_q4_0, fake_quant_q4_1, fake_quant_q8_0

weight = torch.randn(128, 256)

w_q4_0 = fake_quant_q4_0(weight)
w_q4_1 = fake_quant_q4_1(weight)
w_q8_0 = fake_quant_q8_0(weight)
```

### 2. Replace `nn.Linear` with fake-quant wrappers

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
    quant_type="Q4_0",
)
```

The helper replaces block-size-compatible `nn.Linear` modules. It skips `lm_head` by default.

For QAT targeting `llama-quantize ... Q4_0`, use `quant_type="Q4_0"` and create the optimizer after patching the model.

### 3. Optionally add Q8_0 activation fake quantization

```python
model = replace_linear_with_llama_cpp_fake_quant_linear(
    model,
    quant_type="Q4_0",
    activation_quant="Q8_0",
)
```

`activation_quant="Q8_0"` only injects temporary training noise. GGUF does not store activations in quantized form.

In llama.cpp CPU type traits, `Q4_0` dot products use `vec_dot_type = GGML_TYPE_Q8_0`, so activation fake quantization can be useful for ablations. It should not be enabled blindly; measure it against your target model and backend.

### 4. Unpatch before saving a checkpoint

```python
from onebitllms import replace_llama_cpp_fake_quant_linear_with_linear

model = replace_llama_cpp_fake_quant_linear_with_linear(model)
model.save_pretrained("output-hf-checkpoint")
```

Do not save a Hugging Face checkpoint while the model still contains `LlamaCppFakeQuantLinear`; llama.cpp converters expect standard module and weight layouts.

### 5. Use the Q8_0 Triton fast path

The default backend is `torch` for portability and deterministic behavior. On CUDA with Triton installed, you can use the experimental `Q8_0` fast path:

```python
model = replace_linear_with_llama_cpp_fake_quant_linear(
    model,
    quant_type="Q8_0",
    activation_quant="Q8_0",
    backend="auto",
)
```

Backends:

- `backend="torch"`: always use the PyTorch reference implementation.
- `backend="auto"`: use Triton for supported CUDA `Q8_0` tensors and fall back to PyTorch when unsupported.
- `backend="triton"`: require Triton; currently supports `Q8_0` only.

Before using `backend="triton"` in real training, validate it on the target GPU:

```bash
PYTHONPATH=src python scripts/check_llama_cpp_q8_0_triton.py --benchmark
```

## Supported llama.cpp Fake-Quant Types

| Type | Block size | Formula | Export target? |
| --- | ---: | --- | --- |
| `Q1_0` | 128 | `prism-llama-cpp` `quantize_row_q1_0_ref` | Use only if your fork/export path supports it |
| `Q2_0` | 128 | `prism-llama-cpp` `quantize_row_q2_0_ref` | Use only if your fork/export path supports it |
| `Q4_0` | 32 | llama.cpp `quantize_row_q4_0_ref` | Yes, if you will PTQ with `llama-quantize Q4_0` |
| `Q4_1` | 32 | llama.cpp `quantize_row_q4_1_ref` | Yes, if you will PTQ with `llama-quantize Q4_1` |
| `Q8_0` | 32 | llama.cpp `quantize_row_q8_0_ref` / `dequantize_row_q8_0` | Yes, the primary 8-bit target |
| `Q8_1` | 32 | ggml `quantize_row_q8_1_ref` | Not recommended as a normal export target |

`Q8_0` is the recommended 8-bit target. In `prism-llama-cpp`, `Q8_0` is exposed directly in `tools/quantize/quantize.cpp` as a model quantization mode. `Q8_1` exists in `ggml`, but it is mainly a runtime/vector-dot format, not a normal `llama-quantize` export mode. `Q8_K` is also an internal/K-quant format and is not exposed as a QAT target in this fork.

## QAT -> GGUF -> Inference Workflow

Recommended flow:

```text
1. Load a Hugging Face model.
2. Patch selected Linear layers with LlamaCppFakeQuantLinear.
3. Fine-tune / run QAT in PyTorch.
4. Unpatch back to nn.Linear.
5. Save a Hugging Face checkpoint.
6. Convert the checkpoint to F16/BF16 GGUF with llama.cpp.
7. Run llama-quantize with the target that matches the fake quantizer, for example Q4_0.
8. Run llama-cli for an inference smoke test.
9. Run a fixed evaluation to compare QAT+PTQ against PTQ-only.
```

Example PTQ command after you have an F16/BF16 GGUF:

```bash
/path/to/prism-llama-cpp/build/bin/llama-quantize \
  model-f16.gguf \
  model-q4_0.gguf \
  Q4_0
```

Inference smoke test:

```bash
/path/to/prism-llama-cpp/build/bin/llama-cli \
  -m model-q4_0.gguf \
  -p "Explain quantization-aware training in one paragraph." \
  -n 64
```

## Testing and Validation

This fork has two main test layers for llama.cpp-compatible fake quantization.

### Unit tests

If `pytest` is available:

```bash
PYTHONPATH=src python -m pytest tests/test_llama_cpp_fake_quant.py -q
```

If your environment does not have `pytest`:

```bash
PYTHONPATH=src python - <<'PY'
import importlib.util
from pathlib import Path

path = Path("tests/test_llama_cpp_fake_quant.py")
spec = importlib.util.spec_from_file_location("test_llama_cpp_fake_quant", path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

for name in sorted(n for n in dir(mod) if n.startswith("test_")):
    getattr(mod, name)()
    print(f"{name}: ok")
PY
```

These tests check:

- Tensor formulas against local PyTorch references.
- Half-tie rounding against C/C++ `roundf` where applicable.
- `LlamaCppFakeQuantLinear` forward/backward.
- STE gradient flow.
- Patch/unpatch preserving `state_dict` keys and values.

### Q4_0/Q8_0 alignment with gguf-py

The alignment scripts compare fake quantizers against `prism-llama-cpp/gguf-py`:

```bash
PYTHONPATH=src python scripts/check_llama_cpp_q4_0_alignment.py \
  --prism-llama-cpp /path/to/prism-llama-cpp

PYTHONPATH=src python scripts/check_llama_cpp_q8_0_alignment.py \
  --prism-llama-cpp /path/to/prism-llama-cpp
```

You can also set an environment variable:

```bash
PRISM_LLAMA_CPP=/path/to/prism-llama-cpp \
PYTHONPATH=src python scripts/check_llama_cpp_q4_0_alignment.py
```

Passing condition:

```text
summary: max_error=0 mismatches=0
```

If there is any mismatch, do not treat the corresponding fake quantizer as formula-compatible.

### Q8_0 Triton validation

On CUDA with Triton:

```bash
PYTHONPATH=src python scripts/check_llama_cpp_q8_0_triton.py --benchmark
```

On CPU-only environments, you can verify that the script does not break the workflow:

```bash
PYTHONPATH=src python scripts/check_llama_cpp_q8_0_triton.py --allow-missing-cuda
```

Passing condition on GPU:

```text
summary: max_error=0 mismatches=0
```

## BitNet / 1.58-Bit Fine-Tuning

The upstream BitNet path is still available. Example fine-tuning from a pre-quantized checkpoint:

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

After training, quantize the checkpoint back to 1-bit:

```python
from onebitllms import quantize_to_1bit

quantize_to_1bit(
    "model-output-dir",
    "quantized-model-output-dir",
)
```

You can also revert a BitNet checkpoint back to BF16:

```python
from onebitllms import convert_to_bf16

convert_to_bf16(
    "quantized-model-output-dir",
    "bf16-model-output-dir",
)
```

Run BitNet inference with [`bitnet.cpp`](https://github.com/microsoft/BitNet) or another compatible inference backend.

## Triton Kernels

This fork still exposes the upstream Triton kernels:

```python
from onebitllms import activation_quant_triton, weight_quant_triton
```

Meaning:

- `weight_quant_triton`: ternary BitNet weight fake quantization.
- `activation_quant_triton`: row-wise int8 activation fake quantization for the BitNet path.

These kernels are not GGUF packing kernels. If future work optimizes llama.cpp-compatible QAT on GPU, add Triton kernels only after the PyTorch/reference formulas are proven to match `ggml`.

## Repository Layout

```text
src/onebitllms/
  kernels/
    llama_cpp_quant.py      # fake quantizers Q1/Q2/Q4/Q8 following ggml formulas
    activation_quant.py     # Triton activation kernel for BitNet
    weight_quant.py         # Triton weight kernel for BitNet
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
  llama_cpp_q4_0_qat.md
  llama_cpp_q8_0_qat.md
  testing_llama_cpp_fake_quant.md

scripts/
  check_llama_cpp_q4_0_alignment.py
  check_llama_cpp_q8_0_alignment.py
  check_llama_cpp_q8_0_triton.py
```

## Detailed Docs

- [Q4_0 QAT compatible with llama.cpp](docs/llama_cpp_q4_0_qat.md)
- [Q8_0 QAT compatible with llama.cpp](docs/llama_cpp_q8_0_qat.md)
- [llama.cpp fake-quant testing plan](docs/testing_llama_cpp_fake_quant.md)

## Limitations

- Formula-correct fake quantization does not guarantee that a QAT checkpoint will outperform PTQ-only. Run quality evaluation.
- `activation_quant="Q8_0"` does not mean activations are stored in GGUF.
- `Q8_1` and `Q8_K` should not be promoted as normal GGUF export targets in this fork.
- The Triton fast path currently supports only `Q8_0`; `Q4_0` and `Q4_1` use the PyTorch reference path.
- Before exporting a Hugging Face checkpoint to GGUF, unpatch fake-quant wrappers back to `nn.Linear`.
- If the target is a different llama.cpp quantization type, the QAT fake quantizer should match that exact target formula.

## FAQ

### How is this fork different from upstream `tiiuae/onebitllms`?

This fork keeps the upstream BitNet path and adds a llama.cpp / prism-llama-cpp-compatible QAT research path.

### Can this fork run inference directly?

No. This fork is for training, fine-tuning, and fake quantization. Run inference with `llama.cpp`, `prism-llama-cpp`, `bitnet.cpp`, or another compatible inference backend.

### Do I still need `llama-quantize` after QAT?

Yes. Fake quantization during QAT only exposes the model to quantization noise during training. After saving the Hugging Face checkpoint, you still need to convert to GGUF and run `llama-quantize` to create the real quantized file.

### Should I use `Q8_0` or `Q8_1` for 8-bit QAT?

Use `Q8_0` if the goal is GGUF desktop inference, because `Q8_0` is a real model quantization mode in `llama-quantize`. `Q8_1` is a runtime/vector-dot format in `ggml`, not the normal export path.

### Is LoRA supported?

LoRA is not the primary target of this package today. It can be researched, but you must carefully validate interactions between LoRA adapters, layer patch/unpatch, and checkpoint export.

## Citation

This fork is based on upstream `onebitllms` from the Falcon-LLM Team. If you use it for research, cite upstream and the underlying BitNet work:

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

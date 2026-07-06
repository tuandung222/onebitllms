# llama.cpp-compatible Q4_0 QAT

This document describes how `onebitllms` simulates llama.cpp `Q4_0` fake quantization for QAT before exporting a checkpoint to GGUF and running PTQ with `llama.cpp` or `prism-llama-cpp`.

## Short conclusion

`Q4_0` is a direct 4-bit weight quantization target supported by `llama-quantize`, so it is the recommended target when the intended lifecycle is:

```text
QAT in PyTorch
-> save an HF checkpoint in FP16/BF16/FP32
-> convert the checkpoint to GGUF
-> llama-quantize ... Q4_0
-> inference with llama.cpp / prism-llama-cpp
```

The fake quantizer does not pack GGUF bytes. The wrapper keeps trainable floating-point weights, injects llama.cpp-style `Q4_0` quantization noise during the forward pass, uses STE for backward, and must be unpatched back to `nn.Linear` before saving or exporting.

## llama.cpp Q4_0 formula

According to `ggml/src/ggml-common.h`, one `Q4_0` block stores 32 values:

```text
block_q4_0:
  d:  fp16 scale
  qs: uint8[16], with two 4-bit codes packed into each byte
```

According to `ggml/src/ggml-quants.c`, `quantize_row_q4_0_ref` processes each 32-value block as:

```text
signed_absmax = first element with the largest absolute value in the block
d_raw = signed_absmax / -8
id = 1 / d_raw if d_raw != 0 else 0
d = fp16(d_raw)
q[i] = clamp(trunc(x[i] * id + 8.5), 0, 15)
x_dequant[i] = (q[i] - 8) * fp16_to_fp32(d)
```

Important details:

- The block size is 32.
- The scale `d` is stored as fp16, so fake quantization must model fp16 scale rounding.
- `signed_absmax` keeps the sign of the largest-magnitude value. On ties, llama.cpp keeps the first matching element.
- The rounding rule is `trunc(x * id + 8.5)`, not generic `round`.
- `Q4_0` is a GGUF weight quantization target.

## Runtime activation behavior in llama.cpp

In ggml CPU type traits, `Q4_0` uses:

```text
vec_dot      = ggml_vec_dot_q4_0_q8_0
vec_dot_type = GGML_TYPE_Q8_0
```

This means the runtime/input side of a CPU dot product can be temporarily converted to `Q8_0` when multiplying against `Q4_0` weights. This is a temporary kernel format, not an activation tensor stored in GGUF.

For QAT, both of these configurations are useful ablations:

```python
# Simulate Q4_0 PTQ noise on weights.
replace_linear_with_llama_cpp_fake_quant_linear(model, quant_type="Q4_0")

# Also simulate Q8_0 runtime/input quantization noise used by the dot-product path.
replace_linear_with_llama_cpp_fake_quant_linear(
    model,
    quant_type="Q4_0",
    activation_quant="Q8_0",
)
```

Activation fake quantization is not enabled by default because its quality impact depends on the model, backend, and target evaluation path.

## QAT usage

```python
from onebitllms import replace_linear_with_llama_cpp_fake_quant_linear

model = replace_linear_with_llama_cpp_fake_quant_linear(
    model,
    quant_type="Q4_0",
)
```

Create the optimizer after patching:

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
```

Before saving a checkpoint for GGUF export:

```python
from onebitllms import replace_llama_cpp_fake_quant_linear_with_linear

model = replace_llama_cpp_fake_quant_linear_with_linear(model)
model.save_pretrained(output_dir)
```

After converting to F16/BF16 GGUF, run PTQ with the same target:

```bash
/path/to/prism-llama-cpp/build/bin/llama-quantize \
  model-f16.gguf \
  model-q4_0.gguf \
  Q4_0
```

## Validation

Run the unit tests:

```bash
PYTHONPATH=src python -m pytest tests/test_llama_cpp_fake_quant.py -q
```

If `pytest` is not installed, call the test functions directly as described in the README.

Run alignment against Prism's `gguf-py`:

```bash
PYTHONPATH=src python scripts/check_llama_cpp_q4_0_alignment.py \
  --prism-llama-cpp /path/to/prism-llama-cpp
```

or:

```bash
PRISM_LLAMA_CPP=/path/to/prism-llama-cpp \
PYTHONPATH=src python scripts/check_llama_cpp_q4_0_alignment.py
```

The expected result is:

```text
summary: max_error=0 mismatches=0
```

Before using this with a real checkpoint, still run the full export smoke test:

```text
QAT Q4_0 -> unpatch -> save HF -> convert GGUF F16/BF16 -> llama-quantize Q4_0 -> llama-cli smoke test
```

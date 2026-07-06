# llama.cpp fake quant testing plan

The goal of this plan is to ensure that `onebitllms` fake quantizers are not only executable, but also close enough to the `prism-llama-cpp` formulas to be used for QAT before PTQ/export with llama.cpp.

## Scope

The tested components are:

- Fake quant kernels: `Q1_0`, `Q2_0`, `Q4_0`, `Q4_1`, `Q8_0`, and `Q8_1`.
- Activation fake quantization: `activation_quant="Q8_0"`.
- Wrapper layer: `LlamaCppFakeQuantLinear`.
- Model surgery: patching `nn.Linear` to the fake quant wrapper and unpatching back to `nn.Linear`.
- Alignment with `prism-llama-cpp/gguf-py`, with priority on `Q4_0` and `Q8_0` because they are direct `llama-quantize` export targets.
- Triton fast path for `Q8_0` on CUDA.

## Test pyramid

### L0: Static checks

Run:

```bash
PYTHONPATH=src python -m py_compile \
  src/onebitllms/kernels/llama_cpp_quant.py \
  src/onebitllms/kernels/llama_cpp_quant_triton.py \
  src/onebitllms/layers/llama_cpp.py \
  src/onebitllms/utils/monkey_patching.py \
  tests/test_llama_cpp_fake_quant.py \
  scripts/check_llama_cpp_q4_0_alignment.py \
  scripts/check_llama_cpp_q8_0_alignment.py \
  scripts/check_llama_cpp_q8_0_triton.py

git diff --check
```

Passing criteria:

- No syntax errors.
- No trailing whitespace or patch format errors.

### L1: Formula and layer unit tests

If the environment has `pytest`:

```bash
PYTHONPATH=src python -m pytest tests/test_llama_cpp_fake_quant.py -q
```

If `pytest` is not installed, run the test functions directly:

```bash
PYTHONPATH=src python - <<'PY'
import tests.test_llama_cpp_fake_quant as t

for name in sorted(n for n in dir(t) if n.startswith("test_")):
    getattr(t, name)()
    print(f"{name}: ok")
PY
```

Passing criteria:

- PyTorch references and real kernels match with `torch.equal`.
- Tie and rounding cases match the C/C++ behavior.
- `LlamaCppFakeQuantLinear` forward and backward run correctly.
- STE gradients remain connected.
- Patch/unpatch preserves `state_dict` keys and values.

### L2: Q4_0/Q8_0 alignment with gguf-py

Run:

```bash
PYTHONPATH=src python scripts/check_llama_cpp_q4_0_alignment.py \
  --prism-llama-cpp /path/to/prism-llama-cpp

PYTHONPATH=src python scripts/check_llama_cpp_q8_0_alignment.py \
  --prism-llama-cpp /path/to/prism-llama-cpp
```

Passing criteria:

```text
summary: max_error=0 mismatches=0
```

If `max_error > 0` or `mismatches > 0`, the corresponding implementation should not be considered llama.cpp-compatible.

### L3a: Q8_0 Triton fast path

On a CUDA/Triton machine, run:

```bash
PYTHONPATH=src python scripts/check_llama_cpp_q8_0_triton.py --benchmark
```

In a CPU-only environment, use:

```bash
PYTHONPATH=src python scripts/check_llama_cpp_q8_0_triton.py --allow-missing-cuda
```

Passing criteria on GPU:

```text
summary: max_error=0 mismatches=0
```

If `max_error > 0` or `mismatches > 0`, do not enable `backend="triton"` for real training. Without GPU validation, use `backend="torch"` or `backend="auto"` with PyTorch fallback.

### L3b: QAT layer smoke test

This level verifies that the wrapper can run in a small training loop.

Checklist:

1. Create a toy model with a few `nn.Linear` layers.
2. Patch the target configuration, at minimum `quant_type="Q4_0"` and `quant_type="Q8_0"`.
3. Run a few optimizer steps.
4. Check that the loss and gradients are finite.
5. Unpatch back to `nn.Linear`.
6. Check that `state_dict` keys are unchanged.

These tests are included in `tests/test_llama_cpp_fake_quant.py`. For a real model rollout, add a smoke test with a small Hugging Face model.

### L4: End-to-end GGUF export

This test is required before claiming that a QAT checkpoint is usable with llama.cpp.

Workflow:

```text
1. Load a small HF model or QAT checkpoint.
2. Patch selected linear layers with the target under test, for example quant_type="Q4_0".
3. Run a short train/smoke fine-tune.
4. Unpatch back to nn.Linear.
5. Save the HF checkpoint.
6. Convert the HF checkpoint to F16/BF16 GGUF with llama.cpp.
7. Run llama-quantize input.gguf output-q4_0.gguf Q4_0.
8. Run llama-cli with a fixed prompt.
```

Passing criteria:

- The converter succeeds.
- `llama-quantize ... Q4_0` succeeds.
- `llama-cli` loads the Q4_0 model.
- The fixed prompt generates non-empty output without NaN, inf, or logit crashes.

### L5: Quality regression

This level does not prove formula correctness, but it is needed to decide whether QAT improves the target model.

Recommended comparisons:

- FP16/BF16 baseline.
- PTQ `Q4_0` without QAT.
- QAT fake `Q4_0`, then PTQ `Q4_0`.
- QAT fake `Q4_0` plus activation fake `Q8_0`, then PTQ `Q4_0`.
- PTQ `Q8_0` without QAT.
- QAT fake `Q8_0`, then PTQ `Q8_0`.
- QAT fake `Q8_0` plus activation fake `Q8_0`, then PTQ `Q8_0`.

Record the exact `onebitllms` commit, `prism-llama-cpp` commit, model checkpoint, seed, and command.

The passing threshold depends on the quality target, but QAT should at least avoid becoming worse than PTQ-only on a small fixed eval set without a clear explanation.

## Q4_0/Q8_0 test matrix

| Case group | Purpose |
| --- | --- |
| Random tensors | Catch general formula mismatches |
| All zeros | Catch divide-by-zero behavior |
| Constant tensors | Catch small/even scale issues |
| Large range | Catch saturation and signed int8 range issues |
| Half ties | Catch rounding differences |
| Multi-block rows | Catch reshape/block boundary issues |
| Layer forward/backward | Catch `nn.Module` integration issues |
| Patch/unpatch | Catch export lifecycle issues |
| gguf-py exact alignment | Catch differences from llama.cpp reference implementation |
| Triton exact alignment | Catch CUDA fast path differences from the PyTorch reference |

## Claims to avoid

- Passing `Q4_0`/`Q8_0` formula tests does not mean every QAT checkpoint will beat PTQ-only.
- Activation fake quantization with `Q8_0` does not mean activations are stored in GGUF.
- `Q8_1` is not a standard `llama-quantize` target in the current fork.
- `Q8_K` is not exposed in `onebitllms` because it is not the main export target in this fork.
- The Triton fast path currently exists for `Q8_0`; `Q4_0` and `Q4_1` still use the PyTorch reference path.

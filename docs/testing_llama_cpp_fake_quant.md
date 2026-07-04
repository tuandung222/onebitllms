# Kế hoạch kiểm thử llama.cpp fake quant

Mục tiêu của kế hoạch này là đảm bảo các fake quantizer trong `onebitllms` không chỉ chạy được, mà còn bám đúng công thức của `prism-llama-cpp` đủ để dùng trong QAT trước khi PTQ/export bằng llama.cpp.

## Phạm vi

Các thành phần được kiểm thử:

- Kernel fake quant: `Q1_0`, `Q2_0`, `Q4_0`, `Q4_1`, `Q8_0`, `Q8_1`.
- Activation fake quant: `activation_quant="Q8_0"`.
- Wrapper layer: `LlamaCppFakeQuantLinear`.
- Model surgery: patch `nn.Linear` sang fake quant wrapper và unpatch về `nn.Linear`.
- Alignment với `prism-llama-cpp/gguf-py`, ưu tiên `Q8_0` vì đây là target export 8-bit thật của `llama-quantize`.

## Test pyramid

### L0: Static checks

Chạy:

```bash
PYTHONPATH=src python -m py_compile \
  src/onebitllms/kernels/llama_cpp_quant.py \
  src/onebitllms/layers/llama_cpp.py \
  src/onebitllms/utils/monkey_patching.py \
  tests/test_llama_cpp_fake_quant.py

git diff --check
```

Điều kiện pass:

- Không có syntax error.
- Không có trailing whitespace hoặc lỗi patch format.

### L1: Unit tests công thức và layer

Nếu env có `pytest`:

```bash
PYTHONPATH=src python -m pytest tests/test_llama_cpp_fake_quant.py -q
```

Nếu env chưa có `pytest`, chạy trực tiếp:

```bash
PYTHONPATH=src python - <<'PY'
import tests.test_llama_cpp_fake_quant as t

for name in sorted(n for n in dir(t) if n.startswith("test_")):
    getattr(t, name)()
    print(f"{name}: ok")
PY
```

Điều kiện pass:

- Reference PyTorch và kernel thật khớp bằng `torch.equal`.
- Case tie rounding khớp C/C++ `roundf`.
- Forward/backward của `LlamaCppFakeQuantLinear` chạy được.
- STE gradient không bị đứt.
- Patch/unpatch không đổi `state_dict` keys/values.

### L2: Alignment Q8_0 với gguf-py

Chạy script:

```bash
PYTHONPATH=src python scripts/check_llama_cpp_q8_0_alignment.py \
  --prism-llama-cpp /path/to/prism-llama-cpp
```

Trong workspace hiện tại:

```bash
PYTHONPATH=src /opt/anaconda3/envs/llm/bin/python \
  scripts/check_llama_cpp_q8_0_alignment.py \
  --prism-llama-cpp /Users/admin/TuanDung/research-workspace/prism-llama-cpp
```

Điều kiện pass:

```text
summary: max_error=0 mismatches=0
```

Nếu `max_error > 0` hoặc `mismatches > 0`, không được xem implementation `Q8_0` là tương thích.

### L3: Smoke test QAT layer

Mục tiêu là đảm bảo wrapper dùng được trong train loop nhỏ.

Checklist:

1. Tạo model toy có vài `nn.Linear`.
2. Patch với `quant_type="Q8_0"`.
3. Chạy vài bước optimizer.
4. Kiểm tra loss finite, gradient finite.
5. Unpatch về `nn.Linear`.
6. Kiểm tra `state_dict` keys không đổi.

Các test này đã nằm trong `tests/test_llama_cpp_fake_quant.py`. Khi mở rộng sang model thật, cần thêm smoke test với một model nhỏ từ Hugging Face.

### L4: End-to-end GGUF export

Đây là test bắt buộc trước khi tuyên bố một checkpoint QAT dùng được với llama.cpp.

Quy trình:

```text
1. Load model HF nhỏ hoặc checkpoint QAT.
2. Patch selected linear layers với quant_type="Q8_0".
3. Chạy train/smoke fine-tune ngắn.
4. Unpatch về nn.Linear.
5. Save HF checkpoint.
6. Convert HF checkpoint sang F16/BF16 GGUF bằng script llama.cpp.
7. Chạy llama-quantize input.gguf output-q8_0.gguf Q8_0.
8. Chạy llama-cli với prompt cố định.
```

Điều kiện pass:

- Converter không lỗi.
- `llama-quantize ... Q8_0` không lỗi.
- `llama-cli` load được model Q8_0.
- Prompt cố định sinh token không rỗng và không có NaN/inf/logit crash.

### L5: Quality regression

Lớp này không chứng minh công thức đúng, nhưng cần để biết QAT có đáng dùng không.

Khuyến nghị:

- Dùng một tập eval nhỏ cố định, ví dụ perplexity trên một subset WikiText/C4 hoặc benchmark nội bộ.
- So sánh các cấu hình:
  - FP16/BF16 baseline.
  - PTQ Q8_0 không QAT.
  - QAT fake `Q8_0` rồi PTQ `Q8_0`.
  - QAT fake `Q8_0` + activation fake `Q8_0` rồi PTQ `Q8_0`.
- Ghi lại exact commit của `onebitllms`, commit của `prism-llama-cpp`, model checkpoint, seed và command.

Điều kiện pass phụ thuộc mục tiêu chất lượng, nhưng tối thiểu QAT không được làm model tệ hơn PTQ-only trên eval nhỏ mà không có lý do rõ ràng.

## Ma trận kiểm thử Q8_0

| Nhóm case | Mục đích |
| --- | --- |
| Random tensors | Bắt lỗi công thức chung |
| All zeros | Bắt lỗi chia 0 |
| Constant tensors | Bắt lỗi scale nhỏ/đều |
| Large range | Bắt lỗi saturation và signed int8 range |
| Half ties | Bắt lỗi rounding khác `roundf` |
| Multi-block rows | Bắt lỗi reshape/block boundary |
| Layer forward/backward | Bắt lỗi integration trong `nn.Module` |
| Patch/unpatch | Bắt lỗi export lifecycle |
| gguf-py exact alignment | Bắt lỗi lệch với llama.cpp reference implementation |

## Những điều không được tuyên bố quá mức

- Pass công thức `Q8_0` không đồng nghĩa mọi checkpoint QAT sẽ tốt hơn PTQ-only.
- Activation fake quant `Q8_0` không có nghĩa activation được lưu trong GGUF.
- `Q8_1` không phải target `llama-quantize` thông thường trong fork hiện tại.
- `Q8_K` chưa được expose trong `onebitllms` vì chưa phải target export chính của fork này.


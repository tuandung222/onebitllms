# Q8_0 QAT tương thích llama.cpp

Tài liệu này mô tả cách `onebitllms` mô phỏng fake quantization `Q8_0` để phục vụ QAT trước khi export checkpoint sang GGUF và quantize bằng `llama.cpp` / `prism-llama-cpp`.

## Kết luận ngắn

`Q8_0` là target 8-bit nên dùng nếu mục tiêu là:

```text
QAT trong PyTorch
-> lưu checkpoint HF ở FP16/BF16/FP32
-> convert checkpoint sang GGUF
-> llama-quantize ... Q8_0
-> inference bằng llama.cpp / prism-llama-cpp
```

Trong `prism-llama-cpp`, `Q8_0` xuất hiện trực tiếp trong `tools/quantize/quantize.cpp` như một mode quantize model. `Q8_1` và `Q8_K` có tồn tại trong `ggml`, nhưng không phải lựa chọn `llama-quantize` thông thường trong fork này. Vì vậy, QAT 8-bit cho desktop inference nên mặc định dùng `Q8_0`.

## Công thức Q8_0 trong llama.cpp

Theo `ggml/src/ggml-common.h`, một block `Q8_0` có 32 giá trị:

```text
block_q8_0:
  d:  fp16 scale
  qs: int8[32]
```

Theo `ggml/src/ggml-quants.c`, `quantize_row_q8_0_ref` làm theo từng block 32 phần tử:

```text
amax = max(abs(x))
d_raw = amax / 127
id = 1 / d_raw nếu d_raw != 0, ngược lại 0
d = fp16(d_raw)
q[i] = roundf(x[i] * id)
x_dequant[i] = q[i] * fp16_to_fp32(d)
```

Điểm quan trọng:

- Block size là 32.
- Scale `d` được lưu bằng fp16, nên fake quant phải mô phỏng bước fp16 rounding.
- Rounding phải giống C/C++ `roundf`, tức half-away-from-zero, không phải banker rounding mặc định của một số thư viện.
- `Q8_0` là weight quantization target trong GGUF. Activation fake quant `Q8_0` trong `onebitllms` chỉ là nhiễu tạm thời khi train, không có nghĩa activation được lưu vào GGUF.

## Hiện thực trong onebitllms

API chính:

```python
from onebitllms import fake_quant_q8_0

w_q = fake_quant_q8_0(weight)
```

Thay `nn.Linear` bằng wrapper fake quant:

```python
from onebitllms import replace_linear_with_llama_cpp_fake_quant_linear

model = replace_linear_with_llama_cpp_fake_quant_linear(
    model,
    quant_type="Q8_0",
)
```

Nếu muốn thêm nhiễu activation int8 tạm thời trong QAT:

```python
model = replace_linear_with_llama_cpp_fake_quant_linear(
    model,
    quant_type="Q8_0",
    activation_quant="Q8_0",
)
```

Nếu train trên CUDA và đã cài Triton, có thể bật fast path experimental cho `Q8_0`:

```python
model = replace_linear_with_llama_cpp_fake_quant_linear(
    model,
    quant_type="Q8_0",
    activation_quant="Q8_0",
    backend="auto",
)
```

Các backend:

- `backend="torch"`: dùng PyTorch reference, là default.
- `backend="auto"`: dùng Triton cho CUDA `Q8_0` nếu có thể, fallback về PyTorch.
- `backend="triton"`: bắt buộc dùng Triton, hiện chỉ hỗ trợ `Q8_0`.

Triton fast path chỉ là tối ưu tốc độ cho fake quant trong QAT, không phải GGUF packing kernel và không phải inference kernel.

Trước khi lưu checkpoint để export GGUF, phải chuyển wrapper về `nn.Linear`:

```python
from onebitllms import replace_llama_cpp_fake_quant_linear_with_linear

model = replace_llama_cpp_fake_quant_linear_with_linear(model)
model.save_pretrained(output_dir)
```

Lý do: wrapper fake quant dùng weight float trainable và chỉ inject nhiễu trong forward pass. Checkpoint HF nên giữ cấu trúc layer chuẩn để converter của llama.cpp đọc được.

## Q8_0 khác gì Q8_1 và Q8_K?

`Q8_1` trong `ggml` cũng dùng block 32 phần tử và cùng giá trị dequant `q * d`, nhưng block có thêm:

```text
s = fp16(sum(q) * d)
```

Trường `s` phục vụ đường vector-dot/runtime. Trong bảng type của `ggml.c`, `Q8_0` có `to_float` và `from_float_ref`; `Q8_1` có `from_float_ref` nhưng không được expose như mode `llama-quantize` model trong `tools/quantize/quantize.cpp`.

`Q8_K` cũng là format nội bộ/K-quant dùng trong runtime và vector-dot. Fork hiện tại không expose `Q8_K` như một mode quantize model thông thường, nên `onebitllms` chưa xem `Q8_K` là QAT target chính.

## Chiến lược dùng Q8_0 cho QAT

Khuyến nghị thực tế:

1. Dùng `quant_type="Q8_0"` cho các linear layer cần train với nhiễu quantization.
2. Nếu muốn mô phỏng thêm nhiễu activation, bật `activation_quant="Q8_0"` nhưng phải đo ablation riêng vì GGUF không lưu activation quantized.
3. Sau train, unpatch về `nn.Linear`.
4. Lưu checkpoint HF.
5. Convert sang GGUF F16/BF16 bằng converter của llama.cpp.
6. Chạy `llama-quantize input.gguf output-q8_0.gguf Q8_0`.
7. Chạy inference smoke test và eval metric cố định.

## Điều kiện chấp nhận

Một thay đổi `Q8_0` chỉ nên được xem là hợp lệ nếu đạt các điều kiện sau:

| Lớp kiểm thử | Điều kiện pass |
| --- | --- |
| Công thức tensor | `fake_quant_q8_0(..., use_ste=False)` khớp reference PyTorch cho nhiều shape/block |
| Rounding | Case half tie khớp C/C++ `roundf` |
| GGUF alignment | Dequant từ `gguf-py` `Q8_0` khớp fake quant với `max_error = 0` và `mismatches = 0` |
| Triton alignment | `fake_quant_q8_0_triton` khớp PyTorch reference với `max_error = 0` và `mismatches = 0` trên CUDA |
| Layer wrapper | Forward/backward chạy, gradient đi qua STE, output shape đúng |
| Model surgery | Patch/unpatch không đổi key trong `state_dict`, không replace `lm_head` nếu cấu hình skip |
| Export path | HF checkpoint sau unpatch convert được sang GGUF và `llama-quantize Q8_0` chạy thành công |
| Inference | `llama-cli` load được GGUF Q8_0 và sinh text ổn định với prompt cố định |

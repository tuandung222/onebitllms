# Copyright 2025 The Falcon-LLM Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");

import torch
import torch.nn as nn

from onebitllms import (
    LlamaCppFakeQuantLinear,
    fake_quant_q1_0,
    fake_quant_q2_0,
    fake_quant_q4_0,
    fake_quant_q4_1,
    replace_linear_with_llama_cpp_fake_quant_linear,
)


def round_away_from_zero(x):
    return torch.where(x >= 0, torch.floor(x + 0.5), torch.ceil(x - 0.5))


def q1_0_reference(x):
    orig_shape = x.shape
    xb = x.contiguous().to(torch.float32).view(-1, x.shape[-1] // 128, 128)
    d = xb.abs().mean(dim=-1, keepdim=True).to(torch.float16).to(torch.float32)
    sign = torch.where(xb >= 0.0, torch.ones_like(xb), -torch.ones_like(xb))
    return (d * sign).reshape(orig_shape).to(x.dtype)


def q2_0_reference(x):
    orig_shape = x.shape
    xb = x.contiguous().to(torch.float32).view(-1, x.shape[-1] // 128, 128)
    amax = xb.abs().max(dim=-1, keepdim=True).values
    d = amax.to(torch.float16).to(torch.float32)
    inv = torch.where(amax > 0.0, 1.0 / amax, torch.zeros_like(amax))
    q = (round_away_from_zero(xb * inv) + 1.0).clamp(0.0, 3.0)
    return ((q - 1.0) * d).reshape(orig_shape).to(x.dtype)


def q4_0_reference(x):
    orig_shape = x.shape
    xb = x.contiguous().to(torch.float32).view(-1, x.shape[-1] // 32, 32)
    max_idx = xb.abs().argmax(dim=-1, keepdim=True)
    signed_absmax = xb.gather(dim=-1, index=max_idx)
    d_raw = signed_absmax / -8.0
    d = d_raw.to(torch.float16).to(torch.float32)
    inv_d = torch.where(d_raw != 0, 1.0 / d_raw, torch.zeros_like(d_raw))
    q = torch.trunc(xb * inv_d + 8.5).clamp(0.0, 15.0)
    return ((q - 8.0) * d).reshape(orig_shape).to(x.dtype)


def q4_1_reference(x):
    orig_shape = x.shape
    xb = x.contiguous().to(torch.float32).view(-1, x.shape[-1] // 32, 32)
    x_min = xb.min(dim=-1, keepdim=True).values
    x_max = xb.max(dim=-1, keepdim=True).values
    d_raw = (x_max - x_min) / 15.0
    d = d_raw.to(torch.float16).to(torch.float32)
    m = x_min.to(torch.float16).to(torch.float32)
    inv_d = torch.where(d_raw != 0, 1.0 / d_raw, torch.zeros_like(d_raw))
    q = torch.trunc((xb - x_min) * inv_d + 0.5).clamp(0.0, 15.0)
    return (q * d + m).reshape(orig_shape).to(x.dtype)


def test_q1_0_matches_reference():
    torch.manual_seed(10)
    tensor = torch.randn(5, 256, dtype=torch.float32)

    actual = fake_quant_q1_0(tensor, use_ste=False)
    expected = q1_0_reference(tensor)

    assert torch.equal(actual, expected)


def test_q1_0_zero_and_sign_boundary_match_reference():
    tensor = torch.zeros(1, 128, dtype=torch.float32)
    tensor[0, 1::2] = -0.0
    tensor[0, 2::4] = -1.0

    actual = fake_quant_q1_0(tensor, use_ste=False)
    expected = q1_0_reference(tensor)

    assert torch.equal(actual, expected)


def test_q2_0_matches_reference():
    torch.manual_seed(11)
    tensor = torch.randn(5, 256, dtype=torch.float32)

    actual = fake_quant_q2_0(tensor, use_ste=False)
    expected = q2_0_reference(tensor)

    assert torch.equal(actual, expected)


def test_q2_0_half_ties_use_cpp_rounding():
    tensor = torch.zeros(1, 128, dtype=torch.float32)
    tensor[0, :8] = torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, -1.5])

    actual = fake_quant_q2_0(tensor, use_ste=False)
    expected = q2_0_reference(tensor)

    assert torch.equal(actual, expected)


def test_q4_0_matches_reference():
    torch.manual_seed(0)
    tensor = torch.randn(7, 96, dtype=torch.float32)

    actual = fake_quant_q4_0(tensor, use_ste=False)
    expected = q4_0_reference(tensor)

    assert torch.equal(actual, expected)


def test_q4_1_matches_reference():
    torch.manual_seed(1)
    tensor = torch.randn(7, 96, dtype=torch.float32)

    actual = fake_quant_q4_1(tensor, use_ste=False)
    expected = q4_1_reference(tensor)

    assert torch.equal(actual, expected)


def test_llama_cpp_fake_quant_uses_ste_gradients():
    weight = torch.randn(4, 64, dtype=torch.float32, requires_grad=True)

    quantized = fake_quant_q4_1(weight)
    loss = quantized.square().mean()
    loss.backward()

    assert weight.grad is not None
    assert torch.isfinite(weight.grad).all()


def test_llama_cpp_fake_quant_linear_forward_and_backward():
    torch.manual_seed(2)
    layer = LlamaCppFakeQuantLinear(128, 8, bias=True, quant_type="Q2_0")
    x = torch.randn(3, 128, dtype=torch.float32, requires_grad=True)

    y = layer(x)
    loss = y.square().mean()
    loss.backward()

    assert y.shape == (3, 8)
    assert x.grad is not None
    assert layer.weight.grad is not None


def test_replace_linear_with_llama_cpp_fake_quant_linear():
    model = nn.Sequential(
        nn.Linear(128, 16),
        nn.ReLU(),
        nn.Sequential(nn.Linear(128, 8), nn.Linear(64, 4)),
    )

    replaced = replace_linear_with_llama_cpp_fake_quant_linear(model, quant_type="Q1_0")

    assert replaced is model
    assert isinstance(model[0], LlamaCppFakeQuantLinear)
    assert isinstance(model[2][0], LlamaCppFakeQuantLinear)
    assert isinstance(model[2][1], nn.Linear)
    assert model[0].quant_type == "Q1_0"


def test_llama_cpp_fake_quant_linear_rejects_invalid_block_size():
    try:
        LlamaCppFakeQuantLinear(64, 8, quant_type="Q1_0")
    except ValueError as exc:
        assert "divisible by 128" in str(exc)
    else:
        raise AssertionError("expected invalid Q1_0 block size to fail")

# Copyright 2025 The Falcon-LLM Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");

import torch
import torch.nn as nn

from onebitllms import (
    LlamaCppFakeQuantLinear,
    assert_state_dict_keys_unchanged,
    fake_quant_q1_0,
    fake_quant_q2_0,
    fake_quant_q4_0,
    fake_quant_q4_1,
    fake_quant_q8_0_activation,
    replace_linear_with_llama_cpp_fake_quant_linear,
    replace_llama_cpp_fake_quant_linear_with_linear,
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


def q8_0_activation_reference(x):
    orig_shape = x.shape
    xb = x.contiguous().to(torch.float32).view(-1, x.shape[-1] // 32, 32)
    d_raw = xb.abs().max(dim=-1, keepdim=True).values / 127.0
    d = d_raw.to(torch.float16).to(torch.float32)
    inv_d = torch.where(d_raw != 0, 1.0 / d_raw, torch.zeros_like(d_raw))
    q = round_away_from_zero(xb * inv_d).clamp(-128.0, 127.0)
    return (q * d).reshape(orig_shape).to(x.dtype)


class ToyExportModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(96, 16)
        self.block = nn.Sequential(nn.Linear(64, 8), nn.ReLU(), nn.Linear(30, 4))
        self.lm_head = nn.Linear(96, 16)

    def forward(self, x):
        return self.lm_head(x) + self.proj(x)


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


def test_q8_0_activation_matches_reference():
    torch.manual_seed(4)
    tensor = torch.randn(3, 2, 96, dtype=torch.float32)

    actual = fake_quant_q8_0_activation(tensor, use_ste=False)
    expected = q8_0_activation_reference(tensor)

    assert torch.equal(actual, expected)


def test_q8_0_activation_half_ties_use_cpp_rounding():
    tensor = torch.zeros(1, 32, dtype=torch.float32)
    tensor[0, :8] = torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, -1.5])

    actual = fake_quant_q8_0_activation(tensor, use_ste=False)
    expected = q8_0_activation_reference(tensor)

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


def test_llama_cpp_fake_quant_linear_supports_q4_types():
    torch.manual_seed(3)

    for quant_type in ("Q4_0", "Q4_1"):
        layer = LlamaCppFakeQuantLinear(96, 8, bias=True, quant_type=quant_type)
        x = torch.randn(2, 96, dtype=torch.float32, requires_grad=True)

        y = layer(x)
        loss = y.square().mean()
        loss.backward()

        assert y.shape == (2, 8)
        assert x.grad is not None
        assert layer.weight.grad is not None
        assert layer.quant_type == quant_type


def test_llama_cpp_fake_quant_linear_supports_q8_0_activation_quant():
    torch.manual_seed(5)
    layer = LlamaCppFakeQuantLinear(96, 8, bias=True, quant_type="Q4_0", activation_quant="Q8_0")
    x = torch.randn(2, 96, dtype=torch.float32, requires_grad=True)

    y = layer(x)
    loss = y.square().mean()
    loss.backward()

    assert y.shape == (2, 8)
    assert x.grad is not None
    assert layer.weight.grad is not None
    assert layer.activation_quant == "Q8_0"


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


def test_replace_linear_with_llama_cpp_fake_quant_linear_supports_q4_types():
    model = nn.Sequential(
        nn.Linear(96, 16),
        nn.ReLU(),
        nn.Sequential(nn.Linear(64, 8), nn.Linear(30, 4)),
    )

    replaced = replace_linear_with_llama_cpp_fake_quant_linear(model, quant_type="Q4_1")

    assert replaced is model
    assert isinstance(model[0], LlamaCppFakeQuantLinear)
    assert isinstance(model[2][0], LlamaCppFakeQuantLinear)
    assert isinstance(model[2][1], nn.Linear)
    assert model[0].quant_type == "Q4_1"
    assert model[2][0].quant_type == "Q4_1"


def test_replace_linear_with_llama_cpp_fake_quant_linear_supports_q8_0_activation_quant():
    model = nn.Sequential(
        nn.Linear(96, 16),
        nn.ReLU(),
        nn.Sequential(nn.Linear(64, 8), nn.Linear(30, 4)),
    )

    replaced = replace_linear_with_llama_cpp_fake_quant_linear(
        model,
        quant_type="Q4_1",
        activation_quant="Q8_0",
    )

    assert replaced is model
    assert isinstance(model[0], LlamaCppFakeQuantLinear)
    assert isinstance(model[2][0], LlamaCppFakeQuantLinear)
    assert isinstance(model[2][1], nn.Linear)
    assert model[0].activation_quant == "Q8_0"
    assert model[2][0].activation_quant == "Q8_0"


def test_fake_quant_patch_unpatch_preserves_state_dict_keys_and_values():
    torch.manual_seed(6)
    model = ToyExportModel().to(dtype=torch.float64)
    model.eval()
    model.proj.weight.requires_grad_(False)
    model.proj.bias.requires_grad_(False)

    before_keys = tuple(model.state_dict().keys())
    before_state = {key: value.detach().clone() for key, value in model.state_dict().items()}

    replace_linear_with_llama_cpp_fake_quant_linear(
        model,
        quant_type="Q4_1",
        activation_quant="Q8_0",
    )

    assert_state_dict_keys_unchanged(before_keys, model)
    assert isinstance(model.proj, LlamaCppFakeQuantLinear)
    assert isinstance(model.block[0], LlamaCppFakeQuantLinear)
    assert isinstance(model.block[2], nn.Linear)
    assert isinstance(model.lm_head, nn.Linear)
    assert model.proj.training is False
    assert model.proj.weight.requires_grad is False
    assert model.proj.weight.dtype == torch.float64
    assert not any("activation_quant" in key for key in model.state_dict())

    for key, value in model.state_dict().items():
        assert torch.equal(value, before_state[key])

    replace_llama_cpp_fake_quant_linear_with_linear(model)

    assert_state_dict_keys_unchanged(before_keys, model)
    assert isinstance(model.proj, nn.Linear)
    assert isinstance(model.block[0], nn.Linear)
    assert isinstance(model.block[2], nn.Linear)
    assert isinstance(model.lm_head, nn.Linear)
    assert model.proj.training is False
    assert model.proj.weight.requires_grad is False
    assert model.proj.weight.dtype == torch.float64

    for key, value in model.state_dict().items():
        assert torch.equal(value, before_state[key])


def test_assert_state_dict_keys_unchanged_reports_key_changes():
    try:
        assert_state_dict_keys_unchanged(("a.weight",), ("a.weight", "extra"))
    except AssertionError as exc:
        assert "unexpected" in str(exc)
    else:
        raise AssertionError("expected state_dict key mismatch to fail")


def test_llama_cpp_fake_quant_linear_rejects_invalid_block_size():
    try:
        LlamaCppFakeQuantLinear(64, 8, quant_type="Q1_0")
    except ValueError as exc:
        assert "divisible by 128" in str(exc)
    else:
        raise AssertionError("expected invalid Q1_0 block size to fail")


def test_llama_cpp_fake_quant_linear_rejects_invalid_activation_quant():
    try:
        LlamaCppFakeQuantLinear(64, 8, quant_type="Q4_0", activation_quant="Q4_0")
    except ValueError as exc:
        assert "activation fake quant type" in str(exc)
    else:
        raise AssertionError("expected invalid activation fake quant type to fail")

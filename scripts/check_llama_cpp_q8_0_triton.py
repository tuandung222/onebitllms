#!/usr/bin/env python3
"""Validate the Q8_0 Triton fast path against the PyTorch reference."""

from __future__ import annotations

import argparse
import time

import torch

from onebitllms import fake_quant_q8_0, fake_quant_q8_0_triton


def _case_tensors(seed: int, device: torch.device) -> list[tuple[str, torch.Tensor]]:
    generator = torch.Generator(device=device).manual_seed(seed)
    cases = [
        ("random_1x32", torch.randn((1, 32), generator=generator, dtype=torch.float32, device=device)),
        ("random_3x96", torch.randn((3, 96), generator=generator, dtype=torch.float32, device=device) * 2.5),
        ("random_9x128", torch.randn((9, 128), generator=generator, dtype=torch.float32, device=device) * 3.0),
        ("zeros_2x64", torch.zeros((2, 64), dtype=torch.float32, device=device)),
        ("constant_2x32", torch.full((2, 32), 0.125, dtype=torch.float32, device=device)),
        ("large_range_4x64", torch.linspace(-32.0, 31.5, 256, dtype=torch.float32, device=device).reshape(4, 64)),
    ]
    ties = torch.zeros((1, 32), dtype=torch.float32, device=device)
    ties[0, :8] = torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, -1.5], device=device)
    cases.append(("round_ties_1x32", ties))
    return cases


def _benchmark(device: torch.device, seed: int, iterations: int) -> None:
    generator = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn((4096, 4096), generator=generator, dtype=torch.float32, device=device)

    fake_quant_q8_0(x, use_ste=False)
    fake_quant_q8_0_triton(x, use_ste=False)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iterations):
        fake_quant_q8_0(x, use_ste=False)
    torch.cuda.synchronize()
    torch_time = (time.perf_counter() - start) / iterations

    start = time.perf_counter()
    for _ in range(iterations):
        fake_quant_q8_0_triton(x, use_ste=False)
    torch.cuda.synchronize()
    triton_time = (time.perf_counter() - start) / iterations

    print(f"benchmark: torch={torch_time:.6f}s triton={triton_time:.6f}s speedup={torch_time / triton_time:.2f}x")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--allow-missing-cuda", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--benchmark-iterations", type=int, default=20)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        message = "CUDA is required to validate fake_quant_q8_0_triton"
        if args.allow_missing_cuda:
            print(f"skip: {message}")
            return 0
        raise RuntimeError(message)

    device = torch.device("cuda")
    total_mismatches = 0
    max_error = 0.0

    for name, tensor in _case_tensors(args.seed, device):
        expected = fake_quant_q8_0(tensor, use_ste=False)
        actual = fake_quant_q8_0_triton(tensor, use_ste=False)
        delta = (actual - expected).abs()
        case_max_error = float(delta.max().item()) if delta.numel() else 0.0
        case_mismatches = int(torch.count_nonzero(delta).item())
        total_mismatches += case_mismatches
        max_error = max(max_error, case_max_error)
        print(f"{name}: max_error={case_max_error:.8g} mismatches={case_mismatches}")

    print(f"summary: max_error={max_error:.8g} mismatches={total_mismatches}")
    if total_mismatches != 0 or max_error != 0.0:
        raise AssertionError("Q8_0 Triton fake quantization is not bit-exact with the PyTorch reference")

    if args.benchmark:
        _benchmark(device, args.seed, args.benchmark_iterations)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

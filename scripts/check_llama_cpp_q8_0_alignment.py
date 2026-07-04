#!/usr/bin/env python3
"""Check onebitllms Q8_0 fake quantization against prism-llama-cpp gguf-py."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

from onebitllms import fake_quant_q8_0


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _default_prism_llama_cpp() -> Path | None:
    repo_root = _repo_root()
    candidates = [
        os.environ.get("PRISM_LLAMA_CPP"),
        repo_root.parent / "prism-llama-cpp",
        repo_root.parent.parent / "prism-llama-cpp",
    ]
    for candidate in candidates:
        if candidate is None:
            continue
        path = Path(candidate).expanduser().resolve()
        if (path / "gguf-py" / "gguf" / "quants.py").exists():
            return path
    return None


def _load_gguf(prism_llama_cpp: Path):
    gguf_py = prism_llama_cpp / "gguf-py"
    if not gguf_py.exists():
        raise FileNotFoundError(f"missing gguf-py directory: {gguf_py}")
    sys.path.insert(0, str(gguf_py))

    from gguf import GGMLQuantizationType  # pylint: disable=import-outside-toplevel
    from gguf.quants import dequantize, quantize  # pylint: disable=import-outside-toplevel

    return GGMLQuantizationType, quantize, dequantize


def _case_tensors(seed: int) -> list[tuple[str, torch.Tensor]]:
    generator = torch.Generator().manual_seed(seed)
    cases = [
        ("random_1x32", torch.randn((1, 32), generator=generator, dtype=torch.float32)),
        ("random_3x96", torch.randn((3, 96), generator=generator, dtype=torch.float32) * 2.5),
        ("random_9x128", torch.randn((9, 128), generator=generator, dtype=torch.float32) * 3.0),
        ("zeros_2x64", torch.zeros((2, 64), dtype=torch.float32)),
        ("constant_2x32", torch.full((2, 32), 0.125, dtype=torch.float32)),
        ("large_range_4x64", torch.linspace(-32.0, 31.5, 256, dtype=torch.float32).reshape(4, 64)),
    ]
    ties = torch.zeros((1, 32), dtype=torch.float32)
    ties[0, :8] = torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, -1.5])
    cases.append(("round_ties_1x32", ties))
    return cases


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prism-llama-cpp",
        type=Path,
        default=None,
        help="Path to prism-llama-cpp. Defaults to PRISM_LLAMA_CPP or nearby workspace paths.",
    )
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    prism_llama_cpp = args.prism_llama_cpp.expanduser().resolve() if args.prism_llama_cpp else _default_prism_llama_cpp()
    if prism_llama_cpp is None:
        raise FileNotFoundError(
            "Cannot find prism-llama-cpp. Pass --prism-llama-cpp or set PRISM_LLAMA_CPP."
        )

    GGMLQuantizationType, quantize, dequantize = _load_gguf(prism_llama_cpp)
    total_mismatches = 0
    max_error = 0.0

    for name, tensor in _case_tensors(args.seed):
        gguf_dequant = dequantize(
            quantize(tensor.numpy(), GGMLQuantizationType.Q8_0),
            GGMLQuantizationType.Q8_0,
        )
        fake = fake_quant_q8_0(tensor, use_ste=False).numpy()
        delta = np.abs(gguf_dequant - fake)
        case_max_error = float(delta.max()) if delta.size else 0.0
        case_mismatches = int(np.count_nonzero(delta))
        total_mismatches += case_mismatches
        max_error = max(max_error, case_max_error)
        print(f"{name}: max_error={case_max_error:.8g} mismatches={case_mismatches}")

    print(f"summary: max_error={max_error:.8g} mismatches={total_mismatches}")
    if total_mismatches != 0 or max_error != 0.0:
        raise AssertionError("Q8_0 fake quantization is not bit-exact with gguf-py Q8_0 dequantization")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

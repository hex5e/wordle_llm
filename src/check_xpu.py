"""Verify that PyTorch can execute tensor operations on an Intel XPU."""

from __future__ import annotations

import platform
import sys


def _print_environment(torch: object) -> None:
    """Print versions that are useful when diagnosing an XPU failure."""
    print(f"Python: {platform.python_version()}")
    print(f"Platform: {platform.platform()}")
    print(f"PyTorch: {torch.__version__}")


def main() -> int:
    """Run a small calculation on xpu:0 and return a process exit code."""
    try:
        import torch
    except ImportError as exc:
        print(f"FAIL: PyTorch could not be imported: {exc}", file=sys.stderr)
        print("Run `uv sync` and try again.", file=sys.stderr)
        return 1

    _print_environment(torch)

    if not hasattr(torch, "xpu"):
        print("XPU available: False")
        print("FAIL: This PyTorch build does not include the XPU backend.", file=sys.stderr)
        return 1

    try:
        available = torch.xpu.is_available()
    except Exception as exc:
        print("XPU available: False")
        print(f"FAIL: PyTorch could not initialize the XPU backend: {exc}", file=sys.stderr)
        return 1

    print(f"XPU available: {available}")
    if not available:
        print(
            "FAIL: PyTorch cannot access an Intel XPU. Check the installed "
            "PyTorch build and Intel graphics driver.",
            file=sys.stderr,
        )
        return 1

    try:
        device_count = torch.xpu.device_count()
        print(f"XPU device count: {device_count}")
        for index in range(device_count):
            print(f"XPU {index}: {torch.xpu.get_device_name(index)}")

        device = torch.device("xpu:0")
        values = torch.tensor([1.0, 2.0, 3.0, 4.0], device=device)
        result = values.square().sum()
        torch.xpu.synchronize()
        result_value = result.item()
    except Exception as exc:
        print(f"FAIL: Tensor execution on xpu:0 failed: {exc}", file=sys.stderr)
        return 1

    expected = 30.0
    print(f"Sanity check result: {result_value:.1f} (expected {expected:.1f})")
    if result_value != expected:
        print("FAIL: The XPU calculation returned an unexpected result.", file=sys.stderr)
        return 1

    print("PASS: PyTorch can execute tensor operations on the Intel XPU.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

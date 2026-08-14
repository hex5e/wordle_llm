# worldle_llm

A small model trained on a laptop to play Wordle.

## Milestone 1: verify Intel XPU support

The first milestone checks that PyTorch can see the Intel GPU and execute a
real tensor operation on it. The project uses
[`uv`](https://docs.astral.sh/uv/) for Python and dependency management.

### Prerequisites

- Windows 11
- A current Intel graphics driver
- `uv`

### Run the check

From the repository root:

```powershell
uv sync
uv run python src/check_xpu.py
```

`uv sync` installs the pinned Python version and the PyTorch build from the
official Intel XPU wheel index. A successful check ends with output similar to:

```text
Python: 3.12.x
Platform: Windows-11-...
PyTorch: ...+xpu
XPU available: True
XPU device count: 1
XPU 0: Intel(R) Arc(TM) 140T GPU
Sanity check result: 30.0 (expected 30.0)
PASS: PyTorch can execute tensor operations on the Intel XPU.
```

If `XPU available` is `False`, update the Intel graphics driver and run the
check again. The script exits nonzero on failure so it can also be used in
automated diagnostics.

## Milestone 2: benchmark a pretrained transformer

The next check loads
[`HuggingFaceTB/SmolLM2-135M-Instruct`](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct)
and benchmarks Hugging Face generation on the XPU in FP32, FP16, and BF16:

```powershell
uv sync
uv run python src/benchmark_model.py
```

The first run downloads the model. Run the command a second time for load-time
measurements that do not include network transfer. Each dtype reports model
load success, load time, allocated model memory, peak generation memory,
generation throughput, and a sample generation. An unsupported dtype is
reported as a failed result without preventing the remaining dtypes from
running.

For a quicker smoke test or a custom model:

```powershell
uv run python src/benchmark_model.py --max-new-tokens 16 --repeats 1
uv run python src/benchmark_model.py --model HuggingFaceTB/SmolLM2-360M-Instruct
```

## References

- [PyTorch: Getting Started on Intel GPU](https://docs.pytorch.org/docs/stable/notes/get_start_xpu.html)
- [uv: Using uv with PyTorch](https://docs.astral.sh/uv/guides/integration/pytorch/)

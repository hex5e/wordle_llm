"""Benchmark Hugging Face text generation on an Intel XPU."""

from __future__ import annotations

import argparse
import gc
import platform
import sys
import time
from dataclasses import dataclass

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"
DEFAULT_PROMPT = (
    "We are playing Wordle. Reply with exactly one common five-letter English "
    "word and no other text."
)

DTYPES = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


@dataclass
class BenchmarkResult:
    """Measurements from one model dtype, or the error it produced."""

    dtype: str
    parameters_millions: float | None = None
    load_seconds: float | None = None
    model_mib: float | None = None
    peak_mib: float | None = None
    tokens_per_second: float | None = None
    generated_text: str | None = None
    error: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark a small pretrained transformer on Intel XPU."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Hugging Face model ID")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument(
        "--dtypes",
        nargs="+",
        choices=DTYPES,
        default=list(DTYPES),
        help="Precisions to benchmark (default: fp32 fp16 bf16)",
    )
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be at least 1")
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")
    return args


def mib(byte_count: int) -> float:
    return byte_count / (1024 * 1024)


def print_environment() -> None:
    print(f"Python: {platform.python_version()}")
    print(f"Platform: {platform.platform()}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Transformers: {transformers.__version__}")
    print(f"XPU: {torch.xpu.get_device_name(0)}")


def release_xpu_memory() -> None:
    gc.collect()
    try:
        torch.xpu.empty_cache()
        torch.xpu.synchronize()
    except Exception:
        # Cleanup should not hide the dtype-specific error we want to report.
        pass


def benchmark_dtype(
    *,
    model_id: str,
    tokenizer: object,
    inputs: object,
    dtype_name: str,
    max_new_tokens: int,
    repeats: int,
) -> BenchmarkResult:
    """Load and benchmark one dtype without allowing failures to stop later runs."""
    result = BenchmarkResult(dtype=dtype_name)
    model = None
    generated = None
    try:
        release_xpu_memory()
        torch.xpu.memory.reset_peak_memory_stats(0)

        started = time.perf_counter()
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=DTYPES[dtype_name],
        ).to("xpu:0")
        model.eval()
        torch.xpu.synchronize()
        result.load_seconds = time.perf_counter() - started
        result.parameters_millions = sum(
            parameter.numel() for parameter in model.parameters()
        ) / 1_000_000
        result.model_mib = mib(torch.xpu.memory.memory_allocated(0))

        # Warm up lazy kernels without including compilation/startup overhead.
        with torch.inference_mode():
            model.generate(
                **inputs,
                do_sample=False,
                min_new_tokens=min(8, max_new_tokens),
                max_new_tokens=min(8, max_new_tokens),
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )
        torch.xpu.synchronize()
        torch.xpu.memory.reset_peak_memory_stats(0)

        total_tokens = 0
        generated = None
        started = time.perf_counter()
        with torch.inference_mode():
            for _ in range(repeats):
                generated = model.generate(
                    **inputs,
                    do_sample=False,
                    min_new_tokens=max_new_tokens,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=tokenizer.eos_token_id,
                    use_cache=True,
                )
                total_tokens += generated.shape[-1] - inputs["input_ids"].shape[-1]
        torch.xpu.synchronize()
        elapsed = time.perf_counter() - started

        result.tokens_per_second = total_tokens / elapsed
        result.peak_mib = mib(torch.xpu.memory.max_memory_allocated(0))
        new_tokens = generated[0, inputs["input_ids"].shape[-1] :]
        result.generated_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    except Exception as exc:
        result.error = f"{type(exc).__name__}: {exc}"
    finally:
        del generated
        del model
        release_xpu_memory()
    return result


def print_result(result: BenchmarkResult) -> None:
    print(f"\n[{result.dtype}]")
    if result.error is not None:
        print(f"FAIL: {result.error}")
        return
    print("Model load: PASS")
    print(f"Parameters: {result.parameters_millions:.1f}M")
    print(f"Load time: {result.load_seconds:.2f} s")
    print(f"Model memory: {result.model_mib:.1f} MiB")
    print(f"Peak generation memory: {result.peak_mib:.1f} MiB")
    print(f"Generation throughput: {result.tokens_per_second:.2f} tokens/s")
    print(f"Generated text: {result.generated_text!r}")


def main() -> int:
    args = parse_args()
    if not hasattr(torch, "xpu") or not torch.xpu.is_available():
        print("FAIL: PyTorch cannot access an Intel XPU.", file=sys.stderr)
        return 1

    print_environment()
    print(f"Model: {args.model}")
    print(f"Timed generation: {args.repeats} x {args.max_new_tokens} tokens")
    print("Loading tokenizer...")

    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        messages = [{"role": "user", "content": args.prompt}]
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to("xpu:0")
    except Exception as exc:
        print(f"FAIL: Tokenizer setup failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    results = [
        benchmark_dtype(
            model_id=args.model,
            tokenizer=tokenizer,
            inputs=inputs,
            dtype_name=dtype_name,
            max_new_tokens=args.max_new_tokens,
            repeats=args.repeats,
        )
        for dtype_name in args.dtypes
    ]
    for result in results:
        print_result(result)

    successful = sum(result.error is None for result in results)
    print(f"\nSummary: {successful}/{len(results)} dtype benchmarks passed.")
    return 0 if successful else 1


if __name__ == "__main__":
    raise SystemExit(main())

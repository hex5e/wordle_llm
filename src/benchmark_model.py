"""Benchmark Hugging Face text generation on an Intel XPU."""

from __future__ import annotations

import argparse
import gc
import multiprocessing
import os
import platform
import queue as queue_module
import sys
import time
from dataclasses import asdict, dataclass

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
HEARTBEAT_SECONDS = 10


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


def log(message: str, *, dtype_name: str | None = None, error: bool = False) -> None:
    """Print a timestamped progress message immediately."""
    timestamp = time.strftime("%H:%M:%S")
    dtype_prefix = f" [{dtype_name}]" if dtype_name else ""
    stream = sys.stderr if error else sys.stdout
    print(f"[{timestamp}]{dtype_prefix} {message}", file=stream, flush=True)


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
    print(f"Python: {platform.python_version()}", flush=True)
    print(f"Platform: {platform.platform()}", flush=True)
    print(f"PyTorch: {torch.__version__}", flush=True)
    print(f"Transformers: {transformers.__version__}", flush=True)
    print(f"XPU: {torch.xpu.get_device_name(0)}", flush=True)


def release_xpu_memory(dtype_name: str) -> None:
    log("Releasing Python objects and the XPU cache...", dtype_name=dtype_name)
    gc.collect()
    try:
        torch.xpu.empty_cache()
        torch.xpu.synchronize()
    except Exception as exc:
        log(
            f"XPU cleanup warning: {type(exc).__name__}: {exc}",
            dtype_name=dtype_name,
            error=True,
        )
    else:
        log("XPU cleanup complete.", dtype_name=dtype_name)


def benchmark_dtype(
    *,
    model_id: str,
    prompt: str,
    dtype_name: str,
    max_new_tokens: int,
    repeats: int,
) -> BenchmarkResult:
    """Load and benchmark one dtype inside an isolated worker process."""
    result = BenchmarkResult(dtype=dtype_name)
    model = None
    tokenizer = None
    inputs = None
    generated = None
    try:
        log(f"Worker started (PID {os.getpid()}).", dtype_name=dtype_name)
        log("Checking XPU availability...", dtype_name=dtype_name)
        if not hasattr(torch, "xpu") or not torch.xpu.is_available():
            raise RuntimeError("PyTorch cannot access an Intel XPU")
        log(f"Using {torch.xpu.get_device_name(0)}.", dtype_name=dtype_name)

        release_xpu_memory(dtype_name)
        torch.xpu.memory.reset_peak_memory_stats(0)

        log("Loading tokenizer (network/cache activity may follow)...", dtype_name=dtype_name)
        stage_started = time.perf_counter()
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        log(
            f"Tokenizer loaded in {time.perf_counter() - stage_started:.2f} s.",
            dtype_name=dtype_name,
        )

        log("Tokenizing the benchmark prompt on CPU...", dtype_name=dtype_name)
        messages = [{"role": "user", "content": prompt}]
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        prompt_tokens = inputs["input_ids"].shape[-1]
        log(f"Prompt contains {prompt_tokens} tokens.", dtype_name=dtype_name)

        log(
            "Loading model weights on CPU (network/cache activity may follow)...",
            dtype_name=dtype_name,
        )
        load_started = time.perf_counter()
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=DTYPES[dtype_name],
        )
        result.parameters_millions = sum(
            parameter.numel() for parameter in model.parameters()
        ) / 1_000_000
        log(
            f"CPU model loaded: {result.parameters_millions:.1f}M parameters.",
            dtype_name=dtype_name,
        )

        log("Moving model weights to xpu:0...", dtype_name=dtype_name)
        model = model.to("xpu:0")
        model.eval()
        log("Model transfer returned; synchronizing the XPU...", dtype_name=dtype_name)
        torch.xpu.synchronize()
        result.load_seconds = time.perf_counter() - load_started
        result.model_mib = mib(torch.xpu.memory.memory_allocated(0))
        log(
            f"Model ready on XPU in {result.load_seconds:.2f} s "
            f"({result.model_mib:.1f} MiB allocated).",
            dtype_name=dtype_name,
        )

        log("Moving prompt tensors to xpu:0...", dtype_name=dtype_name)
        inputs = inputs.to("xpu:0")
        torch.xpu.synchronize()
        log("Prompt tensors are ready on XPU.", dtype_name=dtype_name)

        warmup_tokens = min(8, max_new_tokens)
        log(
            f"Starting warmup generation ({warmup_tokens} tokens)...",
            dtype_name=dtype_name,
        )
        warmup_started = time.perf_counter()
        with torch.inference_mode():
            model.generate(
                **inputs,
                do_sample=False,
                min_new_tokens=warmup_tokens,
                max_new_tokens=warmup_tokens,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )
        log("Warmup generation returned; synchronizing the XPU...", dtype_name=dtype_name)
        torch.xpu.synchronize()
        log(
            f"Warmup complete in {time.perf_counter() - warmup_started:.2f} s.",
            dtype_name=dtype_name,
        )
        torch.xpu.memory.reset_peak_memory_stats(0)

        total_tokens = 0
        total_elapsed = 0.0
        with torch.inference_mode():
            for repeat in range(1, repeats + 1):
                log(
                    f"Timed repeat {repeat}/{repeats}: generating "
                    f"{max_new_tokens} tokens...",
                    dtype_name=dtype_name,
                )
                repeat_started = time.perf_counter()
                generated = model.generate(
                    **inputs,
                    do_sample=False,
                    min_new_tokens=max_new_tokens,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=tokenizer.eos_token_id,
                    use_cache=True,
                )
                log(
                    f"Repeat {repeat}/{repeats} returned; synchronizing the XPU...",
                    dtype_name=dtype_name,
                )
                torch.xpu.synchronize()
                repeat_elapsed = time.perf_counter() - repeat_started
                generated_tokens = generated.shape[-1] - prompt_tokens
                total_tokens += generated_tokens
                total_elapsed += repeat_elapsed
                log(
                    f"Repeat {repeat}/{repeats} complete: {generated_tokens} tokens "
                    f"in {repeat_elapsed:.2f} s "
                    f"({generated_tokens / repeat_elapsed:.2f} tokens/s).",
                    dtype_name=dtype_name,
                )

        log("Collecting final memory and generation results...", dtype_name=dtype_name)
        result.tokens_per_second = total_tokens / total_elapsed
        result.peak_mib = mib(torch.xpu.memory.max_memory_allocated(0))
        new_tokens = generated[0, prompt_tokens:]
        result.generated_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        log("Benchmark measurements complete.", dtype_name=dtype_name)
    except Exception as exc:
        result.error = f"{type(exc).__name__}: {exc}"
        log(f"Benchmark failed: {result.error}", dtype_name=dtype_name, error=True)
    finally:
        log("Beginning worker cleanup...", dtype_name=dtype_name)
        del generated
        del inputs
        del tokenizer
        del model
        release_xpu_memory(dtype_name)
        log("Worker finished.", dtype_name=dtype_name)
    return result


def benchmark_worker(
    result_queue: object,
    model_id: str,
    prompt: str,
    dtype_name: str,
    max_new_tokens: int,
    repeats: int,
) -> None:
    """Run one dtype and return its result to the parent process."""
    result = benchmark_dtype(
        model_id=model_id,
        prompt=prompt,
        dtype_name=dtype_name,
        max_new_tokens=max_new_tokens,
        repeats=repeats,
    )
    result_queue.put(asdict(result))


def stop_worker(process: multiprocessing.Process, dtype_name: str) -> None:
    """Terminate a worker, escalating to kill if native code does not exit."""
    log(
        f"Terminating worker PID {process.pid}; this may take a few seconds...",
        dtype_name=dtype_name,
        error=True,
    )
    process.terminate()
    try:
        process.join(timeout=5)
    except KeyboardInterrupt:
        pass
    if process.is_alive():
        log(
            f"Worker PID {process.pid} did not terminate; killing it...",
            dtype_name=dtype_name,
            error=True,
        )
        process.kill()
        try:
            process.join(timeout=5)
        except KeyboardInterrupt:
            pass
    log("Worker stopped.", dtype_name=dtype_name, error=True)


def run_dtype_in_worker(
    *,
    context: multiprocessing.context.BaseContext,
    model_id: str,
    prompt: str,
    dtype_name: str,
    max_new_tokens: int,
    repeats: int,
) -> BenchmarkResult:
    """Run a dtype in a process the parent can terminate on Ctrl+C."""
    result_queue = context.Queue()
    process = context.Process(
        target=benchmark_worker,
        args=(
            result_queue,
            model_id,
            prompt,
            dtype_name,
            max_new_tokens,
            repeats,
        ),
        name=f"xpu-benchmark-{dtype_name}",
    )
    log("Starting isolated benchmark worker...", dtype_name=dtype_name)
    process.start()
    log(
        f"Worker PID {process.pid} is running. Press Ctrl+C to stop it.",
        dtype_name=dtype_name,
    )
    worker_started = time.monotonic()
    next_heartbeat = worker_started + HEARTBEAT_SECONDS
    try:
        while process.is_alive():
            process.join(timeout=0.25)
            now = time.monotonic()
            if process.is_alive() and now >= next_heartbeat:
                elapsed = now - worker_started
                log(
                    f"Worker PID {process.pid} is still running "
                    f"({elapsed:.0f} s elapsed). Press Ctrl+C to stop it.",
                    dtype_name=dtype_name,
                )
                next_heartbeat = now + HEARTBEAT_SECONDS
    except KeyboardInterrupt:
        stop_worker(process, dtype_name)
        result_queue.cancel_join_thread()
        result_queue.close()
        raise

    try:
        result = BenchmarkResult(**result_queue.get(timeout=2))
    except queue_module.Empty:
        result = BenchmarkResult(
            dtype=dtype_name,
            error=f"Worker exited with code {process.exitcode} without returning a result",
        )
    finally:
        result_queue.cancel_join_thread()
        result_queue.close()
    return result


def print_result(result: BenchmarkResult) -> None:
    print(f"\n[{result.dtype}]", flush=True)
    if result.error is not None:
        print(f"FAIL: {result.error}", flush=True)
        return
    print("Model load: PASS", flush=True)
    print(f"Parameters: {result.parameters_millions:.1f}M", flush=True)
    print(f"Load time: {result.load_seconds:.2f} s", flush=True)
    print(f"Model memory: {result.model_mib:.1f} MiB", flush=True)
    print(f"Peak generation memory: {result.peak_mib:.1f} MiB", flush=True)
    print(f"Generation throughput: {result.tokens_per_second:.2f} tokens/s", flush=True)
    print(f"Generated text: {result.generated_text!r}", flush=True)


def main() -> int:
    args = parse_args()
    log("Checking XPU availability in the parent process...")
    if not hasattr(torch, "xpu") or not torch.xpu.is_available():
        log("FAIL: PyTorch cannot access an Intel XPU.", error=True)
        return 1

    print_environment()
    print(f"Model: {args.model}", flush=True)
    print(f"Timed generation: {args.repeats} x {args.max_new_tokens} tokens", flush=True)
    print(f"Dtypes: {', '.join(args.dtypes)}", flush=True)
    log("Each dtype will run in an isolated, interruptible worker process.")

    context = multiprocessing.get_context("spawn")
    results = []
    try:
        for dtype_name in args.dtypes:
            results.append(
                run_dtype_in_worker(
                    context=context,
                    model_id=args.model,
                    prompt=args.prompt,
                    dtype_name=dtype_name,
                    max_new_tokens=args.max_new_tokens,
                    repeats=args.repeats,
                )
            )
            print_result(results[-1])
    except KeyboardInterrupt:
        log("Benchmark interrupted by user. No further dtypes will run.", error=True)
        return 130

    successful = sum(result.error is None for result in results)
    print(f"\nSummary: {successful}/{len(results)} dtype benchmarks passed.", flush=True)
    return 0 if successful else 1


if __name__ == "__main__":
    multiprocessing.freeze_support()
    raise SystemExit(main())

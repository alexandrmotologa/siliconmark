"""High-precision, device-synchronized benchmark timing harnesses.

Provides synchronized kernel execution measurement across CUDA, MPS, and CPU,
avoiding asynchronous launch skew and cold-start artifacts.
"""

from __future__ import annotations

import time
from typing import Callable, List, Tuple
import torch

from siliconmark.core.config import IterationStats


def synchronize_device(device: torch.device) -> None:
    """Flush pending asynchronous compute queues across backends."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
        torch.mps.synchronize()
    elif device.type == "xpu" and hasattr(torch, "xpu") and hasattr(torch.xpu, "synchronize"):
        torch.xpu.synchronize(device)


def benchmark_function(
    fn: Callable[[], None],
    device: torch.device,
    warmup_iters: int = 10,
    bench_iters: int = 30,
) -> IterationStats:
    """Execute a callable with device synchronization and collect latency statistics.

    Args:
        fn: Zero-argument callable containing the benchmark kernel.
        device: Active PyTorch device (cuda, mps, cpu, etc.).
        warmup_iters: Number of untimed warmup executions.
        bench_iters: Number of timed executions.

    Returns:
        IterationStats populated with mean, median, min, max, p95, and std latencies.
    """
    # Warmup phase: stabilize caches, memory controllers, and GPU clocks
    for _ in range(max(1, warmup_iters)):
        fn()
    synchronize_device(device)

    latencies_ms: List[float] = []

    if device.type == "cuda":
        # CUDA event-based timing gives microsecond GPU-accurate measurements
        for _ in range(bench_iters):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            fn()
            end_event.record()
            torch.cuda.synchronize(device)
            elapsed_ms = start_event.elapsed_time(end_event)
            latencies_ms.append(float(elapsed_ms))
    else:
        # High-resolution monotonic wall-clock timing with synchronization
        for _ in range(bench_iters):
            synchronize_device(device)
            t0 = time.perf_counter_ns()
            fn()
            synchronize_device(device)
            t1 = time.perf_counter_ns()
            elapsed_ms = (t1 - t0) / 1_000_000.0
            latencies_ms.append(elapsed_ms)

    latencies_ms.sort()
    n = len(latencies_ms)
    mean_ms = sum(latencies_ms) / n
    median_ms = latencies_ms[n // 2] if n % 2 != 0 else (latencies_ms[n // 2 - 1] + latencies_ms[n // 2]) / 2.0
    variance = sum((x - mean_ms) ** 2 for x in latencies_ms) / n
    std_ms = variance ** 0.5
    min_ms = latencies_ms[0]
    max_ms = latencies_ms[-1]
    p95_index = min(int(n * 0.95), n - 1)
    p95_ms = latencies_ms[p95_index]

    return IterationStats(
        iterations=bench_iters,
        mean_ms=round(mean_ms, 4),
        median_ms=round(median_ms, 4),
        std_ms=round(std_ms, 4),
        min_ms=round(min_ms, 4),
        max_ms=round(max_ms, 4),
        p95_ms=round(p95_ms, 4),
        raw_latencies_ms=[round(x, 4) for x in latencies_ms],
    )

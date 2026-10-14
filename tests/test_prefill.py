"""Unit tests for prefill TFLOPS benchmark."""

import pytest
from siliconmark.core.hardware import resolve_torch_device
from siliconmark.benchmarks.prefill import PrefillTflopsBenchmark
from siliconmark.core.config import PrefillBenchmarkConfig, Precision


def test_prefill_benchmark_execution():
    dev, info = resolve_torch_device(device_str="cpu")
    bench = PrefillTflopsBenchmark(dev, info)

    cfg = PrefillBenchmarkConfig(
        batch_sizes=[1],
        prompt_lens=[128],
        hidden_dims=[512],
        precisions=[Precision.FP32],
        warmup_iters=1,
        bench_iters=3,
    )

    result = bench.run(cfg)

    assert result.peak_tflops > 0.0
    assert result.sustained_tflops > 0.0
    assert result.peak_tokens_per_sec > 0.0
    assert len(result.points) == 1
    pt = result.points[0]
    assert pt.total_flops == 2 * 128 * 512 * 512
    assert pt.achieved_tflops > 0.0

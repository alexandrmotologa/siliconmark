"""Unit tests for attention kernel efficiency benchmark."""

import pytest
from siliconmark.core.hardware import resolve_torch_device
from siliconmark.benchmarks.attention import AttentionKernelBenchmark
from siliconmark.core.config import AttentionBenchmarkConfig, Precision


def test_attention_benchmark_execution():
    dev, info = resolve_torch_device(device_str="cpu")
    bench = AttentionKernelBenchmark(dev, info)

    cfg = AttentionBenchmarkConfig(
        seq_lens=[128],
        num_heads=4,
        head_dim=64,
        batch_size=1,
        precisions=[Precision.FP32],
        warmup_iters=1,
        bench_iters=3,
    )

    result = bench.run(cfg)

    assert result.sdpa_peak_tflops > 0.0
    assert result.naive_peak_tflops > 0.0
    assert result.mean_speedup > 0.0
    assert len(result.points) == 2  # 1 Naive, 1 SDPA

    sdpa_pt = [p for p in result.points if p.kernel_type == "sdpa"][0]
    naive_pt = [p for p in result.points if p.kernel_type == "naive"][0]

    assert sdpa_pt.total_flops == 4 * 1 * 4 * (128 ** 2) * 64
    assert naive_pt.speedup_over_naive == 1.0
    assert sdpa_pt.speedup_over_naive is not None

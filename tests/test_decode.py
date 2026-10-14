"""Unit tests for decode memory bandwidth benchmark."""

import pytest
from siliconmark.core.hardware import resolve_torch_device
from siliconmark.benchmarks.decode import DecodeMemoryBandwidthBenchmark
from siliconmark.core.config import DecodeBenchmarkConfig, Precision


def test_decode_benchmark_execution():
    dev, info = resolve_torch_device(device_str="cpu")
    bench = DecodeMemoryBandwidthBenchmark(dev, info)

    cfg = DecodeBenchmarkConfig(
        batch_sizes=[1],
        hidden_dims=[512],
        seq_lens=[256],
        precisions=[Precision.FP32],
        warmup_iters=1,
        bench_iters=3,
        include_paged_kv=True,
    )

    result = bench.run(cfg)

    assert result.peak_bandwidth_gbps > 0.0
    assert result.sustained_bandwidth_gbps > 0.0
    assert len(result.points) >= 3  # GEMV + KV-cache + Paged KV

    gemv_pts = [p for p in result.points if p.workload == "gemv_weight_streaming"]
    assert len(gemv_pts) == 1
    assert gemv_pts[0].bytes_transferred > 0
    assert gemv_pts[0].achieved_bandwidth_gbps > 0.0

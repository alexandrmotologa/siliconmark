"""Unit tests for SiliconMark core schemas and configurations."""

import pytest
from siliconmark.core.config import (
    BenchmarkRunConfig,
    DecodeBenchmarkConfig,
    DeviceInfo,
    DeviceType,
    IterationStats,
    Precision,
    PrefillBenchmarkConfig,
    AttentionBenchmarkConfig,
    SiliconMarkScores,
)


def test_device_info_validation():
    dev = DeviceInfo(
        device_type=DeviceType.CPU,
        device_name="Test Processor",
        device_index=0,
        total_memory_bytes=16 * 1024**3,
        total_memory_gb=16.0,
        available_memory_bytes=8 * 1024**3,
        available_memory_gb=8.0,
        compute_units=8,
        theoretical_bandwidth_gbps=51.2,
        os_platform="Linux",
        python_version="3.12.0",
        torch_version="2.2.0",
    )
    assert dev.device_type == DeviceType.CPU
    assert dev.total_memory_gb == 16.0
    assert dev.theoretical_bandwidth_gbps == 51.2


def test_iteration_stats():
    stats = IterationStats(
        iterations=5,
        mean_ms=1.2,
        median_ms=1.1,
        std_ms=0.1,
        min_ms=1.0,
        max_ms=1.4,
        p95_ms=1.4,
        raw_latencies_ms=[1.0, 1.1, 1.1, 1.2, 1.4],
    )
    assert stats.median_ms == 1.1
    assert len(stats.raw_latencies_ms) == 5


def test_default_configs():
    run_cfg = BenchmarkRunConfig()
    assert run_cfg.device == "auto"
    assert not run_cfg.quick_mode
    assert 1 in run_cfg.decode.batch_sizes
    assert Precision.FP16 in run_cfg.decode.precisions
    assert 32 in [run_cfg.attention.num_heads]

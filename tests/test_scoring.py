"""Unit tests for SiliconMark composite scoring calculations across all tiers and grades."""

import pytest
from siliconmark.core.config import (
    AttentionBenchmarkResult,
    DecodeBenchmarkResult,
    DeviceInfo,
    DeviceType,
    PrefillBenchmarkResult,
)
from siliconmark.core.scoring import (
    calculate_siliconmark_scores,
    REFERENCE_DECODE_BW_GBPS,
    REFERENCE_PREFILL_TFLOPS,
)


@pytest.fixture
def base_device():
    return DeviceInfo(
        device_type=DeviceType.CUDA,
        device_name="Test GPU",
        device_index=0,
        total_memory_bytes=24 * 1024**3,
        total_memory_gb=24.0,
        available_memory_bytes=24 * 1024**3,
        available_memory_gb=24.0,
        theoretical_bandwidth_gbps=1000.0,
        theoretical_tflops_fp16=300.0,
        os_platform="Linux",
        python_version="3.12.0",
        torch_version="2.2.0",
    )


def test_scoring_calibration(base_device):
    decode_res = DecodeBenchmarkResult(
        peak_bandwidth_gbps=800.0,
        sustained_bandwidth_gbps=800.0,
        gemv_peak_bandwidth_gbps=800.0,
        kv_streaming_peak_bandwidth_gbps=750.0,
        theoretical_bandwidth_gbps=1000.0,
        bandwidth_saturation_pct=80.0,
        points=[],
    )

    prefill_res = PrefillBenchmarkResult(
        peak_tflops=165.0,
        sustained_tflops=165.0,
        peak_tokens_per_sec=50000.0,
        theoretical_tflops_fp16=330.0,
        tflops_efficiency_pct=50.0,
        points=[],
    )

    attn_res = AttentionBenchmarkResult(
        sdpa_peak_tflops=25.0,
        naive_peak_tflops=10.0,
        peak_speedup=2.5,
        mean_speedup=2.5,
        points=[],
    )

    scores = calculate_siliconmark_scores(base_device, decode_res, prefill_res, attn_res)

    assert scores.decode_score == 10000
    assert scores.prefill_score == 10000
    assert scores.attention_score == 10000
    assert scores.overall_score == 10000
    assert scores.relative_to_rtx4090_pct == 100.0
    assert scores.efficiency_grade == "S+"
    assert "Tier 2" in scores.tier_classification


def test_scoring_datacenter_tier1(base_device):
    decode_res = DecodeBenchmarkResult(
        peak_bandwidth_gbps=3000.0,
        sustained_bandwidth_gbps=2500.0,
        gemv_peak_bandwidth_gbps=3000.0,
        kv_streaming_peak_bandwidth_gbps=2800.0,
        bandwidth_saturation_pct=75.0,
        points=[],
    )
    prefill_res = PrefillBenchmarkResult(
        peak_tflops=800.0,
        sustained_tflops=700.0,
        peak_tokens_per_sec=200000.0,
        points=[],
    )
    scores = calculate_siliconmark_scores(base_device, decode_res, prefill_res, None)
    assert scores.overall_score >= 25000
    assert "Tier 1" in scores.tier_classification
    assert scores.efficiency_grade == "S"


def test_scoring_grades_and_tiers(base_device):
    # Tier 3
    decode_res = DecodeBenchmarkResult(
        peak_bandwidth_gbps=500.0,
        sustained_bandwidth_gbps=450.0,
        gemv_peak_bandwidth_gbps=500.0,
        kv_streaming_peak_bandwidth_gbps=450.0,
        bandwidth_saturation_pct=60.0,
        points=[],
    )
    prefill_res = PrefillBenchmarkResult(
        peak_tflops=80.0,
        sustained_tflops=70.0,
        peak_tokens_per_sec=20000.0,
        points=[],
    )
    attn_res = AttentionBenchmarkResult(
        sdpa_peak_tflops=10.0,
        naive_peak_tflops=5.0,
        peak_speedup=2.0,
        mean_speedup=2.0,
        points=[],
    )
    scores = calculate_siliconmark_scores(base_device, decode_res, prefill_res, attn_res)
    assert "Tier 3" in scores.tier_classification
    assert scores.efficiency_grade == "A"

    # Tier 4
    decode_res.sustained_bandwidth_gbps = 150.0
    decode_res.bandwidth_saturation_pct = 45.0
    prefill_res.sustained_tflops = 15.0
    scores = calculate_siliconmark_scores(base_device, decode_res, prefill_res, None)
    assert "Tier 4" in scores.tier_classification
    assert scores.efficiency_grade == "B"

    # Tier 5
    decode_res.sustained_bandwidth_gbps = 20.0
    decode_res.bandwidth_saturation_pct = 20.0
    prefill_res.sustained_tflops = 1.0
    scores = calculate_siliconmark_scores(base_device, decode_res, prefill_res, None)
    assert "Tier 5" in scores.tier_classification
    assert scores.efficiency_grade == "D"

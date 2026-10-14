"""Unit tests for the Roofline model and operational intensity analyzer."""

import pytest
from siliconmark.core.config import DeviceInfo, DeviceType
from siliconmark.core.roofline import (
    calculate_roofline,
    generate_ascii_roofline,
    generate_roofline_svg,
)


@pytest.fixture
def sample_gpu():
    return DeviceInfo(
        device_type=DeviceType.CUDA,
        device_name="RTX 4090 Test",
        total_memory_bytes=24 * 1024**3,
        total_memory_gb=24.0,
        available_memory_bytes=24 * 1024**3,
        available_memory_gb=24.0,
        theoretical_bandwidth_gbps=1000.0,
        theoretical_tflops_fp16=300.0,
        theoretical_tflops_fp32=80.0,
        os_platform="Linux",
        python_version="3.12.0",
        torch_version="2.3.0",
    )


def test_roofline_machine_balance(sample_gpu):
    summary = calculate_roofline(sample_gpu)
    # Peak GFLOPS = 300,000. Peak BW = 1,000. Balance = 300,000 / 1,000 = 300.0 FLOPs/Byte
    assert summary.machine_balance_flops_per_byte == 300.0
    assert summary.peak_compute_tflops == 300.0
    assert summary.peak_bandwidth_gbps == 1000.0


def test_roofline_points_classification(sample_gpu):
    points_data = [
        # Memory-bound workload: GEMV intensity 1.5 FLOPs/Byte
        {
            "name": "gemv_decode",
            "workload_type": "decode",
            "total_flops": 1500000.0,
            "bytes_transferred": 1000000.0,
            "time_sec": 0.001,
        },
        # Compute-bound workload: Dense GEMM intensity 500 FLOPs/Byte (> 300)
        {
            "name": "dense_prefill",
            "workload_type": "prefill",
            "total_flops": 500000000.0,
            "bytes_transferred": 1000000.0,
            "time_sec": 0.002,
        },
    ]

    summary = calculate_roofline(sample_gpu, points_data=points_data)
    assert len(summary.points) == 2
    assert summary.memory_bound_count == 1
    assert summary.compute_bound_count == 1

    pt1 = summary.points[0]
    assert pt1.name == "gemv_decode"
    assert pt1.is_memory_bound is True
    assert pt1.operational_intensity == 1.5

    pt2 = summary.points[1]
    assert pt2.name == "dense_prefill"
    assert pt2.is_memory_bound is False
    assert pt2.operational_intensity == 500.0


def test_ascii_and_svg_generation(sample_gpu):
    summary = calculate_roofline(sample_gpu)
    ascii_out = generate_ascii_roofline(summary)
    assert "ROOFLINE MODEL SUMMARY" in ascii_out
    assert "Machine Balance:" in ascii_out

    svg_out = generate_roofline_svg(summary)
    assert "<svg" in svg_out
    assert "</svg>" in svg_out
    assert "Operational Intensity" in svg_out

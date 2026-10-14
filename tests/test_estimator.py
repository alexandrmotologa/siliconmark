"""Unit tests for LLM inference and sizing estimator."""

import pytest
from siliconmark.core.config import DeviceInfo, DeviceType
from siliconmark.core.estimator import (
    MODEL_CATALOG,
    estimate_model_performance,
)


@pytest.fixture
def gpu_device():
    return DeviceInfo(
        device_type=DeviceType.CUDA,
        device_name="RTX 4090",
        total_memory_bytes=24 * 1024**3,
        total_memory_gb=24.0,
        available_memory_bytes=24 * 1024**3,
        available_memory_gb=24.0,
        theoretical_bandwidth_gbps=1008.0,
        theoretical_tflops_fp16=330.0,
        os_platform="Linux",
        python_version="3.12.0",
        torch_version="2.3.0",
    )


def test_model_catalog_coverage():
    assert "llama-3-8b" in MODEL_CATALOG
    assert "llama-3-70b" in MODEL_CATALOG
    assert "qwen-2.5-32b" in MODEL_CATALOG
    assert "deepseek-v3-moe" in MODEL_CATALOG
    assert MODEL_CATALOG["llama-3-8b"].num_layers == 32


def test_estimator_llama3_8b_fits_vram(gpu_device):
    est = estimate_model_performance(
        model_key="llama-3-8b",
        quant="q4_k_m",
        context_tokens=4096,
        batch_size=1,
        achieved_decode_bw_gbps=800.0,
        achieved_prefill_tflops=165.0,
        device_info=gpu_device,
    )

    assert est.can_fit is True
    assert est.model_weights_gb < 5.0
    assert est.decode_tokens_per_sec > 100.0
    assert est.inter_token_latency_ms < 10.0
    assert est.vram_headroom_gb > 15.0


def test_estimator_llama3_70b_fp16_exceeds_vram(gpu_device):
    # 70B in FP16 requires ~140 GB weights, exceeds 24GB RTX 4090
    est = estimate_model_performance(
        model_key="llama-3-70b",
        quant="fp16",
        context_tokens=4096,
        batch_size=1,
        achieved_decode_bw_gbps=800.0,
        achieved_prefill_tflops=165.0,
        device_info=gpu_device,
    )

    assert est.can_fit is False
    assert est.vram_headroom_gb < 0.0
    assert "Exceeds memory" in est.fit_assessment

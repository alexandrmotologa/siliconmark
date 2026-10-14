"""LLM inference throughput and memory footprint estimator.

Translates empirical bare-metal hardware benchmark results (decode memory bandwidth
and prefill TFLOPS) into expected real-world LLM token generation rates across popular
model architectures and quantization levels.
"""

from __future__ import annotations

from typing import Dict, List, Optional
from pydantic import BaseModel, Field

from siliconmark.core.config import DeviceInfo


class ModelArchitecture(BaseModel):
    """Specification of an open-weight LLM architecture."""
    name: str
    family: str
    total_params_billion: float
    active_params_billion: float
    num_layers: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    hidden_dim: int
    default_context: int = 4096


# Standard catalog of popular open LLM architectures
MODEL_CATALOG: Dict[str, ModelArchitecture] = {
    "llama-3-8b": ModelArchitecture(
        name="Llama-3-8B",
        family="Meta Llama 3",
        total_params_billion=8.03,
        active_params_billion=8.03,
        num_layers=32,
        num_heads=32,
        num_kv_heads=8,
        head_dim=128,
        hidden_dim=4096,
        default_context=8192,
    ),
    "llama-3-70b": ModelArchitecture(
        name="Llama-3-70B",
        family="Meta Llama 3",
        total_params_billion=70.6,
        active_params_billion=70.6,
        num_layers=80,
        num_heads=64,
        num_kv_heads=8,
        head_dim=128,
        hidden_dim=8192,
        default_context=8192,
    ),
    "qwen-2.5-7b": ModelArchitecture(
        name="Qwen-2.5-7B",
        family="Qwen 2.5",
        total_params_billion=7.61,
        active_params_billion=7.61,
        num_layers=28,
        num_heads=28,
        num_kv_heads=4,
        head_dim=128,
        hidden_dim=3584,
        default_context=32768,
    ),
    "qwen-2.5-14b": ModelArchitecture(
        name="Qwen-2.5-14B",
        family="Qwen 2.5",
        total_params_billion=14.7,
        active_params_billion=14.7,
        num_layers=48,
        num_heads=40,
        num_kv_heads=8,
        head_dim=128,
        hidden_dim=5120,
        default_context=32768,
    ),
    "qwen-2.5-32b": ModelArchitecture(
        name="Qwen-2.5-32B",
        family="Qwen 2.5",
        total_params_billion=32.5,
        active_params_billion=32.5,
        num_layers=64,
        num_heads=40,
        num_kv_heads=8,
        head_dim=128,
        hidden_dim=5120,
        default_context=32768,
    ),
    "mistral-7b": ModelArchitecture(
        name="Mistral-7B-v0.3",
        family="Mistral",
        total_params_billion=7.25,
        active_params_billion=7.25,
        num_layers=32,
        num_heads=32,
        num_kv_heads=8,
        head_dim=128,
        hidden_dim=4096,
        default_context=32768,
    ),
    "mixtral-8x7b": ModelArchitecture(
        name="Mixtral-8x7B (MoE)",
        family="Mistral MoE",
        total_params_billion=46.7,
        active_params_billion=12.9,
        num_layers=32,
        num_heads=32,
        num_kv_heads=8,
        head_dim=128,
        hidden_dim=4096,
        default_context=32768,
    ),
    "deepseek-v3-moe": ModelArchitecture(
        name="DeepSeek-V3 (MoE 671B)",
        family="DeepSeek",
        total_params_billion=671.0,
        active_params_billion=37.0,
        num_layers=61,
        num_heads=128,
        num_kv_heads=16,
        head_dim=128,
        hidden_dim=7168,
        default_context=65536,
    ),
    "gemma-2-9b": ModelArchitecture(
        name="Gemma-2-9B",
        family="Google Gemma",
        total_params_billion=9.24,
        active_params_billion=9.24,
        num_layers=42,
        num_heads=16,
        num_kv_heads=8,
        head_dim=256,
        hidden_dim=3584,
        default_context=8192,
    ),
    "phi-3-mini": ModelArchitecture(
        name="Phi-3-Mini (3.8B)",
        family="Microsoft Phi",
        total_params_billion=3.82,
        active_params_billion=3.82,
        num_layers=32,
        num_heads=32,
        num_kv_heads=32,
        head_dim=96,
        hidden_dim=3072,
        default_context=4096,
    ),
}

# Bytes per parameter for each quantization format
QUANTIZATION_BYTES: Dict[str, float] = {
    "fp16": 2.0,
    "bf16": 2.0,
    "q8_0": 1.0625,
    "q6_k": 0.82,
    "q4_k_m": 0.5625,
    "q3_k_m": 0.43,
    "q2_k": 0.32,
}


class ModelInferenceEstimate(BaseModel):
    """Estimated performance metrics for a specific model configuration."""
    model_id: str
    model_name: str
    quantization: str
    context_tokens: int
    batch_size: int

    # Memory breakdown
    model_weights_gb: float
    kv_cache_gb: float
    total_vram_required_gb: float
    available_vram_gb: float
    vram_headroom_gb: float
    can_fit: bool
    fit_assessment: str

    # Performance predictions
    decode_tokens_per_sec: float = Field(
        description="Predicted single-stream token generation speed based on measured decode memory bandwidth"
    )
    prefill_tokens_per_sec: float = Field(
        description="Predicted prompt processing throughput based on measured prefill TFLOPS"
    )
    time_to_first_token_sec: float = Field(
        description="Estimated latency before first token generates for context_tokens prompt"
    )
    inter_token_latency_ms: float = Field(
        description="Average millisecond delay between generated tokens (1000 / decode_tok_per_sec)"
    )


def estimate_model_performance(
    model_key: str,
    quant: str = "q4_k_m",
    context_tokens: int = 4096,
    batch_size: int = 1,
    achieved_decode_bw_gbps: float = 30.0,
    achieved_prefill_tflops: float = 0.5,
    device_info: Optional[DeviceInfo] = None,
) -> ModelInferenceEstimate:
    """Calculate predicted VRAM usage and token generation throughput for an LLM.

    Args:
        model_key: Key in MODEL_CATALOG, e.g. 'llama-3-8b'
        quant: Quantization level, e.g. 'q4_k_m', 'q8_0', 'fp16'
        context_tokens: Total context length in tokens
        batch_size: Number of concurrent streams
        achieved_decode_bw_gbps: Measured sustained decode bandwidth in GB/s
        achieved_prefill_tflops: Measured sustained prefill TFLOPS
        device_info: Hardware specifications for VRAM capacity check

    Returns:
        ModelInferenceEstimate with memory requirements and throughput numbers.
    """
    arch = MODEL_CATALOG.get(model_key.lower())
    if not arch:
        arch = MODEL_CATALOG["llama-3-8b"]

    bytes_per_param = QUANTIZATION_BYTES.get(quant.lower(), 0.5625)

    # 1. Weights memory size (GB)
    weights_bytes = arch.total_params_billion * 1e9 * bytes_per_param
    weights_gb = round(weights_bytes / (1024 ** 3), 2)

    # 2. KV Cache memory size (GB)
    # 2 * layers * kv_heads * head_dim * context_tokens * 2 bytes (FP16 KV) * batch_size
    kv_bytes = (
        2
        * arch.num_layers
        * arch.num_kv_heads
        * arch.head_dim
        * context_tokens
        * 2.0  # FP16 KV cache standard
        * batch_size
    )
    kv_gb = round(kv_bytes / (1024 ** 3), 2)

    # Add 0.5 GB runtime context/activation buffer
    runtime_overhead_gb = 0.5
    total_req_gb = round(weights_gb + kv_gb + runtime_overhead_gb, 2)

    avail_vram_gb = device_info.available_memory_gb if device_info else 16.0
    vram_headroom = round(avail_vram_gb - total_req_gb, 2)
    can_fit = vram_headroom >= 0.0

    if vram_headroom > 4.0:
        fit_assessment = "Comfortable fit in physical memory"
    elif vram_headroom >= 0.0:
        fit_assessment = "Tight fit (near memory capacity)"
    else:
        fit_assessment = f"Exceeds memory by {abs(vram_headroom):.1f} GB (will spill to system RAM / paging)"

    # 3. Decode token generation throughput (tokens/sec)
    # Memory bound: Every generated token reads model weights + active KV cache
    # MoE models only read active parameters + routing weights
    effective_read_gb = (
        (arch.active_params_billion * 1e9 * bytes_per_param + kv_bytes / max(context_tokens, 1))
        / (1024 ** 3)
    )
    decode_tok_sec = round(max(0.1, achieved_decode_bw_gbps / max(effective_read_gb, 0.01)), 1)
    inter_token_ms = round(1000.0 / decode_tok_sec, 1)

    # 4. Prefill token processing throughput (tokens/sec)
    # Compute bound: 2 * active_params * tokens FLOPs
    flops_per_token = 2.0 * arch.active_params_billion * 1e9
    prefill_tok_sec = round(
        max(1.0, (achieved_prefill_tflops * 1e12) / max(flops_per_token, 1.0)), 1
    )
    ttft_sec = round(context_tokens / max(prefill_tok_sec, 1.0), 2)

    return ModelInferenceEstimate(
        model_id=model_key,
        model_name=arch.name,
        quantization=quant.upper(),
        context_tokens=context_tokens,
        batch_size=batch_size,
        model_weights_gb=weights_gb,
        kv_cache_gb=kv_gb,
        total_vram_required_gb=total_req_gb,
        available_vram_gb=avail_vram_gb,
        vram_headroom_gb=vram_headroom,
        can_fit=can_fit,
        fit_assessment=fit_assessment,
        decode_tokens_per_sec=decode_tok_sec,
        prefill_tokens_per_sec=prefill_tok_sec,
        time_to_first_token_sec=ttft_sec,
        inter_token_latency_ms=inter_token_ms,
    )

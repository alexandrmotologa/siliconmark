"""Core Pydantic schemas and configuration models for SiliconMark.

These models define device metadata, benchmark configuration, metrics,
iteration statistics, and composite score outputs.
"""

from __future__ import annotations

from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field


class DeviceType(str, Enum):
    """Supported hardware compute acceleration backends."""
    CUDA = "cuda"
    MPS = "mps"
    ROCM = "rocm"
    CPU = "cpu"
    XPU = "xpu"
    DIRECTML = "directml"


class Precision(str, Enum):
    """Tensor numerical precision."""
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    INT8 = "int8"


class DeviceInfo(BaseModel):
    """Bare-metal device hardware characteristics and theoretical limits."""
    device_type: DeviceType
    device_name: str
    device_index: int = 0
    total_memory_bytes: int
    total_memory_gb: float
    available_memory_bytes: int
    available_memory_gb: float
    compute_units: Optional[int] = Field(
        default=None,
        description="Streaming Multiprocessors (CUDA), GPU Cores (Metal/MPS), or Physical CPU Cores",
    )
    logical_cores: Optional[int] = Field(
        default=None,
        description="Logical CPU threads or execution units",
    )
    clock_mhz: Optional[float] = Field(
        default=None,
        description="Core clock frequency in MHz",
    )
    memory_bus_width_bits: Optional[int] = Field(
        default=None,
        description="Memory bus width in bits (e.g. 384-bit on RTX 4090)",
    )
    memory_clock_mhz: Optional[float] = Field(
        default=None,
        description="Memory clock in MHz",
    )
    theoretical_bandwidth_gbps: Optional[float] = Field(
        default=None,
        description="Theoretical peak memory bandwidth in GB/s",
    )
    theoretical_tflops_fp16: Optional[float] = Field(
        default=None,
        description="Theoretical peak half-precision tensor compute in TFLOPS",
    )
    theoretical_tflops_fp32: Optional[float] = Field(
        default=None,
        description="Theoretical peak single-precision compute in TFLOPS",
    )
    cuda_compute_capability: Optional[str] = None
    driver_version: Optional[str] = None
    os_platform: str
    python_version: str
    torch_version: str


class IterationStats(BaseModel):
    """Statistical distribution of timing measurements across iterations."""
    iterations: int
    mean_ms: float
    median_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    p95_ms: float
    raw_latencies_ms: List[float] = Field(default_factory=list)


class DecodeBenchmarkConfig(BaseModel):
    """Configuration for LLM decode memory bandwidth benchmark."""
    batch_sizes: List[int] = Field(
        default_factory=lambda: [1, 2, 4, 8, 16, 32],
        description="Batch sizes representing active concurrent decode streams",
    )
    hidden_dims: List[int] = Field(
        default_factory=lambda: [4096, 8192],
        description="Model hidden dimensions (LLaMA-7B/8B: 4096, LLaMA-70B: 8192)",
    )
    seq_lens: List[int] = Field(
        default_factory=lambda: [512, 2048, 4096],
        description="Context lengths for KV cache traversal",
    )
    precisions: List[Precision] = Field(
        default_factory=lambda: [Precision.FP16, Precision.FP32],
        description="Precisions tested",
    )
    warmup_iters: int = Field(default=10, ge=1)
    bench_iters: int = Field(default=30, ge=3)
    include_paged_kv: bool = Field(
        default=True,
        description="Benchmark discontinuous paged KV cache lookups alongside contiguous streaming",
    )
    page_block_size: int = Field(default=16, description="Block size in tokens for paged KV cache")


class PrefillBenchmarkConfig(BaseModel):
    """Configuration for LLM prefill TFLOPS benchmark."""
    batch_sizes: List[int] = Field(default_factory=lambda: [1, 4])
    prompt_lens: List[int] = Field(default_factory=lambda: [256, 512, 1024, 2048, 4096])
    hidden_dims: List[int] = Field(default_factory=lambda: [4096, 8192])
    precisions: List[Precision] = Field(default_factory=lambda: [Precision.FP16, Precision.FP32])
    warmup_iters: int = Field(default=10, ge=1)
    bench_iters: int = Field(default=25, ge=3)


class AttentionBenchmarkConfig(BaseModel):
    """Configuration for attention kernel efficiency benchmark."""
    seq_lens: List[int] = Field(default_factory=lambda: [512, 1024, 2048, 4096, 8192])
    num_heads: int = 32
    head_dim: int = 128
    batch_size: int = 1
    num_kv_heads: Optional[int] = Field(
        default=8,
        description="Key/Value heads for Grouped-Query Attention (GQA). None for standard MHA.",
    )
    precisions: List[Precision] = Field(default_factory=lambda: [Precision.FP16, Precision.FP32])
    warmup_iters: int = Field(default=10, ge=1)
    bench_iters: int = Field(default=30, ge=3)


class BenchmarkRunConfig(BaseModel):
    """Global configuration for a full benchmark execution."""
    device: str = "auto"
    decode: DecodeBenchmarkConfig = Field(default_factory=DecodeBenchmarkConfig)
    prefill: PrefillBenchmarkConfig = Field(default_factory=PrefillBenchmarkConfig)
    attention: AttentionBenchmarkConfig = Field(default_factory=AttentionBenchmarkConfig)
    quick_mode: bool = Field(
        default=False,
        description="Run abbreviated benchmark matrix with reduced iteration counts",
    )


class DecodePointResult(BaseModel):
    """Single benchmark measurement point in decode bandwidth evaluation."""
    workload: str = Field(
        description="gemv_weight_streaming, kv_cache_streaming, or paged_kv_lookup"
    )
    batch_size: int
    hidden_dim: int
    seq_len: int
    precision: Precision
    bytes_transferred: int
    stats: IterationStats
    achieved_bandwidth_gbps: float
    saturation_pct: Optional[float] = None


class DecodeBenchmarkResult(BaseModel):
    """Aggregated results for decode memory bandwidth benchmark."""
    peak_bandwidth_gbps: float
    sustained_bandwidth_gbps: float
    gemv_peak_bandwidth_gbps: float
    kv_streaming_peak_bandwidth_gbps: float
    paged_kv_peak_bandwidth_gbps: Optional[float] = None
    theoretical_bandwidth_gbps: Optional[float] = None
    bandwidth_saturation_pct: Optional[float] = None
    points: List[DecodePointResult]
    summary_notes: str = ""


class PrefillPointResult(BaseModel):
    """Single benchmark measurement point in prefill TFLOPS evaluation."""
    batch_size: int
    prompt_len: int
    hidden_dim: int
    precision: Precision
    total_flops: int
    stats: IterationStats
    achieved_tflops: float
    tokens_per_sec: float
    tflops_efficiency_pct: Optional[float] = None


class PrefillBenchmarkResult(BaseModel):
    """Aggregated results for prefill TFLOPS benchmark."""
    peak_tflops: float
    sustained_tflops: float
    peak_tokens_per_sec: float
    theoretical_tflops_fp16: Optional[float] = None
    tflops_efficiency_pct: Optional[float] = None
    points: List[PrefillPointResult]
    summary_notes: str = ""


class AttentionPointResult(BaseModel):
    """Single benchmark measurement point in attention kernel efficiency evaluation."""
    seq_len: int
    num_heads: int
    head_dim: int
    batch_size: int
    precision: Precision
    kernel_type: str = Field(description="sdpa (Flash/Memory-Efficient) or naive")
    total_flops: int
    stats: IterationStats
    achieved_tflops: float
    tokens_per_sec: float
    speedup_over_naive: Optional[float] = None


class AttentionBenchmarkResult(BaseModel):
    """Aggregated results for attention kernel efficiency benchmark."""
    sdpa_peak_tflops: float
    naive_peak_tflops: float
    peak_speedup: float
    mean_speedup: float
    points: List[AttentionPointResult]
    summary_notes: str = ""


class SiliconMarkScores(BaseModel):
    """Composite normalized benchmark scores."""
    overall_score: int
    decode_score: int
    prefill_score: int
    attention_score: int
    efficiency_grade: str
    tier_classification: str
    relative_to_rtx4090_pct: float
    relative_to_m3max_pct: float


class SiliconMarkRunResult(BaseModel):
    """Full execution report of SiliconMark suite."""
    run_id: str
    timestamp: str
    device_info: DeviceInfo
    decode_result: Optional[DecodeBenchmarkResult] = None
    prefill_result: Optional[PrefillBenchmarkResult] = None
    attention_result: Optional[AttentionBenchmarkResult] = None
    scores: Optional[SiliconMarkScores] = None

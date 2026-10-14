"""Decode memory bandwidth saturation benchmark.

Measures hardware memory bandwidth during simulated auto-regressive decode:
1. GEMV Projection (Weight Streaming bound at B=1..32)
2. KV Cache Buffer Streaming (Sequential multi-head attention traversal)
3. Paged KV Cache Block Retrieval (Discontinuous page lookup, vLLM pattern)
"""

from __future__ import annotations

import math
from typing import Callable, List, Optional
import torch

from siliconmark.benchmarks.base import BaseBenchmark
from siliconmark.core.config import (
    DecodeBenchmarkConfig,
    DecodeBenchmarkResult,
    DecodePointResult,
    Precision,
)
from siliconmark.core.timing import benchmark_function, synchronize_device


class DecodeMemoryBandwidthBenchmark(BaseBenchmark):
    """Measures memory bus saturation during LLM auto-regressive decoding."""

    def run(self, config: Optional[DecodeBenchmarkConfig] = None) -> DecodeBenchmarkResult:
        """Run the decode memory bandwidth suite across batch sizes and configurations."""
        if config is None:
            config = DecodeBenchmarkConfig()

        results: List[DecodePointResult] = []

        # 1. Benchmark GEMV Weight Streaming across batch sizes & dimensions
        for prec in config.precisions:
            dtype = self.get_torch_dtype(prec.value)
            dtype_bytes = torch.tensor([], dtype=dtype).element_size()

            for hidden_dim in config.hidden_dims:
                # Weights: [hidden_dim, hidden_dim]
                weights = torch.randn(hidden_dim, hidden_dim, device=self.device, dtype=dtype)

                for b in config.batch_sizes:
                    # In decode, x has shape [b, hidden_dim]
                    x = torch.randn(b, hidden_dim, device=self.device, dtype=dtype)

                    def gemv_kernel() -> None:
                        # y = x @ weights.T -> shape [b, hidden_dim]
                        _ = torch.matmul(x, weights.t())

                    stats = benchmark_function(
                        gemv_kernel,
                        device=self.device,
                        warmup_iters=config.warmup_iters,
                        bench_iters=config.bench_iters,
                    )

                    # Exact physical memory traffic:
                    # Weights read once: hidden_dim * hidden_dim * dtype_bytes
                    # Input vector read: b * hidden_dim * dtype_bytes
                    # Output vector written: b * hidden_dim * dtype_bytes
                    bytes_moved = int(
                        (hidden_dim * hidden_dim + b * hidden_dim + b * hidden_dim) * dtype_bytes
                    )
                    time_sec = (stats.median_ms / 1000.0) if stats.median_ms > 0 else 1e-9
                    achieved_gbps = round((bytes_moved / time_sec) / 1e9, 2)

                    sat_pct = None
                    if self.device_info.theoretical_bandwidth_gbps:
                        sat_pct = round(
                            (achieved_gbps / self.device_info.theoretical_bandwidth_gbps) * 100.0, 1
                        )

                    results.append(
                        DecodePointResult(
                            workload="gemv_weight_streaming",
                            batch_size=b,
                            hidden_dim=hidden_dim,
                            seq_len=1,
                            precision=prec,
                            bytes_transferred=bytes_moved,
                            stats=stats,
                            achieved_bandwidth_gbps=achieved_gbps,
                            saturation_pct=sat_pct,
                        )
                    )

                del weights

        # 2. Benchmark KV Cache Streaming
        # Simulates streaming past keys & values during attention computation
        num_heads = 32
        head_dim = 128
        for prec in config.precisions:
            dtype = self.get_torch_dtype(prec.value)
            dtype_bytes = torch.tensor([], dtype=dtype).element_size()

            for seq_len in config.seq_lens:
                for b in [1, 4]:
                    # KV Cache shape: [b, num_heads, seq_len, head_dim]
                    key_cache = torch.randn(b, num_heads, seq_len, head_dim, device=self.device, dtype=dtype)
                    val_cache = torch.randn(b, num_heads, seq_len, head_dim, device=self.device, dtype=dtype)
                    query = torch.randn(b, num_heads, 1, head_dim, device=self.device, dtype=dtype)

                    def kv_stream_kernel() -> None:
                        # Compute attention scores over KV cache
                        attn_weights = torch.matmul(query, key_cache.transpose(-1, -2))
                        _ = torch.matmul(attn_weights, val_cache)

                    stats = benchmark_function(
                        kv_stream_kernel,
                        device=self.device,
                        warmup_iters=config.warmup_iters,
                        bench_iters=config.bench_iters,
                    )

                    # Bytes read: Key cache + Value cache + Query + Attn weights buffer
                    bytes_moved = int(
                        (2 * b * num_heads * seq_len * head_dim + b * num_heads * head_dim) * dtype_bytes
                    )
                    time_sec = (stats.median_ms / 1000.0) if stats.median_ms > 0 else 1e-9
                    achieved_gbps = round((bytes_moved / time_sec) / 1e9, 2)

                    sat_pct = None
                    if self.device_info.theoretical_bandwidth_gbps:
                        sat_pct = round(
                            (achieved_gbps / self.device_info.theoretical_bandwidth_gbps) * 100.0, 1
                        )

                    results.append(
                        DecodePointResult(
                            workload="kv_cache_streaming",
                            batch_size=b,
                            hidden_dim=num_heads * head_dim,
                            seq_len=seq_len,
                            precision=prec,
                            bytes_transferred=bytes_moved,
                            stats=stats,
                            achieved_bandwidth_gbps=achieved_gbps,
                            saturation_pct=sat_pct,
                        )
                    )

                    del key_cache, val_cache, query

        # 3. Benchmark Paged KV Cache Lookup (vLLM style block table lookups)
        if config.include_paged_kv:
            dtype = self.get_torch_dtype(config.precisions[0].value)
            dtype_bytes = torch.tensor([], dtype=dtype).element_size()
            block_size = config.page_block_size
            total_blocks = 512

            # Block pool: [total_blocks, num_heads, block_size, head_dim]
            block_pool = torch.randn(total_blocks, num_heads, block_size, head_dim, device=self.device, dtype=dtype)
            # Simulating block lookup index
            block_indices = torch.randint(0, total_blocks, (128,), device=self.device, dtype=torch.long)

            def paged_kv_kernel() -> None:
                # Discontinuous index gather simulating paged attention memory access
                _ = torch.index_select(block_pool, 0, block_indices)

            stats = benchmark_function(
                paged_kv_kernel,
                device=self.device,
                warmup_iters=config.warmup_iters,
                bench_iters=config.bench_iters,
            )

            bytes_moved = int(128 * num_heads * block_size * head_dim * dtype_bytes)
            time_sec = (stats.median_ms / 1000.0) if stats.median_ms > 0 else 1e-9
            achieved_gbps = round((bytes_moved / time_sec) / 1e9, 2)

            sat_pct = None
            if self.device_info.theoretical_bandwidth_gbps:
                sat_pct = round(
                    (achieved_gbps / self.device_info.theoretical_bandwidth_gbps) * 100.0, 1
                )

            results.append(
                DecodePointResult(
                    workload="paged_kv_lookup",
                    batch_size=1,
                    hidden_dim=num_heads * head_dim,
                    seq_len=128 * block_size,
                    precision=config.precisions[0],
                    bytes_transferred=bytes_moved,
                    stats=stats,
                    achieved_bandwidth_gbps=achieved_gbps,
                    saturation_pct=sat_pct,
                )
            )

            del block_pool, block_indices

        # Summary calculations
        all_bandwidths = [r.achieved_bandwidth_gbps for r in results]
        peak_bw = max(all_bandwidths) if all_bandwidths else 0.0
        sustained_bw = round(sum(all_bandwidths) / len(all_bandwidths), 2) if all_bandwidths else 0.0

        gemv_bws = [r.achieved_bandwidth_gbps for r in results if r.workload == "gemv_weight_streaming"]
        gemv_peak = max(gemv_bws) if gemv_bws else 0.0

        kv_bws = [r.achieved_bandwidth_gbps for r in results if r.workload == "kv_cache_streaming"]
        kv_peak = max(kv_bws) if kv_bws else 0.0

        paged_bws = [r.achieved_bandwidth_gbps for r in results if r.workload == "paged_kv_lookup"]
        paged_peak = max(paged_bws) if paged_bws else None

        sat_pct_overall = None
        if self.device_info.theoretical_bandwidth_gbps and self.device_info.theoretical_bandwidth_gbps > 0:
            sat_pct_overall = round(
                (peak_bw / self.device_info.theoretical_bandwidth_gbps) * 100.0, 1
            )

        summary = (
            f"Peak decode memory bandwidth: {peak_bw:.2f} GB/s "
            f"(Sustained: {sustained_bw:.2f} GB/s). "
        )
        if sat_pct_overall:
            summary += f"Hardware bus saturation: {sat_pct_overall:.1f}% of theoretical peak ({self.device_info.theoretical_bandwidth_gbps:.1f} GB/s)."

        return DecodeBenchmarkResult(
            peak_bandwidth_gbps=peak_bw,
            sustained_bandwidth_gbps=sustained_bw,
            gemv_peak_bandwidth_gbps=gemv_peak,
            kv_streaming_peak_bandwidth_gbps=kv_peak,
            paged_kv_peak_bandwidth_gbps=paged_peak,
            theoretical_bandwidth_gbps=self.device_info.theoretical_bandwidth_gbps,
            bandwidth_saturation_pct=sat_pct_overall,
            points=results,
            summary_notes=summary,
        )

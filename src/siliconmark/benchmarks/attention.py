"""Attention kernel efficiency benchmark.

Measures scaled dot-product attention (SDPA) performance:
1. PyTorch SDPA (dispatches to FlashAttention-2, Memory-Efficient Attention, or Vector Math)
2. Comparison against standard unfused attention (GEMM + Softmax + GEMM)
3. Sequence length scaling and Grouped-Query Attention (GQA) speedup.
"""

from __future__ import annotations

import math
from typing import List, Optional
import torch

from siliconmark.benchmarks.base import BaseBenchmark
from siliconmark.core.config import (
    AttentionBenchmarkConfig,
    AttentionBenchmarkResult,
    AttentionPointResult,
    Precision,
)
from siliconmark.core.timing import benchmark_function


class AttentionKernelBenchmark(BaseBenchmark):
    """Measures attention kernel efficiency, memory scaling, and SDPA acceleration."""

    def run(self, config: Optional[AttentionBenchmarkConfig] = None) -> AttentionBenchmarkResult:
        """Run attention benchmark matrix comparing native SDPA vs unfused execution."""
        if config is None:
            config = AttentionBenchmarkConfig()

        results: List[AttentionPointResult] = []

        b = config.batch_size
        h_q = config.num_heads
        h_kv = min(config.num_kv_heads, h_q) if config.num_kv_heads is not None else h_q
        d = config.head_dim

        for prec in config.precisions:
            dtype = self.get_torch_dtype(prec.value)

            for s in config.seq_lens:
                # Total FLOPs for attention: 2 * (B * H * S * S * D) + 2 * (B * H * S * D * S)
                # = 4 * B * H * S^2 * D
                total_flops = int(4 * b * h_q * s * s * d)

                # Initialize Query, Key, Value tensors
                # Q: [B, H_q, S, D]
                # K, V: [B, H_kv, S, D]
                q = torch.randn(b, h_q, s, d, device=self.device, dtype=dtype)
                k = torch.randn(b, h_kv, s, d, device=self.device, dtype=dtype)
                v = torch.randn(b, h_kv, s, d, device=self.device, dtype=dtype)

                # If GQA is active (H_kv < H_q), repeat K and V along head dim for naive attention
                if h_kv < h_q:
                    repeat_factor = max(1, h_q // h_kv)
                    k_expanded = k.repeat_interleave(repeat_factor, dim=1)
                    v_expanded = v.repeat_interleave(repeat_factor, dim=1)
                    if k_expanded.shape[1] < h_q:
                        diff = h_q - k_expanded.shape[1]
                        k_expanded = torch.cat([k_expanded, k[:, :diff]], dim=1)
                        v_expanded = torch.cat([v_expanded, v[:, :diff]], dim=1)
                else:
                    k_expanded = k
                    v_expanded = v

                scale = 1.0 / math.sqrt(d)

                # 1. Unfused Naive Attention (Baseline)
                def naive_attention_kernel() -> None:
                    scores = torch.matmul(q, k_expanded.transpose(-2, -1)) * scale
                    probs = torch.softmax(scores, dim=-1)
                    _ = torch.matmul(probs, v_expanded)

                naive_stats = benchmark_function(
                    naive_attention_kernel,
                    device=self.device,
                    warmup_iters=config.warmup_iters,
                    bench_iters=config.bench_iters,
                )

                naive_time_sec = (naive_stats.median_ms / 1000.0) if naive_stats.median_ms > 0 else 1e-9
                naive_tflops = round((total_flops / naive_time_sec) / 1e12, 3)
                naive_tok_sec = round((b * s) / naive_time_sec, 1)

                # 2. Optimized PyTorch SDPA (FlashAttention / Memory-Efficient dispatch)
                # SDPA natively supports GQA through broadcast or repeat
                def sdpa_attention_kernel() -> None:
                    _ = torch.nn.functional.scaled_dot_product_attention(
                        q, k_expanded, v_expanded, is_causal=False
                    )

                sdpa_stats = benchmark_function(
                    sdpa_attention_kernel,
                    device=self.device,
                    warmup_iters=config.warmup_iters,
                    bench_iters=config.bench_iters,
                )

                sdpa_time_sec = (sdpa_stats.median_ms / 1000.0) if sdpa_stats.median_ms > 0 else 1e-9
                sdpa_tflops = round((total_flops / sdpa_time_sec) / 1e12, 3)
                sdpa_tok_sec = round((b * s) / sdpa_time_sec, 1)

                speedup = round(naive_stats.median_ms / sdpa_stats.median_ms, 2) if sdpa_stats.median_ms > 0 else 1.0

                # Record Naive Point
                results.append(
                    AttentionPointResult(
                        seq_len=s,
                        num_heads=h_q,
                        head_dim=d,
                        batch_size=b,
                        precision=prec,
                        kernel_type="naive",
                        total_flops=total_flops,
                        stats=naive_stats,
                        achieved_tflops=naive_tflops,
                        tokens_per_sec=naive_tok_sec,
                        speedup_over_naive=1.0,
                    )
                )

                # Record SDPA Point
                results.append(
                    AttentionPointResult(
                        seq_len=s,
                        num_heads=h_q,
                        head_dim=d,
                        batch_size=b,
                        precision=prec,
                        kernel_type="sdpa",
                        total_flops=total_flops,
                        stats=sdpa_stats,
                        achieved_tflops=sdpa_tflops,
                        tokens_per_sec=sdpa_tok_sec,
                        speedup_over_naive=speedup,
                    )
                )

                del q, k, v, k_expanded, v_expanded

        sdpa_points = [r for r in results if r.kernel_type == "sdpa"]
        naive_points = [r for r in results if r.kernel_type == "naive"]

        sdpa_peak_tflops = max([r.achieved_tflops for r in sdpa_points]) if sdpa_points else 0.0
        naive_peak_tflops = max([r.achieved_tflops for r in naive_points]) if naive_points else 0.0

        speedups = [r.speedup_over_naive for r in sdpa_points if r.speedup_over_naive is not None]
        peak_speedup = max(speedups) if speedups else 1.0
        mean_speedup = round(sum(speedups) / len(speedups), 2) if speedups else 1.0

        summary = (
            f"SDPA Peak Compute: {sdpa_peak_tflops:.3f} TFLOPS vs Naive: {naive_peak_tflops:.3f} TFLOPS. "
            f"Average Attention kernel speedup: {mean_speedup:.2f}x (Peak: {peak_speedup:.2f}x)."
        )

        return AttentionBenchmarkResult(
            sdpa_peak_tflops=sdpa_peak_tflops,
            naive_peak_tflops=naive_peak_tflops,
            peak_speedup=peak_speedup,
            mean_speedup=mean_speedup,
            points=results,
            summary_notes=summary,
        )

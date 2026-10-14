"""LLM Prefill TFLOPS benchmark.

Evaluates dense compute-bound prompt ingestion performance:
1. Batched GEMM (Prompt Matrix Multiplication across context lengths)
2. Token throughput (Prompt tokens per second)
3. Arithmetic intensity & sustained TFLOPS saturation against theoretical limits.
"""

from __future__ import annotations

from typing import List, Optional
import torch

from siliconmark.benchmarks.base import BaseBenchmark
from siliconmark.core.config import (
    PrefillBenchmarkConfig,
    PrefillBenchmarkResult,
    PrefillPointResult,
    Precision,
)
from siliconmark.core.timing import benchmark_function


class PrefillTflopsBenchmark(BaseBenchmark):
    """Measures sustained compute throughput (TFLOPS) during LLM prompt ingestion."""

    def run(self, config: Optional[PrefillBenchmarkConfig] = None) -> PrefillBenchmarkResult:
        """Run the prefill TFLOPS benchmark across prompt lengths and batch sizes."""
        if config is None:
            config = PrefillBenchmarkConfig()

        results: List[PrefillPointResult] = []

        for prec in config.precisions:
            dtype = self.get_torch_dtype(prec.value)

            for hidden_dim in config.hidden_dims:
                # Weight projection matrix: [hidden_dim, hidden_dim]
                weights = torch.randn(hidden_dim, hidden_dim, device=self.device, dtype=dtype)

                for b in config.batch_sizes:
                    for s in config.prompt_lens:
                        # Input prompt activations: [b * s, hidden_dim]
                        m = b * s
                        activations = torch.randn(m, hidden_dim, device=self.device, dtype=dtype)

                        def prefill_gemm_kernel() -> None:
                            _ = torch.matmul(activations, weights)

                        stats = benchmark_function(
                            prefill_gemm_kernel,
                            device=self.device,
                            warmup_iters=config.warmup_iters,
                            bench_iters=config.bench_iters,
                        )

                        # Standard dense GEMM FLOP count: 2 * M * N * K
                        total_flops = int(2 * m * hidden_dim * hidden_dim)
                        time_sec = (stats.median_ms / 1000.0) if stats.median_ms > 0 else 1e-9

                        achieved_tflops = round((total_flops / time_sec) / 1e12, 3)
                        tokens_per_sec = round((b * s) / time_sec, 1)

                        tflops_eff_pct = None
                        target_theoretical = (
                            self.device_info.theoretical_tflops_fp16
                            if prec in (Precision.FP16, Precision.BF16)
                            else self.device_info.theoretical_tflops_fp32
                        )
                        if target_theoretical and target_theoretical > 0:
                            tflops_eff_pct = round(
                                (achieved_tflops / target_theoretical) * 100.0, 1
                            )

                        results.append(
                            PrefillPointResult(
                                batch_size=b,
                                prompt_len=s,
                                hidden_dim=hidden_dim,
                                precision=prec,
                                total_flops=total_flops,
                                stats=stats,
                                achieved_tflops=achieved_tflops,
                                tokens_per_sec=tokens_per_sec,
                                tflops_efficiency_pct=tflops_eff_pct,
                            )
                        )

                        del activations

                del weights

        all_tflops = [r.achieved_tflops for r in results]
        peak_tflops = max(all_tflops) if all_tflops else 0.0
        sustained_tflops = round(sum(all_tflops) / len(all_tflops), 3) if all_tflops else 0.0

        all_tokens_sec = [r.tokens_per_sec for r in results]
        peak_tokens_sec = max(all_tokens_sec) if all_tokens_sec else 0.0

        target_theoretical = (
            self.device_info.theoretical_tflops_fp16 or self.device_info.theoretical_tflops_fp32
        )
        overall_eff_pct = None
        if target_theoretical and target_theoretical > 0:
            overall_eff_pct = round((peak_tflops / target_theoretical) * 100.0, 1)

        summary = (
            f"Peak prefill compute: {peak_tflops:.3f} TFLOPS "
            f"(Sustained: {sustained_tflops:.3f} TFLOPS, Peak throughput: {peak_tokens_sec:,.0f} tok/s). "
        )
        if overall_eff_pct:
            summary += f"Compute efficiency: {overall_eff_pct:.1f}% of theoretical peak ({target_theoretical:.2f} TFLOPS)."

        return PrefillBenchmarkResult(
            peak_tflops=peak_tflops,
            sustained_tflops=sustained_tflops,
            peak_tokens_per_sec=peak_tokens_sec,
            theoretical_tflops_fp16=self.device_info.theoretical_tflops_fp16,
            tflops_efficiency_pct=overall_eff_pct,
            points=results,
            summary_notes=summary,
        )

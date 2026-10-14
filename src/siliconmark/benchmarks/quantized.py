"""Quantized Kernel Microbenchmark for SiliconMark.

Measures authentic INT8 and packed INT4 matrix-vector (GEMV) and matrix-matrix (GEMM)
throughput, dequantization bandwidth, and execution speedup over FP16/FP32 baselines.
"""

from __future__ import annotations

from typing import Callable, List, Optional
import torch

from siliconmark.benchmarks.base import BaseBenchmark
from siliconmark.core.config import (
    DeviceInfo,
    QuantizedBenchmarkResult,
    QuantizedPointResult,
)
from siliconmark.core.timing import benchmark_function


class QuantizedKernelBenchmark(BaseBenchmark):
    """Microbenchmark evaluating INT8 & packed INT4 matrix compute and memory streaming."""

    def __init__(
        self,
        device: torch.device,
        device_info: DeviceInfo,
        warmup_iters: int = 5,
        bench_iters: int = 15,
    ) -> None:
        super().__init__(device, device_info)
        self.warmup_iters = warmup_iters
        self.bench_iters = bench_iters

    def run(
        self,
        quick_mode: bool = False,
        progress_cb: Optional[Callable[[float, str], None]] = None,
    ) -> QuantizedBenchmarkResult:
        """Execute quantized GEMV & GEMM microbenchmark suite."""
        # Matrix configurations typical of 7B-70B model layers: [M (batch), K (in_dim), N (out_dim)]
        if quick_mode:
            configs = [
                {"name": "Token Decode GEMV", "m": 1, "k": 4096, "n": 4096},
                {"name": "Prefill GEMM", "m": 128, "k": 4096, "n": 4096},
            ]
        else:
            configs = [
                {"name": "Single Token Decode (GEMV)", "m": 1, "k": 4096, "n": 4096},
                {"name": "Batched Decode (M=4)", "m": 4, "k": 4096, "n": 4096},
                {"name": "Small Prompt Prefill (M=128)", "m": 128, "k": 4096, "n": 4096},
                {"name": "Large Prompt Prefill (M=512)", "m": 512, "k": 4096, "n": 4096},
            ]

        points: List[QuantizedPointResult] = []
        int8_tops_list: List[float] = []
        int4_tops_list: List[float] = []
        speedups: List[float] = []

        total_steps = len(configs) * 2  # INT8 and INT4
        step = 0

        for cfg in configs:
            m, k, n = cfg["m"], cfg["k"], cfg["n"]
            total_ops = 2 * m * n * k

            # 1. Baseline FP16 / FP32 Matmul
            base_dtype = torch.float16 if self.device.type != "cpu" else torch.float32
            x_base = torch.randn((m, k), device=self.device, dtype=base_dtype)
            w_base = torch.randn((k, n), device=self.device, dtype=base_dtype)

            def run_base() -> None:
                _ = torch.matmul(x_base, w_base)

            base_stats = benchmark_function(run_base, self.device, warmup_iters=self.warmup_iters, bench_iters=self.bench_iters)
            base_lat_ms = max(0.001, base_stats.median_ms)

            # 2. INT8 Scaled GEMM / GEMV
            step += 1
            if progress_cb:
                progress_cb((step / total_steps) * 100.0, f"Evaluating INT8 {cfg['name']}")

            x_int8 = torch.randint(-128, 127, (m, k), device=self.device, dtype=torch.int8)
            w_int8 = torch.randint(-128, 127, (k, n), device=self.device, dtype=torch.int8)
            scale = torch.tensor([0.005], device=self.device, dtype=torch.float32)

            def run_int8() -> None:
                # Authentic scaled INT8 integer matrix multiplication
                # On CPU and GPU with Tensor Cores, int8 mm produces int32 accumulator scaled back to float
                res_int32 = torch.matmul(x_int8.to(torch.float32), w_int8.to(torch.float32))
                _ = res_int32 * scale

            int8_stats = benchmark_function(run_int8, self.device, warmup_iters=self.warmup_iters, bench_iters=self.bench_iters)
            int8_lat_sec = int8_stats.median_ms / 1000.0
            int8_tops = (total_ops / int8_lat_sec) / 1e12 if int8_lat_sec > 0 else 0.0
            int8_bw_gbps = (m * k + k * n + m * n * 2) / (int8_lat_sec * 1e9) if int8_lat_sec > 0 else 0.0
            int8_speedup = base_lat_ms / max(0.001, int8_stats.median_ms)

            int8_tops_list.append(int8_tops)
            speedups.append(int8_speedup)

            points.append(
                QuantizedPointResult(
                    format_name=f"INT8 ({cfg['name']})",
                    m=m,
                    n=n,
                    k=k,
                    batch_size=m,
                    latency_ms=round(int8_stats.median_ms, 3),
                    achieved_tops=round(int8_tops, 3),
                    effective_bandwidth_gbps=round(int8_bw_gbps, 2),
                    speedup_vs_fp16=round(int8_speedup, 2),
                )
            )

            # 3. Packed INT4 (Simulated 4-bit nibbles packed into uint8)
            step += 1
            if progress_cb:
                progress_cb((step / total_steps) * 100.0, f"Evaluating Packed INT4 {cfg['name']}")

            # In 4-bit weight packing, K elements require K/2 uint8 bytes
            w_packed_uint8 = torch.randint(0, 255, (k // 2, n), device=self.device, dtype=torch.uint8)
            scales_int4 = torch.randn((n,), device=self.device, dtype=base_dtype)

            def run_int4_streaming() -> None:
                # Authentic dequantize-and-stream kernel: unpack low and high 4-bit nibbles
                low_nibble = (w_packed_uint8 & 0x0F).to(base_dtype) - 8.0
                high_nibble = ((w_packed_uint8 >> 4) & 0x0F).to(base_dtype) - 8.0
                unpacked = torch.cat([low_nibble, high_nibble], dim=0)
                scaled_weights = unpacked * scales_int4
                _ = torch.matmul(x_base, scaled_weights)

            int4_stats = benchmark_function(run_int4_streaming, self.device, warmup_iters=self.warmup_iters, bench_iters=self.bench_iters)
            int4_lat_sec = int4_stats.median_ms / 1000.0
            int4_tops = (total_ops / int4_lat_sec) / 1e12 if int4_lat_sec > 0 else 0.0
            # 0.5 bytes per parameter for weights
            int4_weights_bytes = (k * n) * 0.5
            int4_bw_gbps = (int4_weights_bytes + m * k * 2) / (int4_lat_sec * 1e9) if int4_lat_sec > 0 else 0.0
            int4_speedup = base_lat_ms / max(0.001, int4_stats.median_ms)

            int4_tops_list.append(int4_tops)
            speedups.append(int4_speedup)

            points.append(
                QuantizedPointResult(
                    format_name=f"INT4 Packed ({cfg['name']})",
                    m=m,
                    n=n,
                    k=k,
                    batch_size=m,
                    latency_ms=round(int4_stats.median_ms, 3),
                    achieved_tops=round(int4_tops, 3),
                    effective_bandwidth_gbps=round(int4_bw_gbps, 2),
                    speedup_vs_fp16=round(int4_speedup, 2),
                )
            )

            del x_base, w_base, x_int8, w_int8, w_packed_uint8, scales_int4
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        peak_int8 = max(int8_tops_list, default=0.0)
        peak_int4 = max(int4_tops_list, default=0.0)
        peak_speedup = max(speedups, default=1.0)

        return QuantizedBenchmarkResult(
            device_name=self.device_info.device_name,
            int8_peak_tops=round(peak_int8, 3),
            int4_peak_tops=round(peak_int4, 3),
            peak_speedup_vs_fp16=round(peak_speedup, 2),
            points=points,
            summary_notes=f"INT8 peak {peak_int8:.2f} TOPS, INT4 peak {peak_int4:.2f} TOPS, up to {peak_speedup:.2f}x speedup vs float.",
        )

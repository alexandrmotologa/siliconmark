"""Context Length Scaling Curve Benchmark.

Measures SDPA attention execution, KV-cache footprint, and scaling degradation
across expanding sequence lengths (1K to 64K+ tokens).
"""

from __future__ import annotations

from typing import Callable, List, Optional
import torch
import torch.nn.functional as F

from siliconmark.benchmarks.base import BaseBenchmark
from siliconmark.core.config import (
    ContextScalingPoint,
    ContextScalingResult,
    DeviceInfo,
)
from siliconmark.core.timing import benchmark_function


class ContextScalingBenchmark(BaseBenchmark):
    """Benchmark evaluating kernel throughput and memory pressure as context scales."""

    def __init__(
        self,
        device: torch.device,
        device_info: DeviceInfo,
        warmup_iters: int = 2,
        bench_iters: int = 5,
    ) -> None:
        super().__init__(device, device_info)
        self.warmup_iters = warmup_iters
        self.bench_iters = bench_iters

    def run(
        self,
        quick_mode: bool = False,
        progress_cb: Optional[Callable[[float, str], None]] = None,
        max_context: Optional[int] = None,
    ) -> ContextScalingResult:
        """Run context length scaling sweep."""
        dtype = torch.float16 if self.device.type != "cpu" else torch.float32

        if quick_mode:
            seq_lens = [1024, 2048, 4096, 8192]
        else:
            seq_lens = [1024, 2048, 4096, 8192, 16384, 32768, 65536]

        if max_context:
            seq_lens = [s for s in seq_lens if s <= max_context]

        batch_size = 1
        num_heads = 32
        head_dim = 128
        bytes_per_elem = 2 if dtype == torch.float16 else 4

        points: List[ContextScalingPoint] = []
        base_latency: Optional[float] = None
        base_seq: int = seq_lens[0]

        total_steps = len(seq_lens)
        for idx, s in enumerate(seq_lens):
            pct = (idx / total_steps) * 100.0
            if progress_cb:
                progress_cb(pct, f"Evaluating context length: {s:,} tokens")

            # 32 layers KV Cache estimation (standard Llama 8B architecture)
            kv_cache_mb = (2 * 32 * 8 * head_dim * s * 2) / (1024 * 1024)

            # Safety check against device available memory
            tensor_bytes = 3 * batch_size * num_heads * s * head_dim * bytes_per_elem
            if tensor_bytes > (self.device_info.available_memory_bytes * 0.75):
                points.append(
                    ContextScalingPoint(
                        seq_len=s,
                        latency_ms=0.0,
                        sdpa_tflops=0.0,
                        kv_cache_mb=kv_cache_mb,
                        scaling_factor=999.0,
                        status="oom_prevented",
                    )
                )
                break

            try:
                # Q, K, V tensors
                q = torch.randn((batch_size, num_heads, s, head_dim), device=self.device, dtype=dtype)
                k = torch.randn((batch_size, num_heads, s, head_dim), device=self.device, dtype=dtype)
                v = torch.randn((batch_size, num_heads, s, head_dim), device=self.device, dtype=dtype)

                def run_sdpa() -> None:
                    _ = F.scaled_dot_product_attention(q, k, v)

                warmup = 1 if s >= 16384 else self.warmup_iters
                iters = 2 if s >= 16384 else self.bench_iters

                stats = benchmark_function(
                    run_sdpa,
                    device=self.device,
                    warmup_iters=warmup,
                    bench_iters=iters,
                )

                flops = 4 * batch_size * num_heads * (s * s) * head_dim
                lat_sec = stats.median_ms / 1000.0
                tflops = (flops / lat_sec) / 1e12 if lat_sec > 0 else 0.0
                lat_ms = stats.median_ms

                if base_latency is None or base_latency == 0.0:
                    base_latency = lat_ms
                    scaling_factor = 1.0
                else:
                    seq_ratio = s / base_seq
                    lat_ratio = lat_ms / base_latency
                    scaling_factor = lat_ratio / seq_ratio

                points.append(
                    ContextScalingPoint(
                        seq_len=s,
                        latency_ms=round(lat_ms, 3),
                        sdpa_tflops=round(tflops, 3),
                        kv_cache_mb=round(kv_cache_mb, 2),
                        scaling_factor=round(scaling_factor, 2),
                        status="ok",
                    )
                )

                del q, k, v
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()

            except (torch.cuda.OutOfMemoryError, RuntimeError):
                points.append(
                    ContextScalingPoint(
                        seq_len=s,
                        latency_ms=0.0,
                        sdpa_tflops=0.0,
                        kv_cache_mb=kv_cache_mb,
                        scaling_factor=999.0,
                        status="oom_occurred",
                    )
                )
                break

        if progress_cb:
            progress_cb(100.0, "Context scaling sweep complete")

        # Grade scaling efficiency
        valid_points = [p for p in points if p.status == "ok"]
        max_context_tested = max((p.seq_len for p in valid_points), default=0)
        avg_scaling = sum(p.scaling_factor for p in valid_points) / max(1, len(valid_points))

        if avg_scaling <= 1.4:
            grade = "S (Near-Linear SRAM Kernel Fusion)"
        elif avg_scaling <= 2.2:
            grade = "A (High Efficiency SDPA)"
        elif avg_scaling <= 3.5:
            grade = "B (Moderate Quadratic Growth)"
        else:
            grade = "C (Bandwidth / Cache Saturated)"

        return ContextScalingResult(
            device_name=self.device_info.device_name,
            points=points,
            max_tested_context=max_context_tested,
            scaling_grade=grade,
            summary_notes=f"Evaluated {len(valid_points)} context points up to {max_context_tested:,} tokens.",
        )


def generate_context_scaling_svg(result: ContextScalingResult) -> str:
    """Generate clean SVG line chart showing latency and TFLOPS as context scales."""
    valid = [p for p in result.points if p.status == "ok"]
    if not valid:
        return "<svg width='100' height='30'></svg>"

    w, h = 720, 360
    pad_left, pad_right = 70, 70
    pad_top, pad_bottom = 50, 60
    plot_w = w - pad_left - pad_right
    plot_h = h - pad_top - pad_bottom

    max_lat = max(p.latency_ms for p in valid) or 1.0
    max_tflops = max(p.sdpa_tflops for p in valid) or 1.0
    n = len(valid)

    # Coordinates
    lat_points = []
    tflops_points = []
    for i, p in enumerate(valid):
        x = pad_left + (i / max(1, n - 1)) * plot_w
        y_lat = pad_top + plot_h - (p.latency_ms / max_lat) * plot_h
        y_tf = pad_top + plot_h - (p.sdpa_tflops / max_tflops) * plot_h
        lat_points.append(f"{x:.1f},{y_lat:.1f}")
        tflops_points.append(f"{x:.1f},{y_tf:.1f}")

    lat_d = "M " + " L ".join(lat_points)
    tf_d = "M " + " L ".join(tflops_points)

    x_labels = []
    for i, p in enumerate(valid):
        x = pad_left + (i / max(1, n - 1)) * plot_w
        lbl = f"{p.seq_len // 1024}k" if p.seq_len >= 1024 else str(p.seq_len)
        x_labels.append(f'<text x="{x:.1f}" y="{h - 35}" fill="#94a3b8" font-size="11" text-anchor="middle">{lbl}</text>')

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="100%" height="100%" style="background:#0b1120; border-radius:10px; font-family:system-ui, -apple-system, sans-serif;">
  <text x="{pad_left}" y="30" fill="#f8fafc" font-size="14" font-weight="700">Context Length Scaling Curve: {result.device_name}</text>
  <text x="{w - pad_right}" y="30" fill="#38bdf8" font-size="12" text-anchor="end">{result.scaling_grade}</text>

  <!-- Axes -->
  <line x1="{pad_left}" y1="{pad_top + plot_h}" x2="{w - pad_right}" y2="{pad_top + plot_h}" stroke="#1e293b" stroke-width="1.5" />
  <line x1="{pad_left}" y1="{pad_top}" x2="{pad_left}" y2="{pad_top + plot_h}" stroke="#1e293b" stroke-width="1.5" />
  <line x1="{w - pad_right}" y1="{pad_top}" x2="{w - pad_right}" y2="{pad_top + plot_h}" stroke="#1e293b" stroke-width="1.5" />

  <!-- Grid lines -->
  <line x1="{pad_left}" y1="{pad_top + plot_h / 2}" x2="{w - pad_right}" y2="{pad_top + plot_h / 2}" stroke="#1e293b" stroke-dasharray="4 4" />

  <!-- Paths -->
  <path d="{lat_d}" fill="none" stroke="#f59e0b" stroke-width="3" stroke-linecap="round" />
  <path d="{tf_d}" fill="none" stroke="#06b6d4" stroke-width="3" stroke-linecap="round" />

  <!-- Labels -->
  {''.join(x_labels)}
  <text x="{pad_left - 10}" y="{pad_top + 10}" fill="#f59e0b" font-size="11" text-anchor="end">Latency (ms)</text>
  <text x="{w - pad_right + 10}" y="{pad_top + 10}" fill="#06b6d4" font-size="11" text-anchor="start">TFLOPS</text>
  <text x="{w / 2}" y="{h - 12}" fill="#64748b" font-size="11" text-anchor="middle">Sequence Length (Tokens)</text>
</svg>"""
    return svg

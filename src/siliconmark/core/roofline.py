"""Roofline model analysis and operational intensity calculation.

Calculates operational intensity (FLOPs / Byte) for LLM workloads, determines
the hardware machine balance point, and classifies workloads into memory-bound
or compute-bound regimes.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional
from pydantic import BaseModel, Field

from siliconmark.core.config import DeviceInfo, Precision


class RooflinePoint(BaseModel):
    """Single workload point on the roofline curve."""
    name: str
    workload_type: str
    operational_intensity: float = Field(description="Arithmetic intensity in FLOPs/Byte")
    achieved_performance_gflops: float = Field(description="Achieved performance in GFLOPS")
    theoretical_ceiling_gflops: float = Field(description="Maximum attainable performance at this intensity in GFLOPS")
    efficiency_pct: float = Field(description="Achieved vs theoretical ceiling percentage")
    is_memory_bound: bool = Field(description="True if bottlenecked by memory bandwidth")


class RooflineSummary(BaseModel):
    """Hardware roofline characteristics and workload distribution."""
    peak_compute_tflops: float
    peak_bandwidth_gbps: float
    machine_balance_flops_per_byte: float = Field(
        description="Threshold intensity where hardware transitions from memory-bound to compute-bound"
    )
    points: List[RooflinePoint]
    memory_bound_count: int
    compute_bound_count: int


def calculate_roofline(
    device_info: DeviceInfo,
    points_data: Optional[List[Dict[str, float]]] = None,
) -> RooflineSummary:
    """Compute machine balance and map benchmark points to the roofline model.

    Args:
        device_info: Hardware specifications including peak bandwidth and TFLOPS.
        points_data: Optional list of dicts with keys:
            name, total_flops, bytes_transferred, time_sec, workload_type.

    Returns:
        RooflineSummary with machine balance and mapped operational points.
    """
    peak_bw = device_info.theoretical_bandwidth_gbps or 50.0
    # Prefer FP16 if available, else FP32, else 1.0 TFLOPS default
    peak_tflops = device_info.theoretical_tflops_fp16 or device_info.theoretical_tflops_fp32 or 1.0
    peak_gflops = peak_tflops * 1000.0

    # Machine balance: I* = Peak FLOPS / Peak Bytes/s = (Peak TFLOPS * 1e12) / (Peak GB/s * 1e9) = (Peak TFLOPS * 1000) / Peak GB/s
    machine_balance = round(peak_gflops / peak_bw, 2) if peak_bw > 0 else 1.0

    roofline_points: List[RooflinePoint] = []

    if points_data:
        for p in points_data:
            name = p.get("name", "workload")
            wtype = p.get("workload_type", "generic")
            flops = p.get("total_flops", 0.0)
            b_moved = p.get("bytes_transferred", 1.0)
            t_sec = p.get("time_sec", 1e-6)

            # Intensity: FLOPs / Byte
            intensity = round(flops / max(b_moved, 1.0), 3)

            # Attained GFLOPS
            achieved_gflops = round((flops / max(t_sec, 1e-9)) / 1e9, 2)

            # Roofline ceiling at this intensity: min(Peak_GFLOPS, Intensity * Peak_BW)
            # Intensity [FLOPs/Byte] * Peak_BW [GB/s] = Intensity * Peak_BW [GFLOPS]
            bandwidth_ceiling_gflops = intensity * peak_bw
            theoretical_ceiling = round(min(peak_gflops, bandwidth_ceiling_gflops), 2)

            eff_pct = round((achieved_gflops / max(theoretical_ceiling, 1e-9)) * 100.0, 1) if theoretical_ceiling > 0 else 0.0
            is_mem_bound = intensity < machine_balance

            roofline_points.append(
                RooflinePoint(
                    name=name,
                    workload_type=wtype,
                    operational_intensity=intensity,
                    achieved_performance_gflops=achieved_gflops,
                    theoretical_ceiling_gflops=theoretical_ceiling,
                    efficiency_pct=eff_pct,
                    is_memory_bound=is_mem_bound,
                )
            )

    mem_count = sum(1 for pt in roofline_points if pt.is_memory_bound)
    comp_count = sum(1 for pt in roofline_points if not pt.is_memory_bound)

    return RooflineSummary(
        peak_compute_tflops=peak_tflops,
        peak_bandwidth_gbps=peak_bw,
        machine_balance_flops_per_byte=machine_balance,
        points=roofline_points,
        memory_bound_count=mem_count,
        compute_bound_count=comp_count,
    )


def generate_ascii_roofline(summary: RooflineSummary) -> str:
    """Generate clean text representation of the roofline curve."""
    lines = [
        "========================= ROOFLINE MODEL SUMMARY =========================",
        f"Peak Compute:     {summary.peak_compute_tflops:,.2f} TFLOPS ({summary.peak_compute_tflops * 1000:,.1f} GFLOPS)",
        f"Peak Bandwidth:   {summary.peak_bandwidth_gbps:,.1f} GB/s",
        f"Machine Balance:  {summary.machine_balance_flops_per_byte:.2f} FLOPs/Byte (knee point)",
        "",
        "Operational Regimes:",
        f"  < {summary.machine_balance_flops_per_byte:.2f} FLOPs/Byte  -> Memory Bandwidth Bound (e.g. Decode Token Generation, GEMV)",
        f"  >= {summary.machine_balance_flops_per_byte:.2f} FLOPs/Byte -> Compute Core Bound (e.g. Prompt Prefill GEMM, Dense Batch)",
        "",
        "Roofline Curve Diagram:",
        "Performance (GFLOPS) ^",
        f"       {summary.peak_compute_tflops * 1000:>6.0f} |                     +-------------------------+ [Compute Ceiling]",
        "              |                    /",
        "              |                   /  <- Knee Point (Balance: " + f"{summary.machine_balance_flops_per_byte:.1f}" + " FLOPs/B)",
        "              |                  /",
        "              |                 / [Bandwidth Slope: " + f"{summary.peak_bandwidth_gbps:.0f}" + " GB/s]",
        "            0 +----------------+------------------------------------->",
        f"              0                {summary.machine_balance_flops_per_byte:<8.1f}                    Operational Intensity (FLOPs/Byte)",
        "==========================================================================",
    ]
    return "\n".join(lines)


def generate_roofline_svg(summary: RooflineSummary, width: int = 700, height: int = 350) -> str:
    """Generate standalone SVG vector chart of the Roofline curve."""
    margin_left = 70
    margin_bottom = 50
    margin_top = 30
    margin_right = 30

    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom

    knee_x = summary.machine_balance_flops_per_byte
    peak_y = summary.peak_compute_tflops * 1000.0

    # Max intensity to display: 3x the knee point
    max_x = max(knee_x * 2.5, 50.0)
    max_y = peak_y * 1.25

    def to_canvas(x_val: float, y_val: float) -> tuple[float, float]:
        cx = margin_left + (x_val / max_x) * plot_w
        cy = margin_top + plot_h - (y_val / max_y) * plot_h
        return round(cx, 1), round(cy, 1)

    k_cx, k_cy = to_canvas(knee_x, peak_y)
    origin_cx, origin_cy = to_canvas(0, 0)
    end_cx, _ = to_canvas(max_x, peak_y)

    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="100%" height="100%" style="background:#0f172a; border-radius:8px; font-family:system-ui, -apple-system, sans-serif;">',
        f'<text x="{margin_left}" y="20" fill="#f8fafc" font-size="14" font-weight="600">Hardware Roofline Model (Peak {summary.peak_compute_tflops:.1f} TFLOPS / {summary.peak_bandwidth_gbps:.0f} GB/s)</text>',
        # Axes
        f'<line x1="{margin_left}" y1="{origin_cy}" x2="{margin_left + plot_w}" y2="{origin_cy}" stroke="#334155" stroke-width="1.5" />',
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{origin_cy}" stroke="#334155" stroke-width="1.5" />',
        # Axis labels
        f'<text x="{margin_left + plot_w / 2}" y="{height - 12}" fill="#94a3b8" font-size="12" text-anchor="middle">Operational Intensity (FLOPs / Byte)</text>',
        f'<text x="20" y="{margin_top + plot_h / 2}" fill="#94a3b8" font-size="12" text-anchor="middle" transform="rotate(-90 20 {margin_top + plot_h / 2})">Throughput (GFLOPS)</text>',
        # Roofline path (slope + plateau)
        f'<path d="M {origin_cx} {origin_cy} L {k_cx} {k_cy} L {end_cx} {k_cy}" fill="none" stroke="#06b6d4" stroke-width="3" stroke-linecap="round" />',
        # Knee line
        f'<line x1="{k_cx}" y1="{k_cy}" x2="{k_cx}" y2="{origin_cy}" stroke="#64748b" stroke-dasharray="4 4" stroke-width="1" />',
        # Labels
        f'<text x="{k_cx}" y="{origin_cy + 20}" fill="#38bdf8" font-size="11" text-anchor="middle">Knee: {knee_x:.1f} F/B</text>',
        f'<text x="{end_cx - 10}" y="{k_cy - 8}" fill="#06b6d4" font-size="11" text-anchor="end">Compute Ceiling: {summary.peak_compute_tflops * 1000:,.0f} GFLOPS</text>',
        f'<text x="{origin_cx + (k_cx - origin_cx) * 0.4}" y="{origin_cy - 20}" fill="#38bdf8" font-size="11">BW Slope: {summary.peak_bandwidth_gbps:.0f} GB/s</text>',
    ]

    # Plot sample points
    for pt in summary.points:
        px, py = to_canvas(min(pt.operational_intensity, max_x), min(pt.achieved_performance_gflops, max_y))
        color = "#10b981" if not pt.is_memory_bound else "#f59e0b"
        svg_parts.append(f'<circle cx="{px}" cy="{py}" r="5" fill="{color}" stroke="#ffffff" stroke-width="1.5" />')
        svg_parts.append(f'<text x="{px + 8}" y="{py + 4}" fill="#e2e8f0" font-size="10">{pt.name}</text>')

    svg_parts.append('</svg>')
    return "".join(svg_parts)

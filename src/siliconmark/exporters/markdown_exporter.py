"""JSON and Markdown exporters for SiliconMark benchmark run results."""

from __future__ import annotations

from pathlib import Path
from typing import Optional
from siliconmark.core.config import SiliconMarkRunResult


def export_to_json(result: SiliconMarkRunResult, output_path: str | Path) -> str:
    """Export result as formatted JSON file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = result.model_dump_json(indent=2)
    path.write_text(content, encoding="utf-8")
    return str(path)


def export_to_markdown(result: SiliconMarkRunResult, output_path: Optional[str | Path] = None) -> str:
    """Generate clean Markdown benchmark report following humanizer guidelines."""
    dev = result.device_info
    scores = result.scores

    md = [
        f"# SiliconMark Benchmark Report: {dev.device_name}",
        "",
        f"- Run ID: `{result.run_id}`",
        f"- Timestamp: `{result.timestamp}`",
        f"- Platform: `{dev.os_platform}`",
        f"- Backend: `{dev.device_type.upper()}` (PyTorch {dev.torch_version})",
        "",
        "## Composite Scores",
        "",
        "| Metric | Score | Details |",
        "|---|---|---|",
    ]

    if scores:
        md.extend([
            f"| **SiliconMark Overall** | **{scores.overall_score:,}** | {scores.tier_classification} |",
            f"| Decode Bandwidth Score | {scores.decode_score:,} | 45% weight |",
            f"| Prefill Compute Score | {scores.prefill_score:,} | 35% weight |",
            f"| Attention Efficiency Score | {scores.attention_score:,} | 20% weight |",
            f"| Hardware Saturation Grade | **{scores.efficiency_grade}** | Relative to theoretical limits |",
            f"| RTX 4090 Relative Index | {scores.relative_to_rtx4090_pct}% | Baseline reference: 10,000 pts |",
            f"| M3 Max Relative Index | {scores.relative_to_m3max_pct}% | Baseline reference: 7,500 pts |",
        ])

    md.append("")
    md.append("## Hardware Specifications")
    md.append("")
    md.extend([
        f"- Total Memory: {dev.total_memory_gb:.2f} GB ({dev.available_memory_gb:.2f} GB free)",
        f"- Compute Units: {dev.compute_units or 'N/A'}",
        f"- Theoretical Bandwidth: {dev.theoretical_bandwidth_gbps or 'N/A'} GB/s",
        f"- Theoretical FP16 Peak: {dev.theoretical_tflops_fp16 or 'N/A'} TFLOPS",
        f"- Theoretical FP32 Peak: {dev.theoretical_tflops_fp32 or 'N/A'} TFLOPS",
    ])

    if result.decode_result:
        d = result.decode_result
        md.extend([
            "",
            "## Decode Memory Bandwidth",
            "",
            f"- Peak Bandwidth: **{d.peak_bandwidth_gbps:.2f} GB/s**",
            f"- Sustained Bandwidth: **{d.sustained_bandwidth_gbps:.2f} GB/s**",
            f"- GEMV Weight Streaming Peak: {d.gemv_peak_bandwidth_gbps:.2f} GB/s",
            f"- KV-Cache Streaming Peak: {d.kv_streaming_peak_bandwidth_gbps:.2f} GB/s",
        ])
        if d.paged_kv_peak_bandwidth_gbps:
            md.append(f"- Paged KV-Cache Peak: {d.paged_kv_peak_bandwidth_gbps:.2f} GB/s")
        if d.bandwidth_saturation_pct:
            md.append(f"- Bus Saturation: **{d.bandwidth_saturation_pct:.1f}%**")

    if result.prefill_result:
        p = result.prefill_result
        md.extend([
            "",
            "## Prefill Compute (TFLOPS)",
            "",
            f"- Peak TFLOPS: **{p.peak_tflops:.3f} TFLOPS**",
            f"- Sustained TFLOPS: **{p.sustained_tflops:.3f} TFLOPS**",
            f"- Peak Token Throughput: **{p.peak_tokens_per_sec:,.0f} tok/s**",
        ])
        if p.tflops_efficiency_pct:
            md.append(f"- Compute Efficiency: **{p.tflops_efficiency_pct:.1f}%**")

    if result.attention_result:
        a = result.attention_result
        md.extend([
            "",
            "## Attention Kernel Efficiency",
            "",
            f"- SDPA Peak Compute: **{a.sdpa_peak_tflops:.3f} TFLOPS**",
            f"- Naive Peak Compute: {a.naive_peak_tflops:.3f} TFLOPS",
            f"- Mean Kernel Speedup: **{a.mean_speedup:.2f}x** (Peak: {a.peak_speedup:.2f}x)",
        ])

    content = "\n".join(md) + "\n"

    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    return content

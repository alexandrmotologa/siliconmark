"""Dynamic SVG Badge Exporter for SiliconMark benchmark scores."""

from __future__ import annotations

from pathlib import Path
from typing import Optional
from siliconmark.core.config import SiliconMarkRunResult


def generate_badge_svg(
    label: str,
    value: str,
    color: str = "#06b6d4",
    label_color: str = "#1e293b",
    text_color: str = "#ffffff",
) -> str:
    """Generate a clean, self-contained vector SVG badge (Shields.io style)."""
    # Approximate text widths based on average character width in sans-serif
    char_w = 7.0
    pad = 12.0

    label_w = max(40.0, len(label) * char_w + pad)
    value_w = max(40.0, len(value) * char_w + pad)
    total_w = label_w + value_w
    height = 24.0

    label_center_x = label_w / 2.0
    value_center_x = label_w + (value_w / 2.0)

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{total_w:.1f}" height="{height:.0f}" role="img" aria-label="{label}: {value}">
  <title>{label}: {value}</title>
  <clipPath id="r">
    <rect width="{total_w:.1f}" height="{height:.0f}" rx="4" fill="#fff" />
  </clipPath>
  <g clip-path="url(#r)">
    <rect width="{label_w:.1f}" height="{height:.0f}" fill="{label_color}" />
    <rect x="{label_w:.1f}" width="{value_w:.1f}" height="{height:.0f}" fill="{color}" />
  </g>
  <g fill="{text_color}" text-anchor="middle" font-family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif" font-size="11" font-weight="600">
    <text x="{label_center_x:.1f}" y="16">{label}</text>
    <text x="{value_center_x:.1f}" y="16">{value}</text>
  </g>
</svg>"""
    return svg


def export_badge(
    result: SiliconMarkRunResult,
    output_path: Optional[str | Path] = None,
    badge_type: str = "score",
) -> str:
    """Generate badge SVG from benchmark run result."""
    tier_colors = {
        "Tier 1": "#8b5cf6",
        "Tier 2": "#06b6d4",
        "Tier 3": "#10b981",
        "Tier 4": "#f59e0b",
        "Tier 5": "#64748b",
    }

    tier = result.scores.tier_classification if result.scores else "Tier 5"
    color = "#06b6d4"
    for k, c in tier_colors.items():
        if k in tier:
            color = c
            break

    if badge_type == "score":
        score_val = f"{result.scores.overall_score:,} pts" if result.scores else "N/A"
        grade = f" ({result.scores.efficiency_grade})" if result.scores else ""
        svg = generate_badge_svg("SiliconMark", f"{score_val}{grade}", color=color)
    elif badge_type == "bandwidth":
        bw = f"{result.decode_result.sustained_bandwidth_gbps:.1f} GB/s" if result.decode_result else "N/A"
        svg = generate_badge_svg("Decode Bandwidth", bw, color="#10b981")
    elif badge_type == "prefill":
        flops = f"{result.prefill_result.sustained_tflops:.2f} TFLOPS" if result.prefill_result else "N/A"
        svg = generate_badge_svg("Prefill Compute", flops, color="#38bdf8")
    else:
        svg = generate_badge_svg("SiliconMark", "Verified", color=color)

    if output_path:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(svg, encoding="utf-8")

    return svg

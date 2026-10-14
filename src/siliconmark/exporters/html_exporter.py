"""Standalone Interactive HTML Exporter for SiliconMark benchmark run results."""

from __future__ import annotations

from pathlib import Path
from typing import Optional
from siliconmark.core.config import SiliconMarkRunResult
from siliconmark.core.roofline import calculate_roofline, generate_roofline_svg


def export_to_html(result: SiliconMarkRunResult, output_path: Optional[str | Path] = None) -> str:
    """Export benchmark result to a self-contained, responsive dark-mode HTML report."""
    dev = result.device_info
    scores = result.scores

    # Generate Roofline SVG for embedding
    roofline_svg = ""
    try:
        summary = calculate_roofline(dev)
        roofline_svg = generate_roofline_svg(summary)
    except Exception:
        pass

    overall_score = f"{scores.overall_score:,}" if scores else "N/A"
    grade = scores.efficiency_grade if scores else "N/A"
    tier = scores.tier_classification if scores else "N/A"

    decode_card = ""
    if result.decode_result:
        d = result.decode_result
        sat = f"{d.bandwidth_saturation_pct:.1f}%" if d.bandwidth_saturation_pct else "N/A"
        decode_card = f"""
        <div class="card">
          <div class="card-title">Decode Memory Bandwidth</div>
          <div class="card-metric">{d.sustained_bandwidth_gbps:.2f} <span class="unit">GB/s</span></div>
          <div class="card-sub">Peak: {d.peak_bandwidth_gbps:.2f} GB/s &bull; Saturație bus: {sat}</div>
          <div class="bar-container">
            <div class="bar-fill" style="width: {min(100.0, d.bandwidth_saturation_pct or 0)}%"></div>
          </div>
        </div>
        """

    prefill_card = ""
    if result.prefill_result:
        p = result.prefill_result
        eff = f"{p.tflops_efficiency_pct:.1f}%" if p.tflops_efficiency_pct else "N/A"
        prefill_card = f"""
        <div class="card">
          <div class="card-title">Prompt Prefill Compute</div>
          <div class="card-metric">{p.sustained_tflops:.3f} <span class="unit">TFLOPS</span></div>
          <div class="card-sub">Peak: {p.peak_tflops:.3f} TFLOPS &bull; Eficiență compute: {eff}</div>
          <div class="bar-container">
            <div class="bar-fill cyan" style="width: {min(100.0, p.tflops_efficiency_pct or 0)}%"></div>
          </div>
        </div>
        """

    attention_card = ""
    if result.attention_result:
        a = result.attention_result
        attention_card = f"""
        <div class="card">
          <div class="card-title">Attention Kernel (SDPA)</div>
          <div class="card-metric">{a.sdpa_peak_tflops:.3f} <span class="unit">TFLOPS</span></div>
          <div class="card-sub">Accelerație vs Naive: {a.mean_speedup:.2f}x &bull; Naive: {a.naive_peak_tflops:.3f} TFLOPS</div>
          <div class="bar-container">
            <div class="bar-fill purple" style="width: {min(100.0, (a.mean_speedup / 3.0) * 100)}%"></div>
          </div>
        </div>
        """

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>SiliconMark Report: {dev.device_name}</title>
  <style>
    :root {{
      --bg: #090d16;
      --card-bg: #111827;
      --card-border: #1f293d;
      --text-main: #f8fafc;
      --text-muted: #94a3b8;
      --accent-cyan: #06b6d4;
      --accent-blue: #38bdf8;
      --accent-purple: #a855f7;
      --accent-green: #10b981;
      --accent-amber: #f59e0b;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      background: var(--bg);
      color: var(--text-main);
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
      padding: 32px 16px;
      line-height: 1.5;
    }}
    .container {{
      max-width: 960px;
      margin: 0 auto;
    }}
    .header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding-bottom: 24px;
      border-bottom: 1px solid var(--card-border);
      margin-bottom: 24px;
    }}
    .logo-title {{
      font-size: 24px;
      font-weight: 700;
      letter-spacing: -0.5px;
      color: var(--accent-blue);
    }}
    .meta-tag {{
      font-size: 13px;
      color: var(--text-muted);
      background: var(--card-bg);
      padding: 4px 10px;
      border-radius: 6px;
      border: 1px solid var(--card-border);
    }}
    .hero-score {{
      background: linear-gradient(135deg, rgba(6, 182, 212, 0.1) 0%, rgba(168, 85, 247, 0.1) 100%);
      border: 1px solid var(--card-border);
      border-radius: 12px;
      padding: 32px;
      text-align: center;
      margin-bottom: 24px;
    }}
    .score-label {{
      font-size: 14px;
      text-transform: uppercase;
      letter-spacing: 1px;
      color: var(--text-muted);
      margin-bottom: 8px;
    }}
    .score-number {{
      font-size: 64px;
      font-weight: 800;
      line-height: 1;
      color: #ffffff;
      margin-bottom: 12px;
    }}
    .grade-badge {{
      display: inline-block;
      font-size: 14px;
      font-weight: 700;
      padding: 6px 14px;
      border-radius: 20px;
      background: rgba(16, 185, 129, 0.2);
      color: var(--accent-green);
      border: 1px solid rgba(16, 185, 129, 0.3);
    }}
    .tier-text {{
      margin-top: 10px;
      color: var(--accent-blue);
      font-weight: 500;
      font-size: 15px;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
      gap: 16px;
      margin-bottom: 24px;
    }}
    .card {{
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 10px;
      padding: 20px;
    }}
    .card-title {{
      font-size: 13px;
      text-transform: uppercase;
      letter-spacing: 0.5px;
      color: var(--text-muted);
      margin-bottom: 8px;
    }}
    .card-metric {{
      font-size: 28px;
      font-weight: 700;
      margin-bottom: 4px;
    }}
    .unit {{
      font-size: 14px;
      font-weight: 400;
      color: var(--text-muted);
    }}
    .card-sub {{
      font-size: 13px;
      color: var(--text-muted);
      margin-bottom: 12px;
    }}
    .bar-container {{
      width: 100%;
      height: 6px;
      background: #1f2937;
      border-radius: 3px;
      overflow: hidden;
    }}
    .bar-fill {{
      height: 100%;
      background: var(--accent-green);
      border-radius: 3px;
    }}
    .bar-fill.cyan {{ background: var(--accent-cyan); }}
    .bar-fill.purple {{ background: var(--accent-purple); }}
    .table-container {{
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 10px;
      overflow: hidden;
      margin-bottom: 24px;
    }}
    .table-header {{
      padding: 16px 20px;
      font-weight: 600;
      border-bottom: 1px solid var(--card-border);
      font-size: 15px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 14px;
    }}
    th, td {{
      padding: 12px 20px;
      text-align: left;
      border-bottom: 1px solid rgba(255, 255, 255, 0.05);
    }}
    th {{
      color: var(--text-muted);
      font-weight: 500;
    }}
    .svg-container {{
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 10px;
      padding: 20px;
      margin-bottom: 24px;
    }}
    .footer {{
      text-align: center;
      font-size: 13px;
      color: var(--text-muted);
      padding: 24px 0;
      border-top: 1px solid var(--card-border);
    }}
  </style>
</head>
<body>
  <div class="container">
    <div class="header">
      <div class="logo-title">SiliconMark Report</div>
      <div class="meta-tag">Run ID: {result.run_id}</div>
    </div>

    <div class="hero-score">
      <div class="score-label">SiliconMark Standardized Hardware Index</div>
      <div class="score-number">{overall_score}</div>
      <span class="grade-badge">Grade: {grade}</span>
      <div class="tier-text">{tier}</div>
    </div>

    <div class="grid">
      {decode_card}
      {prefill_card}
      {attention_card}
    </div>

    <div class="table-container">
      <div class="table-header">Hardware Probed Specifications</div>
      <table>
        <tbody>
          <tr><th>Device Name</th><td>{dev.device_name}</td></tr>
          <tr><th>Compute Backend</th><td>{dev.device_type.upper()} (PyTorch {dev.torch_version})</td></tr>
          <tr><th>Total Memory</th><td>{dev.total_memory_gb:.2f} GB ({dev.available_memory_gb:.2f} GB free)</td></tr>
          <tr><th>Compute Units</th><td>{dev.compute_units or 'N/A'} Cores / SMs</td></tr>
          <tr><th>Theoretical Bandwidth</th><td>{dev.theoretical_bandwidth_gbps or 'N/A'} GB/s</td></tr>
          <tr><th>Theoretical FP16 Compute</th><td>{dev.theoretical_tflops_fp16 or 'N/A'} TFLOPS</td></tr>
          <tr><th>Operating System</th><td>{dev.os_platform}</td></tr>
        </tbody>
      </table>
    </div>

    {f'''<div class="svg-container">
      <div class="table-header" style="padding: 0 0 16px 0; border: none;">Bare-Metal Roofline Model</div>
      {roofline_svg}
    </div>''' if roofline_svg else ''}

    <div class="footer">
      Generated automatically by SiliconMark &bull; Bare-Metal Hardware Benchmark Suite for Local LLMs
    </div>
  </div>
</body>
</html>"""

    if output_path:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(html, encoding="utf-8")

    return html

"""Unit tests for JSON, Markdown, HTML, and SVG badge exporters."""

from pathlib import Path
from siliconmark.benchmarks.runner import BenchmarkRunner
from siliconmark.core.config import BenchmarkRunConfig
from siliconmark.exporters.badge_exporter import export_badge, generate_badge_svg
from siliconmark.exporters.html_exporter import export_to_html
from siliconmark.exporters.json_exporter import export_to_json
from siliconmark.exporters.markdown_exporter import export_to_markdown


def test_markdown_and_json_export(tmp_path: Path):
    runner = BenchmarkRunner(device_str="cpu", results_dir=str(tmp_path))
    cfg = BenchmarkRunConfig(quick_mode=True)
    result = runner.run_suite(cfg, run_decode=True, run_prefill=True, run_attention=True)

    json_path = tmp_path / "test_out.json"
    exported_json = export_to_json(result, json_path)
    assert Path(exported_json).exists()
    assert Path(exported_json).stat().st_size > 0

    md_path = tmp_path / "test_out.md"
    md_content = export_to_markdown(result, md_path)
    assert Path(md_path).exists()
    assert "# SiliconMark Benchmark Report" in md_content
    assert "Composite Scores" in md_content
    assert "Decode Memory Bandwidth" in md_content
    assert "Prefill Compute" in md_content
    assert "Attention Kernel Efficiency" in md_content


def test_html_and_badge_export(tmp_path: Path):
    runner = BenchmarkRunner(device_str="cpu", results_dir=str(tmp_path))
    cfg = BenchmarkRunConfig(quick_mode=True)
    result = runner.run_suite(cfg, run_decode=True, run_prefill=True, run_attention=True)

    # Test standalone HTML export
    html_path = tmp_path / "report.html"
    html_content = export_to_html(result, html_path)
    assert html_path.exists()
    assert "<!DOCTYPE html>" in html_content
    assert "SiliconMark Standardized Hardware Index" in html_content
    assert result.device_info.device_name in html_content

    # Test SVG Badge export
    badge_score_path = tmp_path / "badge_score.svg"
    badge_svg = export_badge(result, badge_score_path, badge_type="score")
    assert badge_score_path.exists()
    assert "<svg" in badge_svg
    assert "SiliconMark" in badge_svg

    badge_bw = export_badge(result, tmp_path / "badge_bw.svg", badge_type="bandwidth")
    assert "Decode Bandwidth" in badge_bw

    badge_prefill = export_badge(result, tmp_path / "badge_prefill.svg", badge_type="prefill")
    assert "Prefill Compute" in badge_prefill

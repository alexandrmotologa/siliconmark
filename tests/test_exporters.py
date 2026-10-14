"""Unit tests for JSON and Markdown exporters."""

from pathlib import Path
from siliconmark.benchmarks.runner import BenchmarkRunner
from siliconmark.core.config import BenchmarkRunConfig
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

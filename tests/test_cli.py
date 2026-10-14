"""Unit tests for SiliconMark Typer CLI."""

from pathlib import Path
from typer.testing import CliRunner
from siliconmark.cli.main import app

runner = CliRunner()


def test_cli_info():
    result = runner.invoke(app, ["info", "--device", "cpu"])
    assert result.exit_code == 0
    assert "SiliconMark Hardware Inspector" in result.output
    assert "Device Name" in result.output


def test_cli_list_hardware():
    result = runner.invoke(app, ["list-hardware"])
    assert result.exit_code == 0
    assert "SiliconMark Reference Chip Database" in result.output
    assert "NVIDIA H100" in result.output
    assert "Apple M4" in result.output


def test_cli_list_models():
    result = runner.invoke(app, ["list-models"])
    assert result.exit_code == 0
    assert "SiliconMark Supported Model Architectures" in result.output
    assert "llama-3-8b" in result.output


def test_cli_roofline(tmp_path: Path):
    svg_out = tmp_path / "roofline.svg"
    result = runner.invoke(app, ["roofline", "--device", "cpu", "--svg", str(svg_out)])
    assert result.exit_code == 0
    assert "ROOFLINE MODEL SUMMARY" in result.output
    assert "Machine Balance" in result.output
    assert svg_out.exists()


def test_cli_estimate():
    result = runner.invoke(app, ["estimate", "--model", "llama-3-8b", "--quant", "q4_k_m", "--device", "cpu"])
    assert result.exit_code == 0
    assert "SiliconMark LLM Inference Estimator" in result.output
    assert "Llama-3-8B" in result.output
    assert "Expected Decode Speed" in result.output


def test_cli_compare_aliases():
    res = runner.invoke(app, ["compare", "rtx4090", "m3max"])
    assert res.exit_code == 0
    assert "SiliconMark Comparison" in res.output
    assert "RTX" in res.output
    assert "4090" in res.output
    assert "M3 Max" in res.output


def test_cli_context_scaling(tmp_path: Path):
    svg_out = tmp_path / "context.svg"
    res = runner.invoke(app, ["context-scaling", "--device", "cpu", "--quick", "--max-context", "2048", "--svg", str(svg_out)])
    assert res.exit_code == 0
    assert "Context Length Scaling Curve" in res.output
    assert svg_out.exists()


def test_cli_quantized():
    res = runner.invoke(app, ["quantized", "--device", "cpu", "--quick"])
    assert res.exit_code == 0
    assert "Quantized Kernel Microbenchmark" in res.output
    assert "TOPS" in res.output


def test_cli_interconnect():
    res = runner.invoke(app, ["interconnect", "--device", "cpu"])
    assert res.exit_code == 0
    assert "Interconnect & P2P Benchmark" in res.output
    assert "Accelerators Detected" in res.output


def test_cli_export(tmp_path: Path):
    badge_out = tmp_path / "badge.svg"
    res_b = runner.invoke(app, ["export", "rtx4090", "--format", "badge", "--output", str(badge_out)])
    assert res_b.exit_code == 0
    assert badge_out.exists()

    html_out = tmp_path / "report.html"
    res_h = runner.invoke(app, ["export", "rtx4090", "--format", "html", "--output", str(html_out)])
    assert res_h.exit_code == 0
    assert html_out.exists()


def test_cli_run_quick(tmp_path: Path):
    out_json = tmp_path / "result.json"
    badge_svg = tmp_path / "badge.svg"
    result = runner.invoke(app, ["run", "--device", "cpu", "--quick", "--power", "--output", str(out_json), "--badge", str(badge_svg)])
    assert result.exit_code == 0
    assert "OVERALL SILICONMARK SCORE" in result.output
    assert out_json.exists()
    assert badge_svg.exists()

    out_md = tmp_path / "result.md"
    result_md = runner.invoke(app, ["run", "--device", "cpu", "--quick", "--output", str(out_md)])
    assert result_md.exit_code == 0
    assert out_md.exists()

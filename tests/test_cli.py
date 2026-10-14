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


def test_cli_run_quick(tmp_path: Path):
    out_json = tmp_path / "result.json"
    result = runner.invoke(app, ["run", "--device", "cpu", "--quick", "--output", str(out_json)])
    assert result.exit_code == 0
    assert "OVERALL SILICONMARK SCORE" in result.output
    assert out_json.exists()

    out_md = tmp_path / "result.md"
    result_md = runner.invoke(app, ["run", "--device", "cpu", "--quick", "--output", str(out_md)])
    assert result_md.exit_code == 0
    assert out_md.exists()

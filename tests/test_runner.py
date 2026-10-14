"""Unit tests for BenchmarkRunner orchestrator."""

from pathlib import Path
import pytest
from siliconmark.benchmarks.runner import BenchmarkRunner
from siliconmark.core.config import BenchmarkRunConfig


def test_runner_quick_mode(tmp_path: Path):
    runner = BenchmarkRunner(device_str="cpu", results_dir=str(tmp_path))
    cfg = BenchmarkRunConfig(quick_mode=True)

    progress_events = []
    def on_prog(stage: str, pct: float, msg: str):
        progress_events.append((stage, pct))

    result = runner.run_suite(
        config=cfg,
        progress_cb=on_prog,
        run_decode=True,
        run_prefill=True,
        run_attention=True,
    )

    assert result.run_id.startswith("sm_")
    assert result.device_info.device_type == "cpu"
    assert result.decode_result is not None
    assert result.prefill_result is not None
    assert result.attention_result is not None
    assert result.scores is not None
    assert result.scores.overall_score > 0
    assert len(progress_events) >= 3

    # Check file saved
    saved_file = tmp_path / f"{result.run_id}.json"
    assert saved_file.exists()

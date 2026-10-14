"""REST API routes for SiliconMark Web Dashboard."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException, Query, Response
from pydantic import BaseModel

from siliconmark.benchmarks.runner import BenchmarkRunner
from siliconmark.core.config import BenchmarkRunConfig, DeviceInfo, SiliconMarkRunResult
from siliconmark.core.database import HARDWARE_DATABASE
from siliconmark.core.estimator import (
    MODEL_CATALOG,
    estimate_model_performance,
)
from siliconmark.core.hardware import detect_device_info
from siliconmark.core.roofline import calculate_roofline, generate_roofline_svg
from siliconmark.exporters.markdown_exporter import export_to_markdown

router = APIRouter(prefix="/api")

# In-memory benchmark progress tracker for live web UI streaming
BENCHMARK_STATUS: Dict[str, Any] = {
    "is_running": False,
    "current_stage": "idle",
    "percentage": 0.0,
    "message": "Ready to benchmark",
    "last_run_id": None,
}


class RunRequest(BaseModel):
    quick_mode: bool = True
    device: str = "auto"
    run_decode: bool = True
    run_prefill: bool = True
    run_attention: bool = True


class EstimateRequest(BaseModel):
    model_key: str = "llama-3-8b"
    quant: str = "q4_k_m"
    context_tokens: int = 4096
    batch_size: int = 1
    device: str = "auto"


@router.get("/system", response_model=DeviceInfo)
def get_system_info(device: str = "auto") -> DeviceInfo:
    """Retrieve hardware characteristics and theoretical limits."""
    return detect_device_info(device_str=device)


@router.get("/chips")
def get_reference_chips() -> Dict[str, Any]:
    """Retrieve reference specifications for leading hardware chips."""
    return {k: v.model_dump() for k, v in HARDWARE_DATABASE.items()}


@router.get("/roofline")
def get_roofline_model(device: str = "auto") -> Dict[str, Any]:
    """Compute machine balance and Roofline SVG visualization for device."""
    dev = detect_device_info(device_str=device)
    summary = calculate_roofline(dev)
    svg_chart = generate_roofline_svg(summary)
    return {
        "summary": summary.model_dump(),
        "svg_chart": svg_chart,
    }


@router.get("/models")
def get_supported_models() -> Dict[str, Any]:
    """Retrieve catalog of supported LLM architectures for inference estimation."""
    return {k: v.model_dump() for k, v in MODEL_CATALOG.items()}


@router.post("/estimate")
def estimate_inference(req: EstimateRequest) -> Dict[str, Any]:
    """Calculate predicted VRAM consumption and tokens/sec throughput."""
    dev = detect_device_info(device_str=req.device)
    results_dir = Path("results")
    achieved_bw = dev.theoretical_bandwidth_gbps or 30.0
    achieved_tflops = dev.theoretical_tflops_fp16 or dev.theoretical_tflops_fp32 or 0.5

    runs = sorted(results_dir.glob("*.json"), reverse=True)
    if runs:
        try:
            latest = SiliconMarkRunResult.model_validate_json(runs[0].read_text(encoding="utf-8"))
            if latest.decode_result:
                achieved_bw = latest.decode_result.sustained_bandwidth_gbps
            if latest.prefill_result:
                achieved_tflops = latest.prefill_result.sustained_tflops
        except Exception:
            pass

    est = estimate_model_performance(
        model_key=req.model_key,
        quant=req.quant,
        context_tokens=req.context_tokens,
        batch_size=req.batch_size,
        achieved_decode_bw_gbps=achieved_bw,
        achieved_prefill_tflops=achieved_tflops,
        device_info=dev,
    )
    return est.model_dump()


@router.get("/status")
def get_benchmark_status() -> Dict[str, Any]:
    """Poll live benchmark execution progress."""
    return BENCHMARK_STATUS


@router.post("/benchmarks/run")
async def trigger_benchmark(req: RunRequest) -> Dict[str, Any]:
    """Execute SiliconMark benchmark suite and return completed results."""
    global BENCHMARK_STATUS
    if BENCHMARK_STATUS["is_running"]:
        raise HTTPException(status_code=409, detail="A benchmark is currently running.")

    BENCHMARK_STATUS["is_running"] = True
    BENCHMARK_STATUS["current_stage"] = "starting"
    BENCHMARK_STATUS["percentage"] = 0.0
    BENCHMARK_STATUS["message"] = "Initializing hardware..."

    def on_prog(stage: str, pct: float, msg: str) -> None:
        BENCHMARK_STATUS["current_stage"] = stage
        BENCHMARK_STATUS["percentage"] = pct
        BENCHMARK_STATUS["message"] = msg

    loop = asyncio.get_event_loop()
    try:
        runner = BenchmarkRunner(device_str=req.device)
        cfg = BenchmarkRunConfig(quick_mode=req.quick_mode)

        result: SiliconMarkRunResult = await loop.run_in_executor(
            None,
            lambda: runner.run_suite(
                config=cfg,
                progress_cb=on_prog,
                run_decode=req.run_decode,
                run_prefill=req.run_prefill,
                run_attention=req.run_attention,
            ),
        )

        BENCHMARK_STATUS["is_running"] = False
        BENCHMARK_STATUS["current_stage"] = "completed"
        BENCHMARK_STATUS["percentage"] = 100.0
        BENCHMARK_STATUS["message"] = "Benchmark completed successfully."
        BENCHMARK_STATUS["last_run_id"] = result.run_id

        return result.model_dump()

    except Exception as exc:
        BENCHMARK_STATUS["is_running"] = False
        BENCHMARK_STATUS["current_stage"] = "error"
        BENCHMARK_STATUS["message"] = str(exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/runs")
def list_past_runs() -> List[Dict[str, Any]]:
    """List historical benchmark runs from disk and standard reference baselines."""
    runs = []
    
    # 1. Search local results directory
    results_dir = Path("results")
    if results_dir.exists():
        for file in sorted(results_dir.glob("*.json"), reverse=True):
            try:
                data = file.read_text(encoding="utf-8")
                res = SiliconMarkRunResult.model_validate_json(data)
                runs.append({
                    "run_id": res.run_id,
                    "timestamp": res.timestamp,
                    "device_name": res.device_info.device_name,
                    "device_type": res.device_info.device_type,
                    "overall_score": res.scores.overall_score if res.scores else 0,
                    "tier": res.scores.tier_classification if res.scores else "Unknown",
                    "grade": res.scores.efficiency_grade if res.scores else "N/A",
                    "filename": file.name,
                    "is_reference": False,
                })
            except Exception:
                continue

    # 2. Add reference fixtures if available
    fixtures_dir = Path(__file__).resolve().parent.parent.parent.parent / "tests" / "fixtures"
    if fixtures_dir.exists():
        for file in fixtures_dir.glob("*.json"):
            try:
                data = file.read_text(encoding="utf-8")
                res = SiliconMarkRunResult.model_validate_json(data)
                runs.append({
                    "run_id": res.run_id,
                    "timestamp": res.timestamp,
                    "device_name": f"{res.device_info.device_name} [Ref]",
                    "device_type": res.device_info.device_type,
                    "overall_score": res.scores.overall_score if res.scores else 0,
                    "tier": res.scores.tier_classification if res.scores else "Unknown",
                    "grade": res.scores.efficiency_grade if res.scores else "N/A",
                    "filename": file.name,
                    "is_reference": True,
                })
            except Exception:
                continue

    return runs


@router.get("/runs/{run_id}")
def get_run_details(run_id: str) -> Dict[str, Any]:
    """Retrieve full result payload for a specific run ID or reference."""
    results_dir = Path("results")
    fixtures_dir = Path(__file__).resolve().parent.parent.parent.parent / "tests" / "fixtures"

    target: Optional[Path] = None

    # Check results dir
    if results_dir.exists():
        if (results_dir / f"{run_id}.json").exists():
            target = results_dir / f"{run_id}.json"
        else:
            matches = list(results_dir.glob(f"*{run_id}*.json"))
            if matches:
                target = matches[0]

    # Check fixtures dir
    if not target and fixtures_dir.exists():
        if (fixtures_dir / f"{run_id}.json").exists():
            target = fixtures_dir / f"{run_id}.json"
        else:
            matches = list(fixtures_dir.glob(f"*{run_id}*.json"))
            if matches:
                target = matches[0]

    if not target or not target.exists():
        raise HTTPException(status_code=404, detail="Run not found.")

    data = target.read_text(encoding="utf-8")
    return SiliconMarkRunResult.model_validate_json(data).model_dump()


@router.get("/runs/{run_id}/markdown")
def download_run_markdown(run_id: str) -> Response:
    """Export benchmark run as Markdown report."""
    res_dict = get_run_details(run_id)
    run_res = SiliconMarkRunResult.model_validate(res_dict)
    
    # Generate in-memory markdown
    from io import StringIO
    import tempfile
    
    with tempfile.NamedTemporaryFile("w+", suffix=".md", delete=False, encoding="utf-8") as tmp:
        tmp_path = Path(tmp.name)
    
    export_to_markdown(run_res, tmp_path)
    content = tmp_path.read_text(encoding="utf-8")
    tmp_path.unlink(missing_ok=True)
    
    return Response(
        content=content,
        media_type="text/markdown",
        headers={"Content-Disposition": f'attachment; filename="{run_id}_report.md"'},
    )

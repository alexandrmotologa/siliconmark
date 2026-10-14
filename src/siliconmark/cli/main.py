"""Rich Typer CLI for SiliconMark hardware benchmarking suite."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional
import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table

from siliconmark.benchmarks.runner import BenchmarkRunner
from siliconmark.core.config import BenchmarkRunConfig, SiliconMarkRunResult
from siliconmark.core.database import HARDWARE_DATABASE
from siliconmark.core.estimator import (
    MODEL_CATALOG,
    QUANTIZATION_BYTES,
    estimate_model_performance,
)
from siliconmark.core.hardware import detect_device_info
from siliconmark.core.roofline import (
    calculate_roofline,
    generate_ascii_roofline,
    generate_roofline_svg,
)
from siliconmark.exporters.markdown_exporter import export_to_markdown

app = typer.Typer(
    name="siliconmark",
    help="Bare-metal hardware benchmark suite measuring LLM prefill TFLOPS, decode memory bandwidth, and attention efficiency.",
    add_completion=False,
)
console = Console()


def _resolve_result_file(target: str) -> Path:
    """Resolve file path or named reference aliases (rtx4090, m3max, latest)."""
    t_lower = target.lower()
    fixtures_dir = Path(__file__).resolve().parent.parent.parent.parent / "tests" / "fixtures"

    if t_lower == "rtx4090":
        p = fixtures_dir / "rtx4090.json"
        if p.exists():
            return p
    elif t_lower == "m3max":
        p = fixtures_dir / "m3max.json"
        if p.exists():
            return p
    elif t_lower == "latest":
        results_dir = Path("results")
        runs = sorted(results_dir.glob("*.json"), reverse=True)
        if runs:
            return runs[0]

    path = Path(target)
    if path.exists():
        return path

    # Check in results dir
    results_path = Path("results") / target
    if results_path.exists():
        return results_path
    if (Path("results") / f"{target}.json").exists():
        return Path("results") / f"{target}.json"

    return path


@app.command()
def info(
    device: str = typer.Option("auto", "--device", "-d", help="Compute device: auto, cpu, cuda, mps, rocm"),
    device_index: int = typer.Option(0, "--index", "-i", help="Device index for multi-GPU setups"),
) -> None:
    """Detect and inspect active hardware, memory geometry, and theoretical limits."""
    with console.status("[bold cyan]Probing bare-metal hardware...[/bold cyan]"):
        dev = detect_device_info(device_str=device, device_index=device_index)

    table = Table(title="[bold yellow]SiliconMark Hardware Inspector[/bold yellow]", show_header=True)
    table.add_column("Property", style="cyan", no_wrap=True)
    table.add_column("Value", style="magenta")

    table.add_row("Device Name", dev.device_name)
    table.add_row("Backend", dev.device_type.value.upper())
    table.add_row("Platform", dev.os_platform)
    table.add_row("Total Memory", f"{dev.total_memory_gb:.2f} GB")
    table.add_row("Available Memory", f"{dev.available_memory_gb:.2f} GB")
    table.add_row("Compute Units (Cores/SMs)", str(dev.compute_units or "N/A"))
    if dev.logical_cores:
        table.add_row("Logical Threads", str(dev.logical_cores))
    if dev.cuda_compute_capability:
        table.add_row("CUDA Compute Capability", dev.cuda_compute_capability)
    table.add_row("Theoretical Bandwidth", f"{dev.theoretical_bandwidth_gbps or 'N/A'} GB/s")
    table.add_row("Theoretical FP16 Compute", f"{dev.theoretical_tflops_fp16 or 'N/A'} TFLOPS")
    table.add_row("Theoretical FP32 Compute", f"{dev.theoretical_tflops_fp32 or 'N/A'} TFLOPS")
    table.add_row("PyTorch Version", dev.torch_version)

    console.print(table)


@app.command()
def run(
    device: str = typer.Option("auto", "--device", "-d", help="Compute device: auto, cpu, cuda, mps, rocm"),
    device_index: int = typer.Option(0, "--index", "-i", help="Device index for multi-GPU setups"),
    quick: bool = typer.Option(False, "--quick", "-q", help="Run abbreviated benchmark matrix"),
    decode_only: bool = typer.Option(False, "--decode-only", help="Run only decode memory bandwidth benchmark"),
    prefill_only: bool = typer.Option(False, "--prefill-only", help="Run only prefill TFLOPS benchmark"),
    attention_only: bool = typer.Option(False, "--attention-only", help="Run only attention kernel efficiency benchmark"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Save benchmark results to file path (.json or .md)"),
    ci: bool = typer.Option(False, "--ci", help="Output GitHub Actions step summary format to stdout"),
) -> None:
    """Execute SiliconMark benchmarks and output standardized composite score."""
    runner = BenchmarkRunner(device_str=device, device_index=device_index)
    cfg = BenchmarkRunConfig(quick_mode=quick)

    run_decode = True
    run_prefill = True
    run_attention = True

    if decode_only or prefill_only or attention_only:
        run_decode = decode_only
        run_prefill = prefill_only
        run_attention = attention_only

    if not ci:
        console.print(
            Panel.fit(
                f"[bold green]Starting SiliconMark Benchmark Suite[/bold green]\n"
                f"Target: [cyan]{runner.device_info.device_name}[/cyan] ([yellow]{runner.device_info.device_type.upper()}[/yellow])\n"
                f"Mode: [magenta]{'Quick (abbreviated matrix)' if quick else 'Standard Comprehensive'}[/magenta]",
                title="[bold yellow]SiliconMark[/bold yellow]",
                border_style="yellow",
            )
        )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
        disable=ci,
    ) as progress:
        task = progress.add_task("[cyan]Benchmarking hardware...", total=100)

        def on_prog(stage: str, pct: float, msg: str) -> None:
            progress.update(task, completed=pct, description=f"[cyan]{msg}[/cyan]")

        result = runner.run_suite(
            config=cfg,
            progress_cb=on_prog,
            run_decode=run_decode,
            run_prefill=run_prefill,
            run_attention=run_attention,
        )
        progress.update(task, completed=100, description="[bold green]Benchmark completed![/bold green]")

    # Print Results Table
    scores = result.scores
    if scores and not ci:
        score_panel = (
            f"[bold yellow]OVERALL SILICONMARK SCORE:[/bold yellow] [bold white]{scores.overall_score:,}[/bold white]\n"
            f"[bold cyan]Classification:[/bold cyan] {scores.tier_classification}\n"
            f"[bold green]Efficiency Grade:[/bold green] [bold magenta]{scores.efficiency_grade}[/bold magenta]\n\n"
            f"• Decode Bandwidth Score: [bold]{scores.decode_score:,}[/bold] (45% weight)\n"
            f"• Prefill Compute Score:  [bold]{scores.prefill_score:,}[/bold] (35% weight)\n"
            f"• Attention Efficiency:   [bold]{scores.attention_score:,}[/bold] (20% weight)\n\n"
            f"Index vs RTX 4090 baseline: [cyan]{scores.relative_to_rtx4090_pct}%[/cyan]\n"
            f"Index vs Apple M3 Max:      [cyan]{scores.relative_to_m3max_pct}%[/cyan]"
        )
        console.print(Panel(score_panel, title="[bold yellow]SiliconMark Scorecard[/bold yellow]", border_style="green"))

    # Print Summary Metrics
    if not ci:
        table = Table(title="[bold]Summary Performance Metrics[/bold]")
        table.add_column("Benchmark Component", style="cyan")
        table.add_column("Key Measurement", style="green")
        table.add_column("Saturation / Efficiency", style="yellow")

        if result.decode_result:
            d = result.decode_result
            sat_text = f"{d.bandwidth_saturation_pct:.1f}%" if d.bandwidth_saturation_pct else "N/A"
            table.add_row(
                "Decode Memory Bandwidth",
                f"Peak: {d.peak_bandwidth_gbps:.2f} GB/s | Sustained: {d.sustained_bandwidth_gbps:.2f} GB/s",
                f"Bus Saturation: {sat_text}",
            )

        if result.prefill_result:
            p = result.prefill_result
            eff_text = f"{p.tflops_efficiency_pct:.1f}%" if p.tflops_efficiency_pct else "N/A"
            table.add_row(
                "Prefill Compute",
                f"Peak: {p.peak_tflops:.3f} TFLOPS | Sustained: {p.sustained_tflops:.3f} TFLOPS",
                f"Compute Eff: {eff_text}",
            )

        if result.attention_result:
            a = result.attention_result
            table.add_row(
                "Attention Kernel",
                f"SDPA: {a.sdpa_peak_tflops:.3f} TFLOPS vs Naive: {a.naive_peak_tflops:.3f} TFLOPS",
                f"Mean Speedup: {a.mean_speedup:.2f}x",
            )

        console.print(table)
        console.print(f"[dim]Run ID: {result.run_id} | Saved to: results/{result.run_id}.json[/dim]")

    if ci and scores:
        console.print(f"## SiliconMark Hardware Benchmark: {result.device_info.device_name}")
        console.print(f"- Overall Score: **{scores.overall_score:,}** ({scores.efficiency_grade})")
        console.print(f"- Classification: {scores.tier_classification}")
        if result.decode_result:
            console.print(f"- Decode Bandwidth: {result.decode_result.sustained_bandwidth_gbps:.2f} GB/s")
        if result.prefill_result:
            console.print(f"- Prefill TFLOPS: {result.prefill_result.sustained_tflops:.3f} TFLOPS")

    if output:
        out_p = Path(output)
        if out_p.suffix.lower() == ".json":
            out_p.write_text(result.model_dump_json(indent=2), encoding="utf-8")
            console.print(f"[bold green]Saved JSON report to {output}[/bold green]")
        else:
            export_to_markdown(result, out_p)
            console.print(f"[bold green]Saved Markdown report to {output}[/bold green]")


@app.command()
def roofline(
    device: str = typer.Option("auto", "--device", "-d", help="Compute device: auto, cpu, cuda, mps, rocm"),
    svg: Optional[Path] = typer.Option(None, "--svg", help="Export standalone SVG roofline chart to file"),
) -> None:
    """Analyze bare-metal Roofline model, machine balance, and operational intensity."""
    dev = detect_device_info(device_str=device)
    summary = calculate_roofline(dev)

    console.print(generate_ascii_roofline(summary))

    if svg:
        svg_content = generate_roofline_svg(summary)
        svg.write_text(svg_content, encoding="utf-8")
        console.print(f"[bold green]Saved Roofline SVG chart to {svg}[/bold green]")


@app.command()
def estimate(
    model: str = typer.Option("llama-3-8b", "--model", "-m", help="Model key (e.g. llama-3-8b, llama-3-70b, mistral-7b, qwen-2.5-32b)"),
    quant: str = typer.Option("q4_k_m", "--quant", "-q", help="Quantization format (fp16, q8_0, q4_k_m, q2_k)"),
    context: int = typer.Option(4096, "--context", "-c", help="Context length in tokens"),
    batch_size: int = typer.Option(1, "--batch", "-b", help="Concurrent inference batch size"),
    device: str = typer.Option("auto", "--device", "-d", help="Probed compute device for memory capacity"),
) -> None:
    """Estimate real-world LLM inference throughput and VRAM footprint based on benchmark data."""
    dev = detect_device_info(device_str=device)

    # Use latest benchmark run if available, else derive from theoretical bandwidth
    results_dir = Path("results")
    achieved_bw = dev.theoretical_bandwidth_gbps or 30.0
    achieved_tflops = dev.theoretical_tflops_fp16 or dev.theoretical_tflops_fp32 or 0.5

    runs = sorted(results_dir.glob("*.json"), reverse=True)
    if runs:
        try:
            latest_res = SiliconMarkRunResult.model_validate_json(runs[0].read_text(encoding="utf-8"))
            if latest_res.decode_result:
                achieved_bw = latest_res.decode_result.sustained_bandwidth_gbps
            if latest_res.prefill_result:
                achieved_tflops = latest_res.prefill_result.sustained_tflops
        except Exception:
            pass

    est = estimate_model_performance(
        model_key=model,
        quant=quant,
        context_tokens=context,
        batch_size=batch_size,
        achieved_decode_bw_gbps=achieved_bw,
        achieved_prefill_tflops=achieved_tflops,
        device_info=dev,
    )

    table = Table(title=f"[bold yellow]SiliconMark LLM Inference Estimator: {est.model_name}[/bold yellow]")
    table.add_column("Parameter / Metric", style="cyan")
    table.add_column("Value", style="magenta")

    table.add_row("Model Architecture", est.model_name)
    table.add_row("Quantization", est.quantization)
    table.add_row("Context Window", f"{est.context_tokens:,} tokens")
    table.add_row("Weights VRAM", f"{est.model_weights_gb:.2f} GB")
    table.add_row("KV Cache VRAM (FP16)", f"{est.kv_cache_gb:.2f} GB")
    table.add_row("Total Required VRAM", f"{est.total_vram_required_gb:.2f} GB")
    table.add_row("Available Device VRAM", f"{est.available_vram_gb:.2f} GB")
    
    fit_style = "bold green" if est.can_fit else "bold red"
    table.add_row("Memory Fit Status", f"[{fit_style}]{est.fit_assessment}[/{fit_style}]")

    table.add_row("Expected Decode Speed", f"[bold green]{est.decode_tokens_per_sec:.1f} tokens/sec[/bold green]")
    table.add_row("Inter-Token Latency", f"{est.inter_token_latency_ms:.1f} ms/token")
    table.add_row("Expected Prefill Speed", f"{est.prefill_tokens_per_sec:,.0f} tokens/sec")
    table.add_row("Time to First Token (TTFT)", f"{est.time_to_first_token_sec:.2f} seconds")

    console.print(table)


@app.command("list-models")
def list_models() -> None:
    """Catalog of supported open-weight LLMs available for inference estimation."""
    table = Table(title="[bold yellow]SiliconMark Supported Model Architectures[/bold yellow]")
    table.add_column("Model Key", style="cyan")
    table.add_column("Name", style="green")
    table.add_column("Total Params", style="magenta")
    table.add_column("Active Params", style="yellow")
    table.add_column("Layers / Heads / GQA", style="white")

    for k, m in MODEL_CATALOG.items():
        gqa = f"{m.num_heads}Q / {m.num_kv_heads}KV"
        table.add_row(
            k,
            m.name,
            f"{m.total_params_billion:.1f}B",
            f"{m.active_params_billion:.1f}B",
            f"{m.num_layers}L • {gqa}",
        )

    console.print(table)


@app.command("list-hardware")
def list_hardware() -> None:
    """Display catalog of reference specifications for leading LLM chips."""
    table = Table(title="[bold yellow]SiliconMark Reference Chip Database[/bold yellow]")
    table.add_column("Chip Name", style="cyan")
    table.add_column("Vendor", style="green")
    table.add_column("Memory", style="magenta")
    table.add_column("Bus Width", style="yellow")
    table.add_column("Peak Bandwidth", style="bold green")
    table.add_column("FP16 Peak", style="bold white")

    for spec in HARDWARE_DATABASE.values():
        table.add_row(
            spec.name,
            spec.vendor,
            f"{spec.vram_gb:.0f}GB {spec.memory_type}",
            f"{spec.bus_width_bits}-bit",
            f"{spec.theoretical_bandwidth_gbps:,.0f} GB/s",
            f"{spec.theoretical_tflops_fp16:,.1f} TFLOPS",
        )

    console.print(table)


@app.command()
def compare(
    file1: str = typer.Argument(..., help="Path or alias (rtx4090, m3max, latest) for first result"),
    file2: str = typer.Argument(..., help="Path or alias (rtx4090, m3max, latest) for second result"),
) -> None:
    """Compare two SiliconMark benchmark runs side by side."""
    p1 = _resolve_result_file(file1)
    p2 = _resolve_result_file(file2)

    if not p1.exists():
        console.print(f"[bold red]Error: Could not find file {file1}[/bold red]")
        raise typer.Exit(code=1)
    if not p2.exists():
        console.print(f"[bold red]Error: Could not find file {file2}[/bold red]")
        raise typer.Exit(code=1)

    res1 = SiliconMarkRunResult.model_validate_json(p1.read_text(encoding="utf-8"))
    res2 = SiliconMarkRunResult.model_validate_json(p2.read_text(encoding="utf-8"))

    table = Table(title="[bold yellow]SiliconMark Comparison[/bold yellow]")
    table.add_column("Metric", style="cyan")
    table.add_column(f"{res1.device_info.device_name}", style="magenta")
    table.add_column(f"{res2.device_info.device_name}", style="green")
    table.add_column("Delta (%)", style="bold yellow")

    def add_diff(label: str, v1: Optional[float], v2: Optional[float], unit: str = "") -> None:
        if v1 is None or v2 is None:
            return
        diff = ((v2 - v1) / v1 * 100.0) if v1 != 0 else 0.0
        diff_str = f"+{diff:.1f}%" if diff > 0 else f"{diff:.1f}%"
        color = "green" if diff > 0 else ("red" if diff < 0 else "white")
        table.add_row(label, f"{v1:.2f} {unit}", f"{v2:.2f} {unit}", f"[{color}]{diff_str}[/{color}]")

    if res1.scores and res2.scores:
        add_diff("Overall Score", float(res1.scores.overall_score), float(res2.scores.overall_score), "pts")
        add_diff("Decode Score", float(res1.scores.decode_score), float(res2.scores.decode_score), "pts")
        add_diff("Prefill Score", float(res1.scores.prefill_score), float(res2.scores.prefill_score), "pts")
        add_diff("Attention Score", float(res1.scores.attention_score), float(res2.scores.attention_score), "pts")

    if res1.decode_result and res2.decode_result:
        add_diff("Peak Decode Bandwidth", res1.decode_result.peak_bandwidth_gbps, res2.decode_result.peak_bandwidth_gbps, "GB/s")
        add_diff("Sustained Decode Bandwidth", res1.decode_result.sustained_bandwidth_gbps, res2.decode_result.sustained_bandwidth_gbps, "GB/s")

    if res1.prefill_result and res2.prefill_result:
        add_diff("Peak Prefill TFLOPS", res1.prefill_result.peak_tflops, res2.prefill_result.peak_tflops, "TFLOPS")

    console.print(table)


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", "--host", "-h", help="Bind host address"),
    port: int = typer.Option(8000, "--port", "-p", help="Bind port"),
    reload: bool = typer.Option(False, "--reload", help="Enable auto-reload for development"),
) -> None:
    """Launch the SiliconMark interactive Web dashboard and API server."""
    import uvicorn
    console.print(f"[bold green]Starting SiliconMark Web Dashboard on http://{host}:{port}[/bold green]")
    uvicorn.run("siliconmark.web.app:app", host=host, port=port, reload=reload)


if __name__ == "__main__":
    app()

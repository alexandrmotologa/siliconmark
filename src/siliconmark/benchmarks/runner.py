"""Unified BenchmarkRunner coordinating complete SiliconMark execution suites."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Callable, Optional
import uuid

from siliconmark.benchmarks.attention import AttentionKernelBenchmark
from siliconmark.benchmarks.context import ContextScalingBenchmark
from siliconmark.benchmarks.decode import DecodeMemoryBandwidthBenchmark
from siliconmark.benchmarks.interconnect import InterconnectBenchmark
from siliconmark.benchmarks.prefill import PrefillTflopsBenchmark
from siliconmark.benchmarks.quantized import QuantizedKernelBenchmark
from siliconmark.core.config import (
    AttentionBenchmarkConfig,
    BenchmarkRunConfig,
    ContextScalingResult,
    DecodeBenchmarkConfig,
    DeviceInfo,
    EnergyMeasurement,
    InterconnectResult,
    PrefillBenchmarkConfig,
    QuantizedBenchmarkResult,
    SiliconMarkRunResult,
)
from siliconmark.core.hardware import resolve_torch_device
from siliconmark.core.power import PowerMonitor
from siliconmark.core.scoring import calculate_siliconmark_scores


ProgressCallback = Callable[[str, float, str], None]


class BenchmarkRunner:
    """Orchestrates SiliconMark suite execution across hardware targets."""

    def __init__(
        self,
        device_str: str = "auto",
        device_index: int = 0,
        results_dir: str = "results",
    ) -> None:
        self.device, self.device_info = resolve_torch_device(
            device_str=device_str, device_index=device_index
        )
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def run_suite(
        self,
        config: Optional[BenchmarkRunConfig] = None,
        progress_cb: Optional[ProgressCallback] = None,
        run_decode: bool = True,
        run_prefill: bool = True,
        run_attention: bool = True,
        measure_power: bool = False,
    ) -> SiliconMarkRunResult:
        """Execute requested benchmark components and compute composite SiliconMark scores."""
        if config is None:
            config = BenchmarkRunConfig()

        # Adjust configs if quick mode is enabled
        decode_cfg = config.decode
        prefill_cfg = config.prefill
        attn_cfg = config.attention

        if config.quick_mode:
            decode_cfg = DecodeBenchmarkConfig(
                batch_sizes=[1, 4],
                hidden_dims=[2048, 4096],
                seq_lens=[512],
                warmup_iters=2,
                bench_iters=5,
                include_paged_kv=True,
            )
            prefill_cfg = PrefillBenchmarkConfig(
                batch_sizes=[1],
                prompt_lens=[256, 1024],
                hidden_dims=[2048, 4096],
                warmup_iters=2,
                bench_iters=5,
            )
            attn_cfg = AttentionBenchmarkConfig(
                seq_lens=[512, 1024, 2048],
                num_heads=16,
                head_dim=128,
                batch_size=1,
                warmup_iters=2,
                bench_iters=5,
            )

        run_id = f"sm_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        timestamp = datetime.now(timezone.utc).isoformat()

        decode_res = None
        prefill_res = None
        attn_res = None
        energy_res: Optional[EnergyMeasurement] = None

        power_monitor: Optional[PowerMonitor] = None
        if measure_power:
            power_monitor = PowerMonitor(self.device_info)
            power_monitor.start()

        total_steps = sum([run_decode, run_prefill, run_attention])
        curr_step = 0

        try:
            # 1. Decode memory bandwidth
            if run_decode:
                if progress_cb:
                    pct = (curr_step / total_steps) * 100.0
                    progress_cb("decode", pct, "Benchmarking decode memory bandwidth saturation...")
                decode_bench = DecodeMemoryBandwidthBenchmark(self.device, self.device_info)
                decode_res = decode_bench.run(decode_cfg)
                curr_step += 1

            # 2. Prefill compute TFLOPS
            if run_prefill:
                if progress_cb:
                    pct = (curr_step / total_steps) * 100.0
                    progress_cb("prefill", pct, "Benchmarking prompt prefill TFLOPS...")
                prefill_bench = PrefillTflopsBenchmark(self.device, self.device_info)
                prefill_res = prefill_bench.run(prefill_cfg)
                curr_step += 1

            # 3. Attention kernel efficiency
            if run_attention:
                if progress_cb:
                    pct = (curr_step / total_steps) * 100.0
                    progress_cb("attention", pct, "Benchmarking SDPA attention kernels...")
                attn_bench = AttentionKernelBenchmark(self.device, self.device_info)
                attn_res = attn_bench.run(attn_cfg)
                curr_step += 1
        finally:
            if power_monitor:
                approx_tokens = 2048 if run_decode else None
                approx_tflops = prefill_res.sustained_tflops if prefill_res else None
                energy_res = power_monitor.stop(
                    tokens_processed=approx_tokens,
                    achieved_tflops=approx_tflops,
                )

        if progress_cb:
            progress_cb("scoring", 100.0, "Calculating standardized SiliconMark scores...")

        scores = calculate_siliconmark_scores(
            device_info=self.device_info,
            decode_res=decode_res,
            prefill_res=prefill_res,
            attn_res=attn_res,
        )

        run_result = SiliconMarkRunResult(
            run_id=run_id,
            timestamp=timestamp,
            device_info=self.device_info,
            decode_result=decode_res,
            prefill_result=prefill_res,
            attention_result=attn_res,
            energy_result=energy_res,
            scores=scores,
        )

        # Save to results directory
        save_path = self.results_dir / f"{run_id}.json"
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(run_result.model_dump_json(indent=2))

        return run_result

    def run_context_scaling(
        self,
        quick_mode: bool = False,
        progress_cb: Optional[Callable[[float, str], None]] = None,
        max_context: Optional[int] = None,
    ) -> ContextScalingResult:
        """Execute context length scaling sweep."""
        bench = ContextScalingBenchmark(self.device, self.device_info)
        return bench.run(quick_mode=quick_mode, progress_cb=progress_cb, max_context=max_context)

    def run_quantized(
        self,
        quick_mode: bool = False,
        progress_cb: Optional[Callable[[float, str], None]] = None,
    ) -> QuantizedBenchmarkResult:
        """Execute INT8 and packed INT4 quantized kernel microbenchmark."""
        bench = QuantizedKernelBenchmark(self.device, self.device_info)
        return bench.run(quick_mode=quick_mode, progress_cb=progress_cb)

    def run_interconnect(
        self,
        progress_cb: Optional[Callable[[float, str], None]] = None,
    ) -> InterconnectResult:
        """Execute Multi-GPU P2P or Host interconnect communication benchmark."""
        bench = InterconnectBenchmark(self.device_info)
        return bench.run(progress_cb=progress_cb)

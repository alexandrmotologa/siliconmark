"""Unit tests for Quantized Kernel Microbenchmark (INT8/INT4)."""

from siliconmark.benchmarks.quantized import QuantizedKernelBenchmark
from siliconmark.core.hardware import resolve_torch_device


def test_quantized_benchmark_execution():
    device, dev = resolve_torch_device("cpu")
    bench = QuantizedKernelBenchmark(device, dev, warmup_iters=1, bench_iters=2)

    res = bench.run(quick_mode=True)

    assert res.device_name == dev.device_name
    assert len(res.points) == 4  # 2 configs x 2 formats (INT8, INT4)
    assert res.int8_peak_tops >= 0.0
    assert res.int4_peak_tops >= 0.0
    assert res.peak_speedup_vs_fp16 > 0.0

    for pt in res.points:
        assert pt.latency_ms > 0
        assert pt.achieved_tops >= 0
        assert pt.effective_bandwidth_gbps >= 0
        assert pt.speedup_vs_fp16 >= 0

"""Unit tests for Context Scaling Curve Benchmark."""

from siliconmark.benchmarks.context import ContextScalingBenchmark, generate_context_scaling_svg
from siliconmark.core.hardware import resolve_torch_device


def test_context_scaling_execution():
    device, dev = resolve_torch_device("cpu")
    bench = ContextScalingBenchmark(device, dev, warmup_iters=1, bench_iters=2)

    # Test quick mode with max context 2048 to keep test fast
    res = bench.run(quick_mode=True, max_context=2048)

    assert res.device_name == dev.device_name
    assert len(res.points) > 0
    first_pt = res.points[0]
    assert first_pt.seq_len == 1024
    assert first_pt.latency_ms > 0
    assert first_pt.sdpa_tflops >= 0
    assert first_pt.kv_cache_mb > 0
    assert res.max_tested_context >= 1024
    assert res.scaling_grade != ""

    # Test SVG generation
    svg = generate_context_scaling_svg(res)
    assert "<svg" in svg
    assert "Context Length Scaling Curve" in svg

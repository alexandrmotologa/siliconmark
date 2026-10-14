"""Unit tests for Interconnect & P2P Communication Benchmark."""

from siliconmark.benchmarks.interconnect import InterconnectBenchmark
from siliconmark.core.hardware import detect_device_info


def test_interconnect_benchmark_execution():
    dev = detect_device_info("cpu")
    bench = InterconnectBenchmark(dev)

    res = bench.run(buffer_size_mb=16, warmup_iters=1, bench_iters=3)

    assert res.device_count >= 1
    assert res.unidirectional_bandwidth_gbps > 0.0
    assert res.bidirectional_bandwidth_gbps > 0.0
    assert res.all_reduce_latency_us > 0.0
    assert res.topology_type != ""
    assert res.summary_notes != ""

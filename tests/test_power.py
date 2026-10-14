"""Unit tests for Power & Energy Efficiency Profiling Engine."""

import time
from siliconmark.core.hardware import detect_device_info
from siliconmark.core.power import PowerMonitor


def test_power_monitor_cpu_telemetry():
    dev = detect_device_info("cpu")
    monitor = PowerMonitor(dev, sampling_interval_sec=0.02)

    monitor.start()
    time.sleep(0.1)  # allow samples to accumulate
    res = monitor.stop(tokens_processed=500, achieved_tflops=0.25)

    assert res.average_power_watts > 0.0
    assert res.peak_power_watts >= res.average_power_watts
    assert res.energy_consumed_joules > 0.0
    assert res.tokens_per_joule is not None
    assert res.tokens_per_joule > 0.0
    assert res.tflops_per_watt is not None
    assert res.tflops_per_watt > 0.0
    assert res.sensor_source != ""

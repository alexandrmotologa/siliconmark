"""Unit tests for SiliconMark hardware detector and chip database."""

import pytest
from siliconmark.core.database import find_chip_spec, estimate_cpu_bandwidth, HARDWARE_DATABASE
from siliconmark.core.hardware import detect_device_info, resolve_torch_device
from siliconmark.core.config import DeviceType


def test_chip_database_lookup():
    rtx4090 = find_chip_spec("NVIDIA GeForce RTX 4090")
    assert rtx4090 is not None
    assert rtx4090.vendor == "NVIDIA"
    assert rtx4090.theoretical_bandwidth_gbps == 1008.0

    m3max = find_chip_spec("Apple M3 Max")
    assert m3max is not None
    assert m3max.vendor == "Apple"
    assert m3max.theoretical_bandwidth_gbps == 400.0

    h100 = find_chip_spec("NVIDIA H100 SXM5 80GB")
    assert h100 is not None
    assert h100.theoretical_tflops_fp16 == 989.0


def test_cpu_bandwidth_estimator():
    bw_small = estimate_cpu_bandwidth(memory_gb=16.0, num_cores=4)
    assert bw_small == 51.2

    bw_server = estimate_cpu_bandwidth(memory_gb=128.0, num_cores=64)
    assert bw_server == 384.0


def test_detect_device_info_cpu():
    dev = detect_device_info(device_str="cpu")
    assert dev.device_type == DeviceType.CPU
    assert dev.total_memory_gb > 0
    assert dev.compute_units is not None
    assert dev.theoretical_bandwidth_gbps is not None
    assert dev.torch_version != ""


def test_resolve_torch_device():
    dev, info = resolve_torch_device(device_str="cpu")
    assert dev.type == "cpu"
    assert info.device_type == DeviceType.CPU

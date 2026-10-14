"""Hardware detector for NVIDIA CUDA, AMD ROCm, Apple Silicon (MPS), and CPUs.

Probes device capabilities, memory geometry, active compute units,
and theoretical compute/bandwidth limits without external dependencies.
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from typing import Optional, Tuple
import psutil
import torch

from siliconmark.core.config import DeviceInfo, DeviceType
from siliconmark.core.database import find_chip_spec, estimate_cpu_bandwidth


def _get_cpu_name() -> str:
    """Retrieve full human-readable processor model string."""
    name = platform.processor() or ""
    if sys.platform == "win32":
        try:
            import winreg
            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
            )
            processor_name, _ = winreg.QueryValueEx(key, "ProcessorNameString")
            winreg.CloseKey(key)
            if processor_name:
                return str(processor_name).strip()
        except Exception:
            pass
    elif sys.platform == "darwin":
        try:
            out = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"]).decode()
            if out.strip():
                return out.strip()
        except Exception:
            pass
    elif sys.platform.startswith("linux"):
        try:
            with open("/proc/cpuinfo", "r", encoding="utf-8") as f:
                for line in f:
                    if "model name" in line:
                        return line.split(":", 1)[1].strip()
        except Exception:
            pass

    return name if name else f"CPU ({platform.machine()})"


def detect_device_info(device_str: str = "auto", device_index: int = 0) -> DeviceInfo:
    """Detect hardware metadata and theoretical limits for requested or default device.

    Args:
        device_str: "auto", "cuda", "mps", "rocm", or "cpu".
        device_index: Device index for multi-GPU nodes.

    Returns:
        DeviceInfo model containing probed hardware specs.
    """
    selected_type: DeviceType
    os_name = f"{platform.system()} {platform.release()} ({platform.machine()})"
    py_ver = platform.python_version()
    torch_ver = torch.__version__

    # 1. Device selection
    if device_str == "auto":
        if torch.cuda.is_available():
            # Check if running ROCm (HIP) or CUDA
            if hasattr(torch.version, "hip") and torch.version.hip is not None:
                selected_type = DeviceType.ROCM
            else:
                selected_type = DeviceType.CUDA
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            selected_type = DeviceType.MPS
        else:
            selected_type = DeviceType.CPU
    else:
        selected_type = DeviceType(device_str.lower())

    # 2. CUDA / ROCm GPU Detection
    if selected_type in (DeviceType.CUDA, DeviceType.ROCM) and torch.cuda.is_available():
        props = torch.cuda.get_device_properties(device_index)
        dev_name = props.name
        total_bytes = props.total_memory
        total_gb = round(total_bytes / (1024 ** 3), 2)

        # Free / available memory
        try:
            free_bytes, _ = torch.cuda.mem_get_info(device_index)
        except Exception:
            free_bytes = total_bytes
        avail_gb = round(free_bytes / (1024 ** 3), 2)

        compute_units = getattr(props, "multi_processor_count", None)
        major = getattr(props, "major", None)
        minor = getattr(props, "minor", None)
        cc = f"{major}.{minor}" if major is not None and minor is not None else None

        spec = find_chip_spec(dev_name)

        theoretical_bw = spec.theoretical_bandwidth_gbps if spec else None
        theoretical_fp16 = spec.theoretical_tflops_fp16 if spec else None
        theoretical_fp32 = spec.theoretical_tflops_fp32 if spec else None
        bus_width = spec.bus_width_bits if spec else None

        return DeviceInfo(
            device_type=selected_type,
            device_name=dev_name,
            device_index=device_index,
            total_memory_bytes=total_bytes,
            total_memory_gb=total_gb,
            available_memory_bytes=free_bytes,
            available_memory_gb=avail_gb,
            compute_units=compute_units,
            memory_bus_width_bits=bus_width,
            theoretical_bandwidth_gbps=theoretical_bw,
            theoretical_tflops_fp16=theoretical_fp16,
            theoretical_tflops_fp32=theoretical_fp32,
            cuda_compute_capability=cc,
            os_platform=os_name,
            python_version=py_ver,
            torch_version=torch_ver,
        )

    # 3. Apple Silicon (MPS) Detection
    if selected_type == DeviceType.MPS and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        vmem = psutil.virtual_memory()
        total_bytes = vmem.total
        avail_bytes = vmem.available
        total_gb = round(total_bytes / (1024 ** 3), 2)
        avail_gb = round(avail_bytes / (1024 ** 3), 2)

        dev_name = _get_cpu_name()
        if "Apple" not in dev_name:
            dev_name = f"Apple Silicon ({platform.processor() or 'M-Series'})"

        spec = find_chip_spec(dev_name)
        theoretical_bw = spec.theoretical_bandwidth_gbps if spec else 100.0
        theoretical_fp16 = spec.theoretical_tflops_fp16 if spec else None
        theoretical_fp32 = spec.theoretical_tflops_fp32 if spec else None

        return DeviceInfo(
            device_type=DeviceType.MPS,
            device_name=dev_name,
            device_index=0,
            total_memory_bytes=total_bytes,
            total_memory_gb=total_gb,
            available_memory_bytes=avail_bytes,
            available_memory_gb=avail_gb,
            compute_units=spec.compute_units if spec else None,
            memory_bus_width_bits=spec.bus_width_bits if spec else None,
            theoretical_bandwidth_gbps=theoretical_bw,
            theoretical_tflops_fp16=theoretical_fp16,
            theoretical_tflops_fp32=theoretical_fp32,
            os_platform=os_name,
            python_version=py_ver,
            torch_version=torch_ver,
        )

    # 4. CPU Detection
    vmem = psutil.virtual_memory()
    total_bytes = vmem.total
    avail_bytes = vmem.available
    total_gb = round(total_bytes / (1024 ** 3), 2)
    avail_gb = round(avail_bytes / (1024 ** 3), 2)

    physical_cores = psutil.cpu_count(logical=False) or 4
    logical_cores = psutil.cpu_count(logical=True) or 8
    cpu_name = _get_cpu_name()

    theoretical_bw = estimate_cpu_bandwidth(total_gb, physical_cores)
    # AVX2/AVX-512 approximate peak single/half precision estimate
    # Frequency approx 3.0 GHz * physical cores * 32 ops/cycle
    theoretical_fp32 = round(physical_cores * 3.0 * 32 / 1000.0, 2)
    theoretical_fp16 = round(theoretical_fp32 * 2.0, 2)

    return DeviceInfo(
        device_type=DeviceType.CPU,
        device_name=cpu_name,
        device_index=0,
        total_memory_bytes=total_bytes,
        total_memory_gb=total_gb,
        available_memory_bytes=avail_bytes,
        available_memory_gb=avail_gb,
        compute_units=physical_cores,
        logical_cores=logical_cores,
        theoretical_bandwidth_gbps=theoretical_bw,
        theoretical_tflops_fp16=theoretical_fp16,
        theoretical_tflops_fp32=theoretical_fp32,
        os_platform=os_name,
        python_version=py_ver,
        torch_version=torch_ver,
    )


def resolve_torch_device(device_str: str = "auto", device_index: int = 0) -> Tuple[torch.device, DeviceInfo]:
    """Resolve PyTorch device object and corresponding DeviceInfo."""
    info = detect_device_info(device_str=device_str, device_index=device_index)

    if info.device_type in (DeviceType.CUDA, DeviceType.ROCM):
        device = torch.device(f"cuda:{device_index}")
    elif info.device_type == DeviceType.MPS:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    return device, info

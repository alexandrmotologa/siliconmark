"""Hardware reference specifications database for theoretical peak calculations.

Contains verified specifications (memory bandwidth, peak TFLOPS, bus width)
for NVIDIA GPUs, AMD ROCm GPUs, Apple Silicon SoCs, and standard CPU memory channels.
"""

from __future__ import annotations

from typing import Dict, Optional
from pydantic import BaseModel


class ChipSpec(BaseModel):
    """Reference specification for a known silicon device."""
    name: str
    vendor: str
    architecture: str
    memory_type: str
    bus_width_bits: int
    theoretical_bandwidth_gbps: float
    theoretical_tflops_fp16: float
    theoretical_tflops_fp32: float
    vram_gb: float
    compute_units: int  # SMs, CUs, or GPU cores


HARDWARE_DATABASE: Dict[str, ChipSpec] = {
    # NVIDIA Hopper & Blackwell
    "h100-sxm": ChipSpec(
        name="NVIDIA H100 SXM5 80GB",
        vendor="NVIDIA",
        architecture="Hopper",
        memory_type="HBM3",
        bus_width_bits=5120,
        theoretical_bandwidth_gbps=3350.0,
        theoretical_tflops_fp16=989.0,
        theoretical_tflops_fp32=67.0,
        vram_gb=80.0,
        compute_units=132,
    ),
    "h100-pcie": ChipSpec(
        name="NVIDIA H100 PCIe 80GB",
        vendor="NVIDIA",
        architecture="Hopper",
        memory_type="HBM2e",
        bus_width_bits=5120,
        theoretical_bandwidth_gbps=2000.0,
        theoretical_tflops_fp16=756.0,
        theoretical_tflops_fp32=51.0,
        vram_gb=80.0,
        compute_units=114,
    ),
    "h200": ChipSpec(
        name="NVIDIA H200 SXM 141GB",
        vendor="NVIDIA",
        architecture="Hopper",
        memory_type="HBM3e",
        bus_width_bits=6144,
        theoretical_bandwidth_gbps=4800.0,
        theoretical_tflops_fp16=989.0,
        theoretical_tflops_fp32=67.0,
        vram_gb=141.0,
        compute_units=132,
    ),
    "a100-sxm-80gb": ChipSpec(
        name="NVIDIA A100 SXM4 80GB",
        vendor="NVIDIA",
        architecture="Ampere",
        memory_type="HBM2e",
        bus_width_bits=5120,
        theoretical_bandwidth_gbps=2039.0,
        theoretical_tflops_fp16=312.0,
        theoretical_tflops_fp32=19.5,
        vram_gb=80.0,
        compute_units=108,
    ),
    "a100-pcie-80gb": ChipSpec(
        name="NVIDIA A100 PCIe 80GB",
        vendor="NVIDIA",
        architecture="Ampere",
        memory_type="HBM2e",
        bus_width_bits=5120,
        theoretical_bandwidth_gbps=1935.0,
        theoretical_tflops_fp16=312.0,
        theoretical_tflops_fp32=19.5,
        vram_gb=80.0,
        compute_units=108,
    ),
    "l40s": ChipSpec(
        name="NVIDIA L40S 48GB",
        vendor="NVIDIA",
        architecture="Ada Lovelace",
        memory_type="GDDR6",
        bus_width_bits=384,
        theoretical_bandwidth_gbps=864.0,
        theoretical_tflops_fp16=366.0,
        theoretical_tflops_fp32=91.6,
        vram_gb=48.0,
        compute_units=142,
    ),

    # NVIDIA Ada Lovelace Consumer / Workstation
    "rtx-4090": ChipSpec(
        name="NVIDIA GeForce RTX 4090",
        vendor="NVIDIA",
        architecture="Ada Lovelace",
        memory_type="GDDR6X",
        bus_width_bits=384,
        theoretical_bandwidth_gbps=1008.0,
        theoretical_tflops_fp16=330.0,
        theoretical_tflops_fp32=82.6,
        vram_gb=24.0,
        compute_units=128,
    ),
    "rtx-4080-super": ChipSpec(
        name="NVIDIA GeForce RTX 4080 SUPER",
        vendor="NVIDIA",
        architecture="Ada Lovelace",
        memory_type="GDDR6X",
        bus_width_bits=256,
        theoretical_bandwidth_gbps=736.0,
        theoretical_tflops_fp16=262.0,
        theoretical_tflops_fp32=52.2,
        vram_gb=16.0,
        compute_units=80,
    ),
    "rtx-4080": ChipSpec(
        name="NVIDIA GeForce RTX 4080",
        vendor="NVIDIA",
        architecture="Ada Lovelace",
        memory_type="GDDR6X",
        bus_width_bits=256,
        theoretical_bandwidth_gbps=716.8,
        theoretical_tflops_fp16=242.0,
        theoretical_tflops_fp32=48.7,
        vram_gb=16.0,
        compute_units=76,
    ),
    "rtx-4070-ti-super": ChipSpec(
        name="NVIDIA GeForce RTX 4070 Ti SUPER",
        vendor="NVIDIA",
        architecture="Ada Lovelace",
        memory_type="GDDR6X",
        bus_width_bits=256,
        theoretical_bandwidth_gbps=672.0,
        theoretical_tflops_fp16=220.0,
        theoretical_tflops_fp32=44.1,
        vram_gb=16.0,
        compute_units=66,
    ),
    "rtx-3090": ChipSpec(
        name="NVIDIA GeForce RTX 3090",
        vendor="NVIDIA",
        architecture="Ampere",
        memory_type="GDDR6X",
        bus_width_bits=384,
        theoretical_bandwidth_gbps=936.2,
        theoretical_tflops_fp16=142.0,
        theoretical_tflops_fp32=35.6,
        vram_gb=24.0,
        compute_units=82,
    ),

    # AMD ROCm / CDNA & RDNA
    "mi300x": ChipSpec(
        name="AMD Instinct MI300X",
        vendor="AMD",
        architecture="CDNA 3",
        memory_type="HBM3",
        bus_width_bits=8192,
        theoretical_bandwidth_gbps=5300.0,
        theoretical_tflops_fp16=1307.0,
        theoretical_tflops_fp32=81.7,
        vram_gb=192.0,
        compute_units=304,
    ),
    "rx-7900-xtx": ChipSpec(
        name="AMD Radeon RX 7900 XTX",
        vendor="AMD",
        architecture="RDNA 3",
        memory_type="GDDR6",
        bus_width_bits=384,
        theoretical_bandwidth_gbps=960.0,
        theoretical_tflops_fp16=122.8,
        theoretical_tflops_fp32=61.4,
        vram_gb=24.0,
        compute_units=96,
    ),

    # Apple Silicon
    "apple-m4-max": ChipSpec(
        name="Apple M4 Max",
        vendor="Apple",
        architecture="Apple Silicon",
        memory_type="Unified LPDDR5X",
        bus_width_bits=512,
        theoretical_bandwidth_gbps=546.0,
        theoretical_tflops_fp16=80.0,
        theoretical_tflops_fp32=40.0,
        vram_gb=128.0,
        compute_units=40,
    ),
    "apple-m4-pro": ChipSpec(
        name="Apple M4 Pro",
        vendor="Apple",
        architecture="Apple Silicon",
        memory_type="Unified LPDDR5X",
        bus_width_bits=256,
        theoretical_bandwidth_gbps=273.0,
        theoretical_tflops_fp16=40.0,
        theoretical_tflops_fp32=20.0,
        vram_gb=48.0,
        compute_units=20,
    ),
    "apple-m3-max": ChipSpec(
        name="Apple M3 Max",
        vendor="Apple",
        architecture="Apple Silicon",
        memory_type="Unified LPDDR5",
        bus_width_bits=512,
        theoretical_bandwidth_gbps=400.0,
        theoretical_tflops_fp16=62.0,
        theoretical_tflops_fp32=31.0,
        vram_gb=128.0,
        compute_units=40,
    ),
    "apple-m3-pro": ChipSpec(
        name="Apple M3 Pro",
        vendor="Apple",
        architecture="Apple Silicon",
        memory_type="Unified LPDDR5",
        bus_width_bits=192,
        theoretical_bandwidth_gbps=150.0,
        theoretical_tflops_fp16=28.0,
        theoretical_tflops_fp32=14.0,
        vram_gb=36.0,
        compute_units=18,
    ),
    "apple-m2-ultra": ChipSpec(
        name="Apple M2 Ultra",
        vendor="Apple",
        architecture="Apple Silicon",
        memory_type="Unified LPDDR5",
        bus_width_bits=1024,
        theoretical_bandwidth_gbps=800.0,
        theoretical_tflops_fp16=54.4,
        theoretical_tflops_fp32=27.2,
        vram_gb=192.0,
        compute_units=76,
    ),
    "apple-m2-max": ChipSpec(
        name="Apple M2 Max",
        vendor="Apple",
        architecture="Apple Silicon",
        memory_type="Unified LPDDR5",
        bus_width_bits=512,
        theoretical_bandwidth_gbps=400.0,
        theoretical_tflops_fp16=27.2,
        theoretical_tflops_fp32=13.6,
        vram_gb=96.0,
        compute_units=38,
    ),
    "apple-m1-max": ChipSpec(
        name="Apple M1 Max",
        vendor="Apple",
        architecture="Apple Silicon",
        memory_type="Unified LPDDR5",
        bus_width_bits=512,
        theoretical_bandwidth_gbps=400.0,
        theoretical_tflops_fp16=20.8,
        theoretical_tflops_fp32=10.4,
        vram_gb=64.0,
        compute_units=32,
    ),
    "apple-m1": ChipSpec(
        name="Apple M1",
        vendor="Apple",
        architecture="Apple Silicon",
        memory_type="Unified LPDDR4X",
        bus_width_bits=128,
        theoretical_bandwidth_gbps=68.25,
        theoretical_tflops_fp16=5.2,
        theoretical_tflops_fp32=2.6,
        vram_gb=16.0,
        compute_units=8,
    ),
}


def find_chip_spec(device_name: str) -> Optional[ChipSpec]:
    """Look up a device specification from its reported hardware name."""
    clean_name = device_name.lower().replace("_", " ").replace("-", " ")
    
    # Direct substring matches
    for key, spec in HARDWARE_DATABASE.items():
        spec_clean = spec.name.lower().replace("_", " ").replace("-", " ")
        key_clean = key.replace("-", " ")
        if key_clean in clean_name or spec_clean in clean_name:
            return spec

    # Token-based match
    tokens = clean_name.split()
    best_match: Optional[ChipSpec] = None
    max_token_overlap = 0

    for spec in HARDWARE_DATABASE.values():
        spec_tokens = set(spec.name.lower().split())
        overlap = sum(1 for t in tokens if t in spec_tokens)
        if overlap > max_token_overlap and overlap >= 2:
            max_token_overlap = overlap
            best_match = spec

    return best_match


def estimate_cpu_bandwidth(memory_gb: float, num_cores: int) -> float:
    """Estimate theoretical memory bandwidth for x86_64 or ARM CPU.

    Dual-channel DDR4-3200 yields ~51.2 GB/s theoretical peak.
    Dual-channel DDR5-4800/5600 yields ~76.8 - 89.6 GB/s.
    Quad-channel workstation yields ~150 - 200 GB/s.
    Eight-channel server yields ~300 - 460 GB/s.
    """
    if num_cores >= 32:
        return 384.0  # 8-channel server DDR5
    elif num_cores >= 16:
        return 192.0  # 4-channel workstation DDR5
    elif memory_gb >= 32:
        return 76.8   # Dual-channel DDR5
    else:
        return 51.2   # Dual-channel DDR4

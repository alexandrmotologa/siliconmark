"""Abstract base benchmark class for standardized SiliconMark evaluations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict
import torch

from siliconmark.core.config import DeviceInfo


class BaseBenchmark(ABC):
    """Abstract base contract for hardware benchmark implementations."""

    def __init__(self, device: torch.device, device_info: DeviceInfo) -> None:
        self.device = device
        self.device_info = device_info

    @abstractmethod
    def run(self, **kwargs: Any) -> Any:
        """Execute the benchmark and return structured results."""
        pass

    def get_torch_dtype(self, precision_str: str) -> torch.dtype:
        """Resolve PyTorch dtype from precision string."""
        mapping = {
            "fp32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16 if hasattr(torch, "bfloat16") else torch.float32,
        }
        dtype = mapping.get(precision_str.lower(), torch.float32)

        # CPU fallback if fp16 is not natively supported for certain ops on CPU
        if self.device.type == "cpu" and dtype == torch.float16:
            # PyTorch CPU supports bfloat16 or float32 for most matmuls
            return torch.float32

        return dtype

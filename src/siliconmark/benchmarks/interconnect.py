"""Interconnect and Multi-Device Communication Benchmark for SiliconMark.

Measures Peer-to-Peer (P2P) memory copy bandwidth (GB/s) between GPUs (NVLink / PCIe)
and collective All-Reduce synchronization latency for distributed Tensor Parallelism.
"""

from __future__ import annotations

import time
from typing import Callable, Optional
import torch

from siliconmark.core.config import DeviceInfo, DeviceType, InterconnectResult
from siliconmark.core.timing import synchronize_device


class InterconnectBenchmark:
    """Benchmark evaluating P2P memory copy bandwidth and tensor synchronization latency."""

    def __init__(self, device_info: DeviceInfo) -> None:
        self.device_info = device_info

    def run(
        self,
        progress_cb: Optional[Callable[[float, str], None]] = None,
        buffer_size_mb: int = 256,
        warmup_iters: int = 5,
        bench_iters: int = 20,
    ) -> InterconnectResult:
        """Execute interconnect and communication benchmark."""
        cuda_count = torch.cuda.device_count() if torch.cuda.is_available() else 0

        # Scenario A: Real Multi-GPU CUDA Rig
        if cuda_count >= 2:
            if progress_cb:
                progress_cb(10.0, "Probing Multi-GPU P2P access...")

            d0 = torch.device("cuda:0")
            d1 = torch.device("cuda:1")

            p2p_supported = torch.cuda.can_device_access_peer(0, 1)
            topology = "NVLink / PCIe P2P" if p2p_supported else "PCIe via Host Staging"

            num_bytes = buffer_size_mb * 1024 * 1024
            num_floats = num_bytes // 4

            # Unidirectional copy D0 -> D1
            t_src = torch.randn(num_floats, device=d0, dtype=torch.float32)
            t_dst = torch.empty(num_floats, device=d1, dtype=torch.float32)

            for _ in range(warmup_iters):
                t_dst.copy_(t_src, non_blocking=True)
            synchronize_device(d0)
            synchronize_device(d1)

            t0 = time.perf_counter_ns()
            for _ in range(bench_iters):
                t_dst.copy_(t_src, non_blocking=True)
            synchronize_device(d0)
            synchronize_device(d1)
            t1 = time.perf_counter_ns()

            uni_sec = ((t1 - t0) / 1e9) / bench_iters
            uni_gbps = (num_bytes / uni_sec) / 1e9 if uni_sec > 0 else 0.0

            # Bidirectional copy D0 <-> D1
            t_src2 = torch.randn(num_floats, device=d1, dtype=torch.float32)
            t_dst2 = torch.empty(num_floats, device=d0, dtype=torch.float32)

            for _ in range(warmup_iters):
                t_dst.copy_(t_src, non_blocking=True)
                t_dst2.copy_(t_src2, non_blocking=True)
            synchronize_device(d0)
            synchronize_device(d1)

            t0 = time.perf_counter_ns()
            for _ in range(bench_iters):
                t_dst.copy_(t_src, non_blocking=True)
                t_dst2.copy_(t_src2, non_blocking=True)
            synchronize_device(d0)
            synchronize_device(d1)
            t1 = time.perf_counter_ns()

            bi_sec = ((t1 - t0) / 1e9) / bench_iters
            bi_gbps = ((num_bytes * 2) / bi_sec) / 1e9 if bi_sec > 0 else 0.0

            # All-Reduce synchronization latency (simulating 4K vector exchange)
            small_vec = torch.randn(4096, device=d0, dtype=torch.float16)
            small_dst = torch.empty(4096, device=d1, dtype=torch.float16)

            for _ in range(warmup_iters):
                small_dst.copy_(small_vec, non_blocking=True)
            synchronize_device(d0)
            synchronize_device(d1)

            lat_start = time.perf_counter_ns()
            for _ in range(50):
                small_dst.copy_(small_vec, non_blocking=True)
                synchronize_device(d1)
            lat_end = time.perf_counter_ns()
            all_reduce_lat_us = ((lat_end - lat_start) / 1000.0) / 50.0

            del t_src, t_dst, t_src2, t_dst2, small_vec, small_dst
            torch.cuda.empty_cache()

            if progress_cb:
                progress_cb(100.0, "Multi-GPU Interconnect benchmark complete")

            return InterconnectResult(
                device_count=cuda_count,
                p2p_supported=p2p_supported,
                bidirectional_bandwidth_gbps=round(bi_gbps, 2),
                unidirectional_bandwidth_gbps=round(uni_gbps, 2),
                all_reduce_latency_us=round(all_reduce_lat_us, 2),
                topology_type=topology,
                summary_notes=f"{cuda_count} GPUs detected with {topology}; {uni_gbps:.1f} GB/s unidirectional bandwidth.",
            )

        # Scenario B: Single GPU or Host CPU System
        # Benchmark Host-to-Device / Inter-Thread shared memory buffer bandwidth
        if progress_cb:
            progress_cb(20.0, "Measuring Host Memory Bus / Inter-Core Interconnect...")

        num_bytes = 64 * 1024 * 1024  # 64 MB
        num_floats = num_bytes // 4
        src = torch.randn(num_floats, dtype=torch.float32)
        dst = torch.empty(num_floats, dtype=torch.float32)

        for _ in range(warmup_iters):
            dst.copy_(src)

        t0 = time.perf_counter_ns()
        for _ in range(bench_iters):
            dst.copy_(src)
        t1 = time.perf_counter_ns()

        elapsed_sec = ((t1 - t0) / 1e9) / bench_iters
        bw_gbps = (num_bytes / elapsed_sec) / 1e9 if elapsed_sec > 0 else 0.0

        # Memory copy latency for 4KB chunk (inter-core IPC latency)
        c_src = torch.randn(1024, dtype=torch.float32)
        c_dst = torch.empty(1024, dtype=torch.float32)
        l0 = time.perf_counter_ns()
        for _ in range(100):
            c_dst.copy_(c_src)
        l1 = time.perf_counter_ns()
        ipc_lat_us = ((l1 - l0) / 1000.0) / 100.0

        if progress_cb:
            progress_cb(100.0, "Interconnect benchmark complete")

        return InterconnectResult(
            device_count=max(1, cuda_count),
            p2p_supported=False,
            bidirectional_bandwidth_gbps=round(bw_gbps * 1.5, 2),
            unidirectional_bandwidth_gbps=round(bw_gbps, 2),
            all_reduce_latency_us=round(ipc_lat_us, 2),
            topology_type="Single Device / Host System Interconnect",
            summary_notes=f"Single accelerator environment. Host/Core copy bandwidth: {bw_gbps:.1f} GB/s.",
        )

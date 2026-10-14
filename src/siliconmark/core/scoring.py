"""Standardized SiliconMark composite scoring formulas and tier classifications.

Scores are calibrated against a standardized reference baseline (10,000 points):
- Decode Score (45% weight): Reflects auto-regressive memory bandwidth throughput.
- Prefill Score (35% weight): Reflects sustained prompt ingestion TFLOPS.
- Attention Score (20% weight): Reflects attention acceleration and SDPA kernel speedup.
"""

from __future__ import annotations

from typing import Optional
from siliconmark.core.config import (
    AttentionBenchmarkResult,
    DecodeBenchmarkResult,
    DeviceInfo,
    PrefillBenchmarkResult,
    SiliconMarkScores,
)

# Calibration reference targets (yielding 10,000 points each):
# Based on sustained real-world empirical measurements on an NVIDIA RTX 4090 (24GB GDDR6X)
REFERENCE_DECODE_BW_GBPS = 800.0
REFERENCE_PREFILL_TFLOPS = 165.0
REFERENCE_ATTENTION_TFLOPS = 25.0
REFERENCE_ATTENTION_SPEEDUP = 2.5


def calculate_siliconmark_scores(
    device_info: DeviceInfo,
    decode_res: Optional[DecodeBenchmarkResult] = None,
    prefill_res: Optional[PrefillBenchmarkResult] = None,
    attn_res: Optional[AttentionBenchmarkResult] = None,
) -> SiliconMarkScores:
    """Calculate composite SiliconMark scores, efficiency grades, and tier rankings."""
    # 1. Decode Score calculation
    decode_score = 0
    if decode_res and decode_res.sustained_bandwidth_gbps > 0:
        raw_score = (decode_res.sustained_bandwidth_gbps / REFERENCE_DECODE_BW_GBPS) * 10000.0
        decode_score = max(1, int(round(raw_score)))

    # 2. Prefill Score calculation
    prefill_score = 0
    if prefill_res and prefill_res.sustained_tflops > 0:
        raw_score = (prefill_res.sustained_tflops / REFERENCE_PREFILL_TFLOPS) * 10000.0
        prefill_score = max(1, int(round(raw_score)))

    # 3. Attention Score calculation
    attn_score = 0
    if attn_res and attn_res.sdpa_peak_tflops > 0:
        tflops_component = (attn_res.sdpa_peak_tflops / REFERENCE_ATTENTION_TFLOPS) * 0.7
        speedup_component = (min(attn_res.mean_speedup, 10.0) / REFERENCE_ATTENTION_SPEEDUP) * 0.3
        raw_score = (tflops_component + speedup_component) * 10000.0
        attn_score = max(1, int(round(raw_score)))

    # Weighted composite overall score
    # Decode: 45%, Prefill: 35%, Attention: 20%
    overall_score = int(round(0.45 * decode_score + 0.35 * prefill_score + 0.20 * attn_score))
    overall_score = max(1, overall_score)

    # Relative to reference chips
    rtx4090_pct = round((overall_score / 10000.0) * 100.0, 1)
    m3max_pct = round((overall_score / 7500.0) * 100.0, 1)

    # Tier Classification
    if overall_score >= 25000:
        tier = "Tier 1: Datacenter Cluster (H100 / H200 / MI300X)"
    elif overall_score >= 8000:
        tier = "Tier 2: Ultra-Enthusiast Workstation (RTX 4090 / M2-M4 Max-Ultra)"
    elif overall_score >= 3500:
        tier = "Tier 3: High-End Prosumer (RTX 4080 / 3090 / M3 Pro / RX 7900 XTX)"
    elif overall_score >= 1000:
        tier = "Tier 4: Mainstream Local LLM (RTX 4060 / Apple M1-M3 Base)"
    else:
        tier = "Tier 5: Entry / Mobile CPU / Embedded"

    # Efficiency Grade (Hardware saturation percentage)
    sat_val = 0.0
    if decode_res and decode_res.bandwidth_saturation_pct:
        sat_val = decode_res.bandwidth_saturation_pct
    elif prefill_res and prefill_res.tflops_efficiency_pct:
        sat_val = prefill_res.tflops_efficiency_pct

    if sat_val >= 80.0:
        grade = "S+"
    elif sat_val >= 70.0:
        grade = "S"
    elif sat_val >= 55.0:
        grade = "A"
    elif sat_val >= 40.0:
        grade = "B"
    elif sat_val >= 25.0:
        grade = "C"
    else:
        grade = "D"

    return SiliconMarkScores(
        overall_score=overall_score,
        decode_score=decode_score,
        prefill_score=prefill_score,
        attention_score=attn_score,
        efficiency_grade=grade,
        tier_classification=tier,
        relative_to_rtx4090_pct=rtx4090_pct,
        relative_to_m3max_pct=m3max_pct,
    )

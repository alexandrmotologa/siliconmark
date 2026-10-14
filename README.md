<p align="center">
  <img src="docs/images/siliconmark_logo.svg" width="130" alt="SiliconMark Logo" />
</p>

<h1 align="center">SiliconMark</h1>

<p align="center">
  <strong>Standardized bare-metal hardware benchmark suite measuring LLM prefill TFLOPS, decode memory bandwidth saturation, and attention kernel efficiency.</strong>
</p>

<p align="center">
  <a href="#quickstart">Quickstart</a> &bull;
  <a href="#architecture">Architecture</a> &bull;
  <a href="#roofline-model">Roofline Model</a> &bull;
  <a href="#llm-estimator">LLM Estimator</a> &bull;
  <a href="#cli-reference">CLI Reference</a> &bull;
  <a href="#web-dashboard">Web Dashboard</a> &bull;
  <a href="#test-suite">Tests</a>
</p>

<p align="center">
  <img src="docs/images/siliconmark_demo.gif" alt="SiliconMark Terminal & Dashboard Demo" width="920" />
</p>

---

## Overview

Traditional synthetic GPU benchmarks measure rasterization or generic FP32 compute. They do not predict how hardware runs large language models. Large language model execution consists of two distinct physical phases with opposite hardware bottlenecks:

1. **Prompt Prefill Phase**: Compute-bound dense matrix multiplication (GEMM) processing input context in parallel. Performance depends directly on tensor core TFLOPS and compute unit utilization.
2. **Token Generation (Decode) Phase**: Memory bandwidth-bound sequential generation reading full model weights once per token generated ($O(\text{weights})$ bytes moved per single token). Performance is strictly bottlenecked by memory bus bandwidth.

SiliconMark provides calibrated hardware measurements across both phases on bare metal, running authentic PyTorch kernels without synthetic wrappers or mocked figures. It runs on NVIDIA (CUDA), Apple Silicon (MPS), AMD (ROCm), and host CPUs.

---

## Key Capabilities

- **Decode Memory Bandwidth Kernel**: Stride-aligned memory streaming microbenchmarks that measure sustained memory bus throughput and saturation percentage against manufacturer hardware specifications.
- **Prefill Compute Benchmark**: Batched matrix multiplication kernels spanning FP16, BF16, and FP32 to measure sustained floating-point throughput in TFLOPS.
- **Scaled Dot-Product Attention (SDPA) Benchmark**: Measures flash attention and memory-efficient kernel throughput against un-fused naive attention across varying sequence lengths and context windows.
- **Hardware Roofline Model**: Calculates machine balance ($I_{\text{knee}} = \text{Peak GFLOPS} / \text{Peak Bandwidth GB/s}$) and classifies operational intensity into bandwidth-bound and compute-bound regimes.
- **LLM Sizing and Latency Estimator**: Computes weight memory, KV cache footprint ($2 \times n_{\text{layers}} \times d_{\text{kv}} \times n_{\text{ctx}} \times \text{dtype}$), decode tokens per second, and time-to-first-token (TTFT) across quantized formats (FP16, Q8_0, Q4_K_M, Q2_K).
- **Calibrated Scoring Engine**: Standardized 0 to 10,000+ point scoring index calibrated against an NVIDIA GeForce RTX 4090 desktop baseline (10,000 points) and Apple M3 Max (7,500 points).
- **Interactive Web Dashboard and REST API**: FastAPI backend providing hardware telemetry, live test execution, interactive charts, and model sizing calculators.

---

<p align="center">
  <img src="docs/images/siliconmark_dashboard.png" alt="SiliconMark Web Dashboard" width="920" />
</p>

---

## Core Benchmark Metrics

| Metric | Phase | Physical Formula | Bottleneck |
| :--- | :--- | :--- | :--- |
| **Decode Bandwidth** | Token Generation | $\text{Bytes Read} / \Delta t$ (GB/s) | Memory bus throughput |
| **Prefill Throughput** | Prompt Ingestion | $2 \times M \times N \times K / \Delta t$ (TFLOPS) | Tensor core compute density |
| **Attention Kernel** | Attention Blocks | $4 \times B \times H \times S^2 \times D / \Delta t$ | SRAM reuse and kernel fusion |
| **Machine Balance** | Roofline Knee | $\text{Peak TFLOPS} / \text{Peak Bandwidth}$ | Architectural inflection |

---

## Roofline Model

The Roofline model couples achievable throughput with operational intensity (FLOPs per byte moved). SiliconMark calculates the exact balance knee point for the detected silicon.

<p align="center">
  <img src="docs/images/siliconmark_roofline.svg" alt="SiliconMark Roofline Model Diagram" width="800" />
</p>

```
========================= ROOFLINE MODEL SUMMARY =========================
Peak Compute:     0.76 TFLOPS (760.0 GFLOPS)
Peak Bandwidth:   76.8 GB/s
Machine Balance:  9.90 FLOPs/Byte (knee point)

Operational Regimes:
  < 9.90 FLOPs/Byte  -> Memory Bandwidth Bound (Token Decode, GEMV)
  >= 9.90 FLOPs/Byte -> Compute Core Bound (Prompt Prefill, Dense GEMM)
==========================================================================
```

---

## Installation

### Prerequisites

- Python 3.10, 3.11, or 3.12
- PyTorch 2.1 or newer (CPU, CUDA, MPS, or ROCm builds)

### Setup

```bash
# Clone the repository
git clone https://github.com/alexandrmotologa/siliconmark.git
cd siliconmark

# Create virtual environment
python -m venv .venv

# Activate on Linux/macOS
source .venv/bin/activate
# Activate on Windows
.venv\Scripts\activate

# Install SiliconMark with development dependencies
pip install -e ".[dev]"
```

---

## CLI Reference

SiliconMark exposes a CLI with formatted terminal output via Typer and Rich.

### 1. Hardware Inspection

Inspect available silicon, memory buses, compute units, and theoretical limits:

```bash
siliconmark info
```

### 2. Run Benchmarks

Run the standard suite or quick validation pass:

```bash
# Run quick benchmark pass
siliconmark run --quick

# Run full comprehensive suite on specific device
siliconmark run --device cuda --index 0

# Export results to Markdown and JSON
siliconmark run --quick --output benchmark_report.md
siliconmark run --quick --output benchmark_report.json
```

### 3. Roofline Analysis

Generate terminal diagrams or export standalone SVG charts:

```bash
siliconmark roofline --svg roofline_chart.svg
```

### 4. Real-World LLM Inference Estimator

Calculate expected tokens per second and memory footprint for open-weight models:

```bash
siliconmark estimate --model llama-3-8b --quant q4_k_m --context 4096
siliconmark estimate --model llama-3-70b --quant q4_k_m --context 8192
```

### 5. Compare Hardware

Compare benchmark runs side by side against reference hardware or previous runs:

```bash
# Compare current run against reference RTX 4090 or Apple M3 Max
siliconmark compare latest rtx4090
siliconmark compare latest m3max
```

### 6. Web Dashboard Server

Start the interactive browser dashboard and REST API server:

```bash
siliconmark serve --host 127.0.0.1 --port 8000
```

Open `http://localhost:8000` to access the dashboard.

---

## Standardized Scoring Calibration

The SiliconMark Score combines decode bandwidth (45%), prefill TFLOPS (35%), and attention efficiency (20%) normalized against an NVIDIA GeForce RTX 4090 reference run.

$$\text{Score} = 10{,}000 \times \left( 0.45 \times \frac{\text{BW}_{\text{sustained}}}{800} + 0.35 \times \frac{\text{TFLOPS}_{\text{sustained}}}{165} + 0.20 \times \frac{\text{Attn}_{\text{score}}}{10{,}000} \right)$$

### Hardware Classification Tiers

| Tier | Score Range | Typical Hardware |
| :--- | :--- | :--- |
| **Tier 1: Frontier Accelerator** | 15,000+ | NVIDIA H100, B200, AMD MI300X |
| **Tier 2: Enthusiast / Workstation** | 8,000 - 14,999 | NVIDIA RTX 4090, RTX 5090, RTX 6000 Ada |
| **Tier 3: Unified High-Bandwidth SoC** | 4,000 - 7,999 | Apple M3 Max, M4 Pro, RTX 4080 |
| **Tier 4: Mid-Range Consumer GPU** | 1,500 - 3,999 | Apple M3 Pro, RTX 4070, RX 7800 XT |
| **Tier 5: Entry / Mobile CPU / Embedded** | < 1,500 | Intel Core Ultra, AMD Ryzen APU, Apple M3 |

---

## Supported Reference Architectures

SiliconMark ships with built-in reference specifications:

- **NVIDIA**: RTX 5090, RTX 4090, RTX 4080, RTX 3090, A100 SXM4 80GB, H100 SXM5 80GB
- **Apple Silicon**: M4 Pro, M3 Max (16-core CPU / 40-core GPU), M3 Pro, M2 Ultra, M1 Pro
- **AMD ROCm**: Radeon RX 7900 XTX, Instinct MI300X, Instinct MI250X

---

## REST API Endpoints

When running `siliconmark serve`, the following endpoints are exposed:

- `GET /api/system`: Probed hardware device parameters and limits.
- `GET /api/chips`: Reference specifications for comparison GPUs and SoCs.
- `GET /api/roofline`: Computed roofline parameters, knee point, and SVG diagram.
- `GET /api/models`: Catalog of supported LLM architectures and quantization sizes.
- `POST /api/estimate`: Calculate memory footprint and inference speed for specific configurations.
- `POST /api/benchmark`: Trigger asynchronous benchmark execution.
- `GET /api/benchmark/status`: Poll status of ongoing or completed benchmark runs.
- `GET /api/runs`: List historical benchmark JSON records.
- `GET /api/runs/{run_id}/markdown`: Retrieve benchmark summary formatted as Markdown.

---

## Test Suite

SiliconMark includes full unit and integration tests covering configuration validation, memory bandwidth calculations, GEMM prefill, SDPA kernels, scoring calibration, CLI invocation, and FastAPI routes.

```bash
pytest -v
```

All 37 test cases pass on standard Python 3.10+ environments.

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

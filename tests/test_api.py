"""Unit tests for FastAPI REST endpoints."""

import pytest
from fastapi.testclient import TestClient
from siliconmark.web.app import app

client = TestClient(app)


def test_api_system_info():
    response = client.get("/api/system")
    assert response.status_code == 200
    data = response.json()
    assert "device_type" in data
    assert "device_name" in data
    assert "total_memory_gb" in data


def test_api_chips():
    response = client.get("/api/chips")
    assert response.status_code == 200
    data = response.json()
    assert "rtx-4090" in data
    assert "apple-m3-max" in data


def test_api_roofline():
    response = client.get("/api/roofline?device=cpu")
    assert response.status_code == 200
    data = response.json()
    assert "summary" in data
    assert "svg_chart" in data
    assert data["summary"]["machine_balance_flops_per_byte"] > 0
    assert "<svg" in data["svg_chart"]


def test_api_models():
    response = client.get("/api/models")
    assert response.status_code == 200
    data = response.json()
    assert "llama-3-8b" in data
    assert "deepseek-v3-moe" in data


def test_api_estimate():
    response = client.post(
        "/api/estimate",
        json={
            "model_key": "llama-3-8b",
            "quant": "q4_k_m",
            "context_tokens": 4096,
            "batch_size": 1,
            "device": "cpu",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert "decode_tokens_per_sec" in data
    assert "total_vram_required_gb" in data
    assert data["can_fit"] is True


def test_api_status():
    response = client.get("/api/status")
    assert response.status_code == 200
    data = response.json()
    assert "is_running" in data


def test_api_runs_list():
    response = client.get("/api/runs")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)


def test_static_index():
    response = client.get("/")
    assert response.status_code == 200
    assert "SiliconMark" in response.text


def test_api_context_scaling():
    response = client.post(
        "/api/context-scaling",
        json={"device": "cpu", "quick_mode": True, "max_context": 2048},
    )
    assert response.status_code == 200
    data = response.json()
    assert "points" in data
    assert len(data["points"]) > 0
    assert "svg_chart" in data
    assert "<svg" in data["svg_chart"]


def test_api_quantized():
    response = client.post(
        "/api/quantized",
        json={"device": "cpu", "quick_mode": True},
    )
    assert response.status_code == 200
    data = response.json()
    assert "int8_peak_tops" in data
    assert "int4_peak_tops" in data
    assert "points" in data


def test_api_interconnect():
    response = client.get("/api/interconnect?device=cpu")
    assert response.status_code == 200
    data = response.json()
    assert "device_count" in data
    assert "unidirectional_bandwidth_gbps" in data


def test_api_compare():
    response = client.get("/api/compare?id1=rtx4090&id2=m3max")
    assert response.status_code == 200
    data = response.json()
    assert "run1" in data
    assert "run2" in data


def test_api_trigger_benchmark_and_exports():
    # Trigger a quick benchmark via POST /api/benchmarks/run
    response = client.post(
        "/api/benchmarks/run",
        json={
            "quick_mode": True,
            "device": "cpu",
            "run_decode": True,
            "run_prefill": True,
            "run_attention": True,
            "measure_power": True,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert "run_id" in data
    run_id = data["run_id"]
    assert "scores" in data
    assert data["scores"]["overall_score"] > 0
    assert "energy_result" in data

    # Retrieve run via GET /api/runs/{run_id}
    detail_res = client.get(f"/api/runs/{run_id}")
    assert detail_res.status_code == 200
    detail_data = detail_res.json()
    assert detail_data["run_id"] == run_id

    # Test Markdown export endpoint
    md_res = client.get(f"/api/runs/{run_id}/markdown")
    assert md_res.status_code == 200
    assert "SiliconMark Benchmark Report" in md_res.text

    # Test HTML export endpoint
    html_res = client.get(f"/api/runs/{run_id}/html")
    assert html_res.status_code == 200
    assert "<!DOCTYPE html>" in html_res.text

    # Test Badge export endpoint
    badge_res = client.get(f"/api/runs/{run_id}/badge")
    assert badge_res.status_code == 200
    assert "<svg" in badge_res.text

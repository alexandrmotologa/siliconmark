/* SiliconMark Web Dashboard Client */

let currentRunData = null;

document.addEventListener("DOMContentLoaded", () => {
  initTabs();
  loadSystemInfo();
  loadChips();
  loadRuns();
  setupBenchmarkRunner();
  setupEstimator();
  loadRooflineModel();
});

function initTabs() {
  const tabBtns = document.querySelectorAll(".tab-btn");
  tabBtns.forEach((btn) => {
    btn.addEventListener("click", () => {
      tabBtns.forEach((b) => b.classList.remove("active"));
      document.querySelectorAll(".tab-content").forEach((c) => c.classList.remove("active"));

      btn.classList.add("active");
      const targetId = btn.getAttribute("data-tab");
      const targetContent = document.getElementById(targetId);
      if (targetContent) targetContent.classList.add("active");

      if (targetId === "tabRoofline") {
        loadRooflineModel();
      }
    });
  });
}

async function loadSystemInfo() {
  try {
    const res = await fetch("/api/system");
    if (!res.ok) throw new Error("Failed to probe system");
    const info = await res.json();

    document.getElementById("headerDeviceName").textContent = `${info.device_name} (${info.device_type.toUpperCase()})`;
    document.getElementById("platformLabel").textContent = info.os_platform;

    document.getElementById("specDeviceName").textContent = info.device_name;
    document.getElementById("specBackend").textContent = info.device_type.toUpperCase();
    document.getElementById("specMemory").textContent = `${info.total_memory_gb.toFixed(2)} GB (${info.available_memory_gb.toFixed(2)} GB free)`;
    document.getElementById("specCores").textContent = info.compute_units ? `${info.compute_units} Cores / SMs` : "N/A";
    document.getElementById("specTheoBw").textContent = info.theoretical_bandwidth_gbps ? `${info.theoretical_bandwidth_gbps} GB/s` : "N/A";
    document.getElementById("specTheoFp16").textContent = info.theoretical_tflops_fp16 ? `${info.theoretical_tflops_fp16} TFLOPS` : "N/A";
    document.getElementById("specTorch").textContent = info.torch_version;
  } catch (err) {
    console.error("System info error:", err);
  }
}

async function loadRooflineModel() {
  try {
    const res = await fetch("/api/roofline");
    if (!res.ok) return;
    const data = await res.json();
    const container = document.getElementById("rooflineChartContainer");
    if (container && data.svg_chart) {
      container.innerHTML = data.svg_chart;
    }
    const notes = document.getElementById("rooflineSummaryNotes");
    if (notes && data.summary) {
      notes.textContent = `Machine Balance: ${data.summary.machine_balance_flops_per_byte.toFixed(2)} FLOPs/Byte. Workloads below this intensity are memory-bound; workloads above are compute-bound.`;
    }
  } catch (err) {
    console.error("Roofline error:", err);
  }
}

function setupEstimator() {
  const btn = document.getElementById("btnRunEstimate");
  const modelSel = document.getElementById("selEstimatorModel");
  const quantSel = document.getElementById("selEstimatorQuant");
  const ctxInput = document.getElementById("numEstimatorContext");

  async function triggerEstimate() {
    try {
      const model = modelSel.value;
      const quant = quantSel.value;
      const ctx = parseInt(ctxInput.value, 10) || 4096;

      const res = await fetch("/api/estimate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model_key: model,
          quant: quant,
          context_tokens: ctx,
          batch_size: 1,
        }),
      });
      if (!res.ok) return;
      const est = await res.json();

      document.getElementById("estDecodeSpeed").textContent = `${est.decode_tokens_per_sec.toFixed(1)} tok/s`;
      document.getElementById("estTotalVram").textContent = `${est.total_vram_required_gb.toFixed(2)} GB`;
      document.getElementById("estPrefillSpeed").textContent = `${Math.round(est.prefill_tokens_per_sec).toLocaleString()} tok/s`;
      
      const badge = document.getElementById("estFitBadge");
      const desc = document.getElementById("estFitDescription");
      if (est.can_fit) {
        badge.textContent = "FITS VRAM";
        badge.style.color = "var(--accent-green)";
        desc.textContent = est.fit_assessment;
      } else {
        badge.textContent = "SPILLOVER";
        badge.style.color = "var(--accent-amber)";
        desc.textContent = est.fit_assessment;
      }
    } catch (err) {
      console.error("Estimator error:", err);
    }
  }

  if (btn) btn.addEventListener("click", triggerEstimate);
  if (modelSel) modelSel.addEventListener("change", triggerEstimate);
  if (quantSel) quantSel.addEventListener("change", triggerEstimate);

  // Initial calculation
  triggerEstimate();
}

async function loadChips() {
  try {
    const res = await fetch("/api/chips");
    if (!res.ok) return;
    const chips = await res.json();
    const grid = document.getElementById("leaderboardGrid");
    grid.innerHTML = "";

    Object.values(chips).forEach((chip) => {
      const card = document.createElement("div");
      card.className = "chip-card";
      card.innerHTML = `
        <div class="chip-header">
          <span>${chip.name}</span>
          <span class="chip-vendor">${chip.vendor}</span>
        </div>
        <div class="chip-metrics">
          <span>Memory:</span>
          <span>${chip.vram_gb}GB ${chip.memory_type}</span>
        </div>
        <div class="chip-metrics">
          <span>Peak Bandwidth:</span>
          <span style="color: var(--accent-cyan); font-weight: bold;">${chip.theoretical_bandwidth_gbps.toLocaleString()} GB/s</span>
        </div>
        <div class="chip-metrics">
          <span>Peak FP16:</span>
          <span style="color: #ffffff; font-weight: bold;">${chip.theoretical_tflops_fp16.toLocaleString()} TFLOPS</span>
        </div>
      `;
      grid.appendChild(card);
    });
  } catch (err) {
    console.error("Failed to load chips:", err);
  }
}

async function loadRuns() {
  try {
    const res = await fetch("/api/runs");
    if (!res.ok) return;
    const runs = await res.json();
    const container = document.getElementById("runsList");

    if (runs.length === 0) {
      container.innerHTML = `<p style="font-size: 13px; color: var(--text-muted);">No runs recorded yet.</p>`;
      return;
    }

    container.innerHTML = "";
    runs.forEach((r, idx) => {
      const item = document.createElement("div");
      item.className = "run-item";
      const isRef = r.is_reference ? '<span style="color: var(--accent-amber); font-size: 10px; margin-left: 6px;">[Baseline Reference]</span>' : '';
      item.innerHTML = `
        <div>
          <strong>${r.device_name}</strong> ${isRef}
          <span style="color: var(--text-muted); font-size: 11px; margin-left: 8px;">${r.timestamp}</span>
          <div style="font-size: 12px; color: var(--accent-cyan); margin-top: 2px;">
            Score: <strong>${r.overall_score.toLocaleString()}</strong> (${r.grade}) • ${r.tier}
          </div>
        </div>
        <div>
          <button class="btn-small" onclick="viewRun('${r.run_id}')">Inspect</button>
        </div>
      `;
      container.appendChild(item);

      if (idx === 0) {
        viewRun(r.run_id);
      }
    });
  } catch (err) {
    console.error("Failed to load runs:", err);
  }
}

async function viewRun(runId) {
  try {
    const res = await fetch(`/api/runs/${runId}`);
    if (!res.ok) return;
    const data = await res.json();
    currentRunData = data;
    renderBenchmarkResult(data);
  } catch (err) {
    console.error("Failed to view run:", err);
  }
}

function renderBenchmarkResult(data) {
  const scores = data.scores;
  if (scores) {
    document.getElementById("overallScore").textContent = scores.overall_score.toLocaleString();
    document.getElementById("gradeBadge").textContent = scores.efficiency_grade;
    document.getElementById("tierLabel").textContent = scores.tier_classification;

    document.getElementById("decodeScore").textContent = scores.decode_score.toLocaleString();
    document.getElementById("prefillScore").textContent = scores.prefill_score.toLocaleString();
    document.getElementById("attentionScore").textContent = scores.attention_score.toLocaleString();

    document.getElementById("rtx4090Pct").textContent = `${scores.relative_to_rtx4090_pct}%`;
    document.getElementById("rtx4090Bar").style.width = `${Math.min(scores.relative_to_rtx4090_pct, 100)}%`;

    document.getElementById("m3maxPct").textContent = `${scores.relative_to_m3max_pct}%`;
    document.getElementById("m3maxBar").style.width = `${Math.min(scores.relative_to_m3max_pct, 100)}%`;

    // Show export buttons
    const btnHtml = document.getElementById("btnExportHtml");
    const btnBadge = document.getElementById("btnExportBadge");
    const btnMd = document.getElementById("btnExportMd");
    const btnJson = document.getElementById("btnExportJson");

    if (btnHtml) {
      btnHtml.style.display = "inline-block";
      btnHtml.onclick = () => {
        window.open(`/api/runs/${data.run_id}/html`, "_blank");
      };
    }
    if (btnBadge) {
      btnBadge.style.display = "inline-block";
      btnBadge.onclick = () => {
        window.open(`/api/runs/${data.run_id}/badge`, "_blank");
      };
    }
    if (btnMd) {
      btnMd.style.display = "inline-block";
      btnMd.onclick = () => {
        window.open(`/api/runs/${data.run_id}/markdown`, "_blank");
      };
    }
    if (btnJson) {
      btnJson.style.display = "inline-block";
      btnJson.onclick = () => {
        const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = `${data.run_id}.json`;
        a.click();
      };
    }
  }

  // Render Decode Table
  if (data.decode_result) {
    const tbody = document.getElementById("decodeTableBody");
    tbody.innerHTML = "";
    document.getElementById("decodeSummaryNotes").textContent = data.decode_result.summary_notes;

    data.decode_result.points.forEach((p) => {
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td><code>${p.workload}</code></td>
        <td>${p.batch_size}</td>
        <td>Dim: ${p.hidden_dim} / Seq: ${p.seq_len}</td>
        <td>${p.precision.toUpperCase()}</td>
        <td style="color: var(--accent-green); font-weight: bold;">${p.achieved_bandwidth_gbps.toFixed(2)} GB/s</td>
        <td>${p.stats.median_ms.toFixed(3)} ms</td>
        <td>${p.saturation_pct ? p.saturation_pct + "%" : "N/A"}</td>
      `;
      tbody.appendChild(tr);
    });
  }

  // Render Prefill Table
  if (data.prefill_result) {
    const tbody = document.getElementById("prefillTableBody");
    tbody.innerHTML = "";
    document.getElementById("prefillSummaryNotes").textContent = data.prefill_result.summary_notes;

    data.prefill_result.points.forEach((p) => {
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td>${p.prompt_len}</td>
        <td>${p.batch_size}</td>
        <td>${p.hidden_dim}</td>
        <td>${p.precision.toUpperCase()}</td>
        <td style="color: var(--accent-cyan); font-weight: bold;">${p.achieved_tflops.toFixed(3)} TFLOPS</td>
        <td>${p.tokens_per_sec.toLocaleString()} tok/s</td>
        <td>${p.tflops_efficiency_pct ? p.tflops_efficiency_pct + "%" : "N/A"}</td>
      `;
      tbody.appendChild(tr);
    });
  }

  // Render Attention Table
  if (data.attention_result) {
    const tbody = document.getElementById("attentionTableBody");
    tbody.innerHTML = "";
    document.getElementById("attentionSummaryNotes").textContent = data.attention_result.summary_notes;

    data.attention_result.points.forEach((p) => {
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td>${p.seq_len}</td>
        <td>${p.num_heads} × ${p.head_dim}</td>
        <td><span style="color: ${p.kernel_type === 'sdpa' ? 'var(--accent-green)' : 'var(--text-muted)'}">${p.kernel_type.toUpperCase()}</span></td>
        <td>${p.precision.toUpperCase()}</td>
        <td>${p.achieved_tflops.toFixed(3)}</td>
        <td>${p.stats.median_ms.toFixed(3)} ms</td>
        <td style="color: var(--accent-cyan); font-weight: bold;">${p.speedup_over_naive ? p.speedup_over_naive + "x" : "1.0x"}</td>
      `;
      tbody.appendChild(tr);
    });
  }
}

function setupBenchmarkRunner() {
  const btn = document.getElementById("btnRunBenchmark");
  const progressBox = document.getElementById("benchProgress");
  const progressBar = document.getElementById("benchProgressBar");
  const statusText = document.getElementById("benchStatusText");
  const pctText = document.getElementById("benchPctText");

  btn.addEventListener("click", async () => {
    btn.disabled = true;
    progressBox.style.display = "block";
    progressBar.style.width = "5%";
    statusText.textContent = "Starting benchmark suite...";
    pctText.textContent = "5%";

    const pollInterval = setInterval(async () => {
      try {
        const sRes = await fetch("/api/status");
        if (sRes.ok) {
          const st = await sRes.json();
          if (st.is_running) {
            progressBar.style.width = `${Math.max(5, st.percentage)}%`;
            pctText.textContent = `${Math.round(st.percentage)}%`;
            statusText.textContent = st.message;
          }
        }
      } catch (e) {}
    }, 500);

    try {
      const quick = document.getElementById("chkQuickMode").checked;
      const dec = document.getElementById("chkDecode").checked;
      const pref = document.getElementById("chkPrefill").checked;
      const attn = document.getElementById("chkAttention").checked;
      const power = document.getElementById("chkPower") ? document.getElementById("chkPower").checked : false;

      const res = await fetch("/api/benchmarks/run", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          quick_mode: quick,
          device: "auto",
          run_decode: dec,
          run_prefill: pref,
          run_attention: attn,
          measure_power: power,
        }),
      });

      clearInterval(pollInterval);

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || "Benchmark failed");
      }

      const completedRun = await res.json();
      progressBar.style.width = "100%";
      pctText.textContent = "100%";
      statusText.textContent = "Benchmark completed successfully!";

      renderBenchmarkResult(completedRun);
      loadRuns();
      loadRooflineModel();
    } catch (err) {
      clearInterval(pollInterval);
      statusText.textContent = `Error: ${err.message}`;
      progressBar.style.backgroundColor = "var(--accent-red)";
    } finally {
      btn.disabled = false;
      setTimeout(() => {
        progressBox.style.display = "none";
        progressBar.style.width = "0%";
        progressBar.style.backgroundColor = "var(--accent-cyan)";
      }, 3500);
    }
  });
}

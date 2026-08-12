document.addEventListener("DOMContentLoaded", () => {
  fetchMetrics();
  fetchHistory();
  setInterval(fetchMetrics, 3000);
  setInterval(fetchHistory, 3000);
});

function fillPrompt(text) {
  document.getElementById("prompt").value = text;
}

async function fetchMetrics() {
  try {
    const res = await fetch("/v1/guardrail/metrics");
    if (res.ok) {
      const data = await res.json();
      document.getElementById("metric-total").innerText = data.total_inspections;
      document.getElementById("metric-blocked").innerText = data.total_blocked;
      document.getElementById("metric-passed").innerText = data.total_passed;
      document.getElementById("metric-latency").innerText = `${data.avg_latency_ms.toFixed(1)} ms`;
      document.getElementById("metric-p95").innerText = `${data.p95_latency_ms.toFixed(1)} ms`;
    }
  } catch (err) {
    console.error("Error fetching metrics:", err);
  }
}

async function fetchHistory() {
  try {
    const res = await fetch("/v1/guardrail/history");
    if (res.ok) {
      const logs = await res.json();
      renderAuditLog(logs);
    }
  } catch (err) {
    console.error("Error fetching history:", err);
  }
}

function renderAuditLog(logs) {
  const tbody = document.getElementById("audit-log-body");
  if (!logs || logs.length === 0) {
    tbody.innerHTML = `<tr><td colspan="5" style="text-align: center; color: var(--text-muted);">No inspections recorded yet.</td></tr>`;
    return;
  }

  tbody.innerHTML = logs.map(item => {
    const isSafe = item.is_safe;
    const badgeClass = isSafe ? "badge-safe" : "badge-blocked";
    const actionText = isSafe ? "PASSED_TO_VLM" : "BLOCKED";
    const timeStr = new Date(item.timestamp * 1000).toLocaleTimeString();
    
    return `
      <tr>
        <td style="color: var(--text-muted); font-size: 0.8rem;">${timeStr}</td>
        <td class="prompt-truncate" title="${escapeHtml(item.prompt)}">${escapeHtml(item.prompt)}</td>
        <td><strong style="color: ${item.jailbreak_risk_score > 0.6 ? 'var(--accent-red)' : 'var(--accent-green)'}">${item.jailbreak_risk_score.toFixed(4)}</strong></td>
        <td>${item.latency_ms.toFixed(1)} ms</td>
        <td><span class="badge ${badgeClass}">${actionText}</span></td>
      </tr>
    `;
  }).join("");
}

function escapeHtml(str) {
  return str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}

document.getElementById("guardrail-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const promptInput = document.getElementById("prompt").value;
  const imageInput = document.getElementById("image_file").files[0];
  const btnInspect = document.getElementById("btn-inspect");

  btnInspect.disabled = true;
  btnInspect.innerText = "Inspecting Multimodal Intent...";

  const formData = new FormData();
  formData.append("prompt", promptInput);
  if (imageInput) {
    formData.append("image_file", imageInput);
  }

  try {
    const startTime = performance.now();
    const res = await fetch("/v1/guardrail/inspect", {
      method: "POST",
      body: formData
    });
    const endTime = performance.now();

    if (res.ok || res.status === 403) {
      const data = await res.json();
      displayResult(data);
      fetchMetrics();
      fetchHistory();
    } else {
      alert("Error inspecting prompt: " + res.statusText);
    }
  } catch (err) {
    console.error("Inspection error:", err);
    alert("Failed to connect to PolyGuard-VLM_Plus server.");
  } finally {
    btnInspect.disabled = false;
    btnInspect.innerText = "Inspect Security Guardrail";
  }
});

function displayResult(data) {
  const resultBox = document.getElementById("result-box");
  const resultBadge = document.getElementById("result-badge");
  const resultLatency = document.getElementById("result-latency");
  const resultScore = document.getElementById("result-score");
  const riskBar = document.getElementById("risk-bar");
  const resultLang = document.getElementById("result-lang");
  const resultVlm = document.getElementById("result-vlm");

  resultBox.classList.add("show");
  
  if (data.is_safe) {
    resultBadge.className = "badge badge-safe";
    resultBadge.innerText = "PASSED (SAFE INTENT)";
    riskBar.style.backgroundColor = "var(--accent-green)";
  } else {
    resultBadge.className = "badge badge-blocked";
    resultBadge.innerText = "BLOCKED (ADVERSARIAL INTENT)";
    riskBar.style.backgroundColor = "var(--accent-red)";
  }

  const scorePct = Math.min(100, Math.max(0, data.jailbreak_risk_score * 100));
  riskBar.style.width = `${scorePct}%`;

  resultLatency.innerText = `Latency: ${data.latency_ms.toFixed(2)} ms`;
  resultScore.innerText = `${data.jailbreak_risk_score.toFixed(4)}`;
  resultLang.innerText = data.language_detected || "Auto-CrossLingual-LaBSE";
  
  if (data.vlm_response) {
    resultVlm.innerText = typeof data.vlm_response === "string" ? data.vlm_response : JSON.stringify(data.vlm_response, null, 2);
  } else if (data.detail) {
    resultVlm.innerText = `Security Exception 403: ${data.detail}`;
  } else {
    resultVlm.innerText = "No VLM response returned.";
  }
}

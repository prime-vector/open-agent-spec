"""Dashboard -- a lightweight web UI for watching the orchestration loop.

Starts a FastAPI server that serves:
  GET  /           → HTML dashboard (auto-refreshes)
  GET  /api/status → JSON snapshot of board + agents + events
  POST /api/run    → Kick off a new objective (async)

No JS framework needed — uses plain HTML with fetch polling.
"""

from __future__ import annotations

import json
import logging
import threading
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# We import FastAPI lazily so the rest of the package doesn't require it.
_app = None
_loop_instance = None
_run_thread: Optional[threading.Thread] = None
_last_result: Optional[Dict[str, Any]] = None


def _get_app():  # noqa: ANN202
    global _app
    if _app is not None:
        return _app

    try:
        from fastapi import FastAPI
        from fastapi.responses import HTMLResponse, JSONResponse
    except ImportError:
        raise ImportError(
            "The dashboard requires fastapi and uvicorn.\n"
            "Install them with: pip install -r examples/multi-agent/requirements.txt"
        )

    _app = FastAPI(title="OA Spec Orchestration Dashboard")

    @_app.get("/", response_class=HTMLResponse)
    async def dashboard() -> str:
        return _DASHBOARD_HTML

    @_app.get("/api/status", response_class=JSONResponse)
    async def status() -> Dict[str, Any]:
        if _loop_instance is None:
            return {"error": "No orchestration loop configured"}
        data = _loop_instance.status()
        if _last_result:
            data["last_result"] = {
                "objective": _last_result.get("objective"),
                "iterations": _last_result.get("iterations"),
                "board": _last_result.get("board"),
            }
        data["running"] = _run_thread is not None and _run_thread.is_alive()
        return data

    @_app.post("/api/run", response_class=JSONResponse)
    async def run_objective(payload: Dict[str, Any]) -> Dict[str, Any]:
        global _run_thread, _last_result
        if _loop_instance is None:
            return {"error": "No orchestration loop configured"}
        if _run_thread and _run_thread.is_alive():
            return {"error": "Already running"}

        objective = payload.get("objective", "")
        if not objective:
            return {"error": "objective is required"}

        def _run() -> None:
            global _last_result
            try:
                _last_result = _loop_instance.run(objective)
            except Exception as exc:
                logger.error("Orchestration failed: %s", exc)
                _last_result = {"error": str(exc)}

        _run_thread = threading.Thread(target=_run, daemon=True)
        _run_thread.start()
        return {"status": "started", "objective": objective}

    return _app


def serve(
    loop: Any,
    host: str = "0.0.0.0",
    port: int = 8420,
) -> None:
    """Start the dashboard server with the given orchestration loop."""
    global _loop_instance
    _loop_instance = loop
    app = _get_app()
    import uvicorn

    logger.info("Dashboard starting at http://%s:%s", host, port)
    uvicorn.run(app, host=host, port=port, log_level="info")


# ---------------------------------------------------------------------------
# Inline HTML dashboard — no template files needed.
# ---------------------------------------------------------------------------

_DASHBOARD_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>OA Spec — Orchestration Dashboard</title>
<style>
  :root {
    --bg: #0f1117; --surface: #1a1d27; --border: #2a2d3a;
    --text: #e1e4ed; --muted: #8b8fa3;
    --green: #4ade80; --blue: #60a5fa; --amber: #fbbf24;
    --red: #f87171; --purple: #a78bfa;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: 'Inter', system-ui, -apple-system, sans-serif;
    background: var(--bg); color: var(--text);
    line-height: 1.5; padding: 24px; max-width: 1400px; margin: 0 auto;
  }
  h1 { font-size: 1.5rem; margin-bottom: 8px; }
  h2 { font-size: 1.1rem; color: var(--muted); margin-bottom: 16px; font-weight: 400; }
  h3 { font-size: 0.95rem; text-transform: uppercase; letter-spacing: 0.05em;
       color: var(--muted); margin-bottom: 12px; }

  .header { display: flex; justify-content: space-between; align-items: center;
            margin-bottom: 32px; flex-wrap: wrap; gap: 16px; }
  .status-badge { padding: 4px 12px; border-radius: 12px; font-size: 0.8rem;
                  font-weight: 600; }
  .status-badge.running { background: rgba(74,222,128,0.15); color: var(--green); }
  .status-badge.idle { background: rgba(139,143,163,0.15); color: var(--muted); }

  .input-row { display: flex; gap: 8px; margin-bottom: 32px; }
  .input-row input {
    flex: 1; padding: 10px 14px; border-radius: 8px; border: 1px solid var(--border);
    background: var(--surface); color: var(--text); font-size: 0.95rem;
  }
  .input-row input:focus { outline: none; border-color: var(--blue); }
  .input-row button {
    padding: 10px 24px; border-radius: 8px; border: none;
    background: var(--blue); color: #fff; font-weight: 600; cursor: pointer;
    font-size: 0.95rem;
  }
  .input-row button:hover { opacity: 0.9; }
  .input-row button:disabled { opacity: 0.5; cursor: not-allowed; }

  .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 24px; margin-bottom: 32px; }
  @media (max-width: 900px) { .grid { grid-template-columns: 1fr; } }

  .card {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 12px; padding: 20px;
  }

  .stat-row { display: flex; gap: 16px; margin-bottom: 24px; flex-wrap: wrap; }
  .stat {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 12px; padding: 16px 20px; flex: 1; min-width: 120px;
  }
  .stat .value { font-size: 2rem; font-weight: 700; }
  .stat .label { font-size: 0.8rem; color: var(--muted); }

  table { width: 100%; border-collapse: collapse; font-size: 0.85rem; }
  th { text-align: left; padding: 8px; color: var(--muted); font-weight: 500;
       border-bottom: 1px solid var(--border); }
  td { padding: 8px; border-bottom: 1px solid var(--border); }

  .badge { padding: 2px 8px; border-radius: 6px; font-size: 0.75rem; font-weight: 600; }
  .badge.pending { background: rgba(139,143,163,0.15); color: var(--muted); }
  .badge.assigned, .badge.in_progress { background: rgba(96,165,250,0.15); color: var(--blue); }
  .badge.completed { background: rgba(74,222,128,0.15); color: var(--green); }
  .badge.failed { background: rgba(248,113,113,0.15); color: var(--red); }

  .event-log {
    max-height: 400px; overflow-y: auto; font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem; line-height: 1.8;
  }
  .event-log .ev { padding: 2px 0; border-bottom: 1px solid var(--border); }
  .event-log .ev-type { color: var(--purple); font-weight: 600; }
  .event-log .ev-time { color: var(--muted); margin-right: 8px; }
</style>
</head>
<body>

<div class="header">
  <div>
    <h1>OA Spec Orchestration</h1>
    <h2>Multi-agent workflow dashboard</h2>
  </div>
  <span id="status-badge" class="status-badge idle">Idle</span>
</div>

<div class="input-row">
  <input id="objective" type="text" placeholder="Enter an objective for the agent team..." />
  <button id="run-btn" onclick="runObjective()">Run</button>
</div>

<div class="stat-row">
  <div class="stat"><div class="value" id="s-total">0</div><div class="label">Total Tasks</div></div>
  <div class="stat"><div class="value" id="s-completed" style="color:var(--green)">0</div><div class="label">Completed</div></div>
  <div class="stat"><div class="value" id="s-progress" style="color:var(--blue)">0</div><div class="label">In Progress</div></div>
  <div class="stat"><div class="value" id="s-agents" style="color:var(--purple)">0</div><div class="label">Agents</div></div>
</div>

<div class="grid">
  <div class="card">
    <h3>Task Board</h3>
    <table>
      <thead><tr><th>Task</th><th>Role</th><th>Agent</th><th>Status</th></tr></thead>
      <tbody id="task-body"></tbody>
    </table>
  </div>
  <div class="card">
    <h3>Agents</h3>
    <table>
      <thead><tr><th>Agent</th><th>Role</th><th>Done</th><th>Failed</th></tr></thead>
      <tbody id="agent-body"></tbody>
    </table>
  </div>
</div>

<div class="card">
  <h3>Event Log</h3>
  <div id="event-log" class="event-log"></div>
</div>

<script>
const POLL_MS = 1500;

async function poll() {
  try {
    const res = await fetch('/api/status');
    const d = await res.json();
    if (d.error) return;

    // Status badge
    const badge = document.getElementById('status-badge');
    badge.textContent = d.running ? 'Running' : 'Idle';
    badge.className = 'status-badge ' + (d.running ? 'running' : 'idle');
    document.getElementById('run-btn').disabled = d.running;

    // Stats
    const bs = d.board?.by_status || {};
    document.getElementById('s-total').textContent = d.board?.total || 0;
    document.getElementById('s-completed').textContent = bs.completed || 0;
    document.getElementById('s-progress').textContent =
      (bs.in_progress || 0) + (bs.assigned || 0);
    document.getElementById('s-agents').textContent = d.agents?.length || 0;

    // Task table
    const tb = document.getElementById('task-body');
    tb.innerHTML = (d.tasks || []).map(t => `
      <tr>
        <td>${esc(t.title)}</td>
        <td>${esc(t.required_role)}</td>
        <td>${esc(t.assigned_to || '-')}</td>
        <td><span class="badge ${t.status}">${t.status}</span></td>
      </tr>`).join('');

    // Agent table
    const ab = document.getElementById('agent-body');
    ab.innerHTML = (d.agents || []).map(a => `
      <tr>
        <td>${esc(a.id)}</td>
        <td>${esc(a.role)}</td>
        <td>${a.tasks_completed}</td>
        <td>${a.tasks_failed}</td>
      </tr>`).join('');

    // Events
    const el = document.getElementById('event-log');
    el.innerHTML = (d.events || []).map(e => {
      const t = new Date(e.timestamp * 1000).toLocaleTimeString();
      const extra = Object.keys(e).filter(k => !['type','timestamp'].includes(k))
        .map(k => `${k}=${JSON.stringify(e[k]).slice(0,80)}`).join(' ');
      return `<div class="ev"><span class="ev-time">${t}</span><span class="ev-type">${e.type}</span> ${esc(extra)}</div>`;
    }).reverse().join('');
  } catch(e) {}
}

function esc(s) { const d = document.createElement('div'); d.textContent = s || ''; return d.innerHTML; }

async function runObjective() {
  const obj = document.getElementById('objective').value.trim();
  if (!obj) return;
  document.getElementById('run-btn').disabled = true;
  await fetch('/api/run', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({objective: obj}),
  });
}

setInterval(poll, POLL_MS);
poll();
</script>
</body>
</html>
"""

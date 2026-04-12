"""Dashboard -- a lightweight web UI for watching the orchestration loop.

Starts a FastAPI server that serves:
  GET  /           → HTML dashboard (auto-refreshes)
  GET  /api/status → JSON snapshot of board + agents + events
  POST /api/run    → Kick off a new objective (async)

No JS framework needed — uses plain HTML with fetch polling.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

logger = logging.getLogger(__name__)

# We import FastAPI lazily so the rest of the package doesn't require it.
_app = None
_loop_instance = None
_run_thread: threading.Thread | None = None
_last_result: dict[str, Any] | None = None


def _get_app():
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
    async def status() -> dict[str, Any]:
        if _loop_instance is None:
            return {"error": "No orchestration loop configured"}
        data = _loop_instance.status()
        if _last_result:
            data["last_result"] = {
                "objective": _last_result.get("objective"),
                "iterations": _last_result.get("iterations"),
                "board": _last_result.get("board"),
                "summary": _last_result.get("summary"),
                "tasks": _last_result.get("tasks", []),
            }
        data["running"] = _run_thread is not None and _run_thread.is_alive()
        return data

    @_app.post("/api/run", response_class=JSONResponse)
    async def run_objective(payload: dict[str, Any]) -> dict[str, Any]:
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
<title>OA Orchestration Dashboard</title>
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

  .section-toggle {
    background: none; border: 1px solid var(--border); color: var(--muted);
    border-radius: 6px; padding: 4px 12px; font-size: 0.8rem; cursor: pointer;
    float: right; margin-top: -2px;
  }
  .section-toggle:hover { color: var(--text); border-color: var(--text); }

  .results-panel { margin-bottom: 24px; }
  .results-summary {
    background: rgba(74,222,128,0.07); border: 1px solid rgba(74,222,128,0.2);
    border-radius: 8px; padding: 14px 16px; margin-bottom: 16px;
    font-size: 0.9rem; line-height: 1.6; white-space: pre-wrap;
  }
  .results-objective {
    font-size: 0.85rem; color: var(--muted); margin-bottom: 12px;
    font-style: italic;
  }
  .task-result { margin-bottom: 12px; border: 1px solid var(--border); border-radius: 8px; overflow: hidden; }
  .task-result-header {
    display: flex; justify-content: space-between; align-items: center;
    padding: 10px 14px; background: var(--surface); cursor: pointer;
    font-size: 0.85rem;
  }
  .task-result-header:hover { background: rgba(255,255,255,0.03); }
  .task-result-header .role-badge {
    font-size: 0.75rem; color: var(--blue); background: rgba(96,165,250,0.1);
    padding: 2px 8px; border-radius: 4px;
  }
  .task-result-header .collapse-icon {
    font-size: 0.7rem; color: var(--muted); margin-right: 8px;
  }
  .task-result-body {
    padding: 14px 16px; border-top: 1px solid var(--border);
    font-size: 0.85rem; line-height: 1.7; white-space: pre-wrap;
    word-break: break-word;
  }
  .task-result-body p { margin-bottom: 8px; }
  .task-result-body ul, .task-result-body ol { margin: 8px 0 8px 20px; }
  .task-result-body li { margin-bottom: 4px; }
  .task-result-body strong { color: var(--blue); }

  /* Task summary cards */
  .task-summaries { display: flex; flex-direction: column; gap: 10px; }
  .task-summary-card {
    background: rgba(96,165,250,0.05); border: 1px solid var(--border);
    border-radius: 8px; padding: 14px 16px;
  }
  .task-summary-card .task-title {
    font-size: 0.85rem; font-weight: 600; color: var(--blue);
    margin-bottom: 6px; display: flex; justify-content: space-between; align-items: center;
  }
  .task-summary-card .task-title .role-badge {
    font-size: 0.7rem; color: var(--muted); background: rgba(139,143,163,0.12);
    padding: 2px 8px; border-radius: 4px; font-weight: 500;
  }
  .task-summary-card .task-content {
    font-size: 0.85rem; line-height: 1.7; color: var(--text);
  }
  .task-summary-card .task-content p { margin-bottom: 6px; }
  .task-summary-card .task-content ul, .task-summary-card .task-content ol { margin: 6px 0 6px 20px; }
  .task-summary-card .task-content li { margin-bottom: 3px; }
  .task-summary-card .task-content table {
    width: 100%; border-collapse: collapse; margin: 12px 0; font-size: 0.82rem;
  }
  .task-summary-card .task-content th {
    text-align: left; padding: 6px 10px; background: rgba(96,165,250,0.1);
    color: var(--blue); font-weight: 600; border-bottom: 2px solid var(--border);
  }
  .task-summary-card .task-content td {
    padding: 5px 10px; border-bottom: 1px solid var(--border);
  }
  .task-summary-card .task-content tr:hover td { background: rgba(255,255,255,0.02); }

  /* History */
  .history-panel { margin-bottom: 24px; }
  .history-run { margin-bottom: 12px; border: 1px solid var(--border); border-radius: 8px; overflow: hidden; }
  .history-run-header {
    display: flex; justify-content: space-between; align-items: center;
    padding: 10px 14px; background: rgba(139,143,163,0.06); cursor: pointer;
    font-size: 0.85rem;
  }
  .history-run-header:hover { background: rgba(255,255,255,0.03); }
  .history-run-header .run-time { font-size: 0.75rem; color: var(--muted); }
  .history-run-header .run-stats { font-size: 0.75rem; color: var(--green); }
  .history-run-body { display: none; padding: 12px 14px; border-top: 1px solid var(--border); }
</style>
</head>
<body>

<div class="header">
  <div>
    <h1>OA Orchestration</h1>
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

<div id="results-section" class="results-panel card" style="display:none">
  <h3>Results</h3>
  <div id="results-objective" class="results-objective"></div>
  <div id="results-summary" class="results-summary" style="display:none"></div>
  <div id="task-results"></div>
</div>

<div id="history-section" class="history-panel card" style="display:none">
  <h3>Previous Runs <button class="section-toggle" onclick="clearHistory()">Clear</button></h3>
  <div id="history-list"></div>
</div>

<div class="card">
  <h3>Event Log <button class="section-toggle" onclick="toggleEventLog()">Show</button></h3>
  <div id="event-log" class="event-log" style="display:none"></div>
</div>

<script>
const POLL_MS = 1500;
let runHistory = [];
let resultsRendered = false;

// -- Parse a result value: unwrap { result: "json string" } and strip markdown fences --
function parseResult(result) {
  if (!result) return null;
  let raw = result;
  if (typeof raw === 'object' && raw.result && typeof raw.result === 'string') {
    raw = raw.result;
  }
  if (typeof raw === 'string') {
    raw = raw.replace(/```json\\s*/gi, '').replace(/```\\s*/g, '');
    raw = raw.replace(/^#[^{]*(?=\\{)/s, '');
    raw = raw.trim();
    try { return JSON.parse(raw); } catch(e) { return raw; }
  }
  return raw;
}

// -- Extract a concise summary from a parsed result --
function extractBrief(obj) {
  if (typeof obj === 'string') {
    return obj.length > 300 ? obj.slice(0, 300) + '...' : obj;
  }
  if (!obj || typeof obj !== 'object') return String(obj || '');

  const lines = [];

  // Title
  const title = obj.title || obj.research_topic || obj.document_type || '';
  if (title) lines.push(title);

  // Collect top-level string values as key points
  const skipKeys = new Set(['format','type','version','research_date','document_type',
    'research_topic','title','research_metadata','methodology','data_quality_note',
    'scope','evaluation_criteria','review_process','disclaimer']);
  for (const [k, v] of Object.entries(obj)) {
    if (skipKeys.has(k)) continue;
    if (typeof v === 'string' && v.length > 10 && v.length < 500) {
      lines.push(v);
    }
  }

  // Extract names from arrays (activities, recommendations, etc.)
  for (const [k, v] of Object.entries(obj)) {
    if (skipKeys.has(k)) continue;
    if (Array.isArray(v)) {
      const items = v.slice(0, 8).map(item => {
        if (typeof item === 'string') return item;
        if (item && typeof item === 'object') {
          return item.name || item.title || item.activity || item.action || item.description || '';
        }
        return '';
      }).filter(Boolean);
      if (items.length > 0) {
        lines.push(items.map(i => '\\u2022 ' + i).join('\\n'));
      }
    }
  }

  // If we got nothing, try one level deeper
  if (lines.length <= 1) {
    for (const [k, v] of Object.entries(obj)) {
      if (skipKeys.has(k) || typeof v !== 'object' || Array.isArray(v) || !v) continue;
      const sub = extractBrief(v);
      if (sub && sub.length > 10) { lines.push(sub); break; }
    }
  }

  return lines.join('\\n\\n') || JSON.stringify(obj).slice(0, 200);
}

// -- Render markdown-ish text as HTML --
function renderMarkdown(text) {
  // First, extract and convert markdown tables before escaping
  var lines = text.split('\\n');
  var out = [];
  var inTable = false;
  var tableRows = [];

  for (var li = 0; li < lines.length; li++) {
    var line = lines[li].trim();
    if (line.match(/^\\|.+\\|$/) && line.indexOf('|') > 0) {
      // This is a table row
      if (line.match(/^\\|[\\s\\-:|]+\\|$/)) {
        // Separator row — skip but mark we have a header
        continue;
      }
      tableRows.push(line);
      inTable = true;
    } else {
      if (inTable && tableRows.length > 0) {
        // Flush table
        out.push(buildTable(tableRows));
        tableRows = [];
        inTable = false;
      }
      out.push(line);
    }
  }
  if (tableRows.length > 0) out.push(buildTable(tableRows));

  var h = esc(out.join('\\n'));
  // Restore table HTML that was escaped
  h = h.replace(/&lt;table/g, '<table').replace(/&lt;\\/table&gt;/g, '</table>');
  h = h.replace(/&lt;thead/g, '<thead').replace(/&lt;\\/thead&gt;/g, '</thead>');
  h = h.replace(/&lt;tbody/g, '<tbody').replace(/&lt;\\/tbody&gt;/g, '</tbody>');
  h = h.replace(/&lt;tr/g, '<tr').replace(/&lt;\\/tr&gt;/g, '</tr>');
  h = h.replace(/&lt;th/g, '<th').replace(/&lt;\\/th&gt;/g, '</th>');
  h = h.replace(/&lt;td/g, '<td').replace(/&lt;\\/td&gt;/g, '</td>');
  h = h.replace(/&amp;gt;/g, '>');
  // Headers
  h = h.replace(/^### (.+)$/gm, '<h4 style="color:var(--blue);margin:16px 0 6px;">$1</h4>');
  h = h.replace(/^## (.+)$/gm, '<h3 style="color:var(--blue);margin:20px 0 8px;">$1</h3>');
  h = h.replace(/^# (.+)$/gm, '<h2 style="color:var(--green);margin:24px 0 10px;">$1</h2>');
  // Horizontal rules
  h = h.replace(/^---$/gm, '<hr style="border:none;border-top:1px solid var(--border);margin:16px 0;">');
  // Bold
  h = h.replace(/\\*\\*(.+?)\\*\\*/g, '<strong>$1</strong>');
  // Bullet lists
  h = h.replace(/^- (.+)$/gm, '<li>$1</li>');
  h = h.replace(/(<li>.*?<\\/li>(?:\\n<li>.*?<\\/li>)*)/gs, '<ul style="margin:6px 0 6px 20px;">$1</ul>');
  // Numbered lists
  h = h.replace(/^\\d+\\. (.+)$/gm, '<li>$1</li>');
  // Double newlines → paragraph
  h = h.replace(/\\n\\n+/g, '</p><p style="margin-bottom:8px;">');
  h = '<p style="margin-bottom:8px;">' + h + '</p>';
  // Single newlines → br
  h = h.replace(/\\n/g, '<br>');
  return h;
}

function buildTable(rows) {
  if (rows.length === 0) return '';
  var cells = rows.map(function(r) {
    return r.split('|').filter(function(c) { return c.trim() !== ''; }).map(function(c) { return c.trim(); });
  });
  var html = '<table>';
  // First row as header
  html += '<thead><tr>' + cells[0].map(function(c) { return '<th>' + c + '</th>'; }).join('') + '</tr></thead>';
  if (cells.length > 1) {
    html += '<tbody>';
    for (var ri = 1; ri < cells.length; ri++) {
      html += '<tr>' + cells[ri].map(function(c) { return '<td>' + c + '</td>'; }).join('') + '</tr>';
    }
    html += '</tbody>';
  }
  html += '</table>';
  return html;
}

// -- Render a parsed object as nicely formatted HTML --
function renderObject(obj, depth) {
  if (!obj) return '';
  if (typeof obj === 'string') return '<p>' + esc(obj) + '</p>';
  if (typeof obj === 'number' || typeof obj === 'boolean') return esc(String(obj));
  if (depth > 3) return '';

  var skipKeys = {format:1, type:1, version:1, document_type:1, research_date:1,
    research_metadata:1, data_quality_note:1, evaluation_criteria:1, review_process:1,
    disclaimer:1, methodology:1, scope:1};

  if (Array.isArray(obj)) {
    // Array of strings: bullet list
    if (obj.length > 0 && typeof obj[0] === 'string') {
      return '<ul>' + obj.map(function(s) { return '<li>' + esc(s) + '</li>'; }).join('') + '</ul>';
    }
    // Array of objects: render each as a card
    return obj.map(function(item, idx) {
      if (typeof item !== 'object' || !item) return '<p>' + esc(String(item)) + '</p>';
      var name = item.name || item.title || item.activity || item.action || '';
      var desc = item.description || '';
      var h = '<div style="margin:10px 0;padding:10px 14px;border-left:3px solid var(--blue);background:rgba(96,165,250,0.04);border-radius:4px;">';
      if (item.rank) h += '<span style="color:var(--green);font-weight:700;margin-right:8px;">#' + item.rank + '</span>';
      if (name) h += '<strong style="color:var(--blue);">' + esc(name) + '</strong><br>';
      if (desc) h += '<span>' + esc(desc) + '</span><br>';
      // Show useful fields
      var fields = ['location','distance_from_hornsby','cost','opening_hours','weather',
        'infant_friendliness','accessibility','why_suits_group','practical_tips',
        'priority','status','reason','message','review_status','decision'];
      for (var fi = 0; fi < fields.length; fi++) {
        var fk = fields[fi];
        var fv = item[fk];
        if (!fv) continue;
        var label = fk.replace(/_/g, ' ');
        if (typeof fv === 'string') {
          h += '<div style="margin-top:4px;"><strong style="color:var(--muted);font-size:0.82rem;">' + esc(label) + ':</strong> ' + esc(fv) + '</div>';
        } else if (Array.isArray(fv) && typeof fv[0] === 'string') {
          h += '<div style="margin-top:4px;"><strong style="color:var(--muted);font-size:0.82rem;">' + esc(label) + ':</strong></div>';
          h += '<ul>' + fv.map(function(s) { return '<li>' + esc(s) + '</li>'; }).join('') + '</ul>';
        } else if (typeof fv === 'object' && !Array.isArray(fv)) {
          // Render sub-object inline
          var subParts = [];
          for (var sk in fv) {
            if (typeof fv[sk] === 'string') subParts.push(esc(sk.replace(/_/g,' ')) + ': ' + esc(fv[sk]));
          }
          if (subParts.length) {
            h += '<div style="margin-top:4px;"><strong style="color:var(--muted);font-size:0.82rem;">' + esc(label) + ':</strong> ' + subParts.join(' | ') + '</div>';
          }
        }
      }
      h += '</div>';
      return h;
    }).join('');
  }

  // Plain object: render key-value sections
  var html = '';
  for (var key in obj) {
    if (skipKeys[key]) continue;
    var val = obj[key];
    if (val === null || val === undefined) continue;
    var label = key.replace(/_/g, ' ').replace(/(?:^|\\s)\\w/g, function(c) { return c.toUpperCase(); });

    if (typeof val === 'string') {
      html += '<div style="margin-bottom:8px;"><strong style="color:var(--blue);">' + esc(label) + ':</strong> ' + esc(val) + '</div>';
    } else if (typeof val === 'object') {
      html += '<div style="margin-bottom:12px;"><strong style="color:var(--blue);font-size:0.9rem;">' + esc(label) + '</strong>';
      html += '<div style="padding-left:8px;margin-top:4px;">' + renderObject(val, depth + 1) + '</div></div>';
    }
  }
  return html;
}

// -- Render results: writer output as hero, rest collapsed --
function renderSummaryCards(lr) {
  var container = document.getElementById('task-results');
  if (!container) return;

  var workerTasks = (lr.tasks || []).filter(function(t) {
    return t.result && t.required_role !== 'planner' && t.required_role !== 'chat';
  });

  var finalTask = null;
  var supportTasks = [];
  for (var ti = 0; ti < workerTasks.length; ti++) {
    if (workerTasks[ti].required_role === 'writer') finalTask = workerTasks[ti];
    else supportTasks.push(workerTasks[ti]);
  }

  var html = '';

  // Hero: writer output — look for 'content' field first (markdown text)
  if (finalTask) {
    var parsed = parseResult(finalTask.result);
    var bodyHTML = '';

    // If the result has a 'content' string, render it as markdown
    var contentStr = null;
    if (typeof parsed === 'string') contentStr = parsed;
    else if (parsed && parsed.content && typeof parsed.content === 'string') contentStr = parsed.content;

    if (contentStr && contentStr.length > 50) {
      bodyHTML = renderMarkdown(contentStr);
    } else {
      // Fallback to structured rendering
      bodyHTML = renderObject(parsed, 0);
    }

    html += '<div class="task-summary-card" style="border:1px solid rgba(74,222,128,0.3);background:rgba(74,222,128,0.04);">';
    html += '<div class="task-title" style="margin-bottom:10px;">';
    html += '<span>' + esc(finalTask.title) + '</span>';
    html += '<span class="role-badge" style="background:rgba(74,222,128,0.12);color:var(--green);">writer</span>';
    html += '</div>';
    html += '<div class="task-content" style="line-height:1.8;">' + bodyHTML + '</div>';
    html += '</div>';
  }

  // Supporting work: collapsed
  if (supportTasks.length > 0) {
    html += '<div style="margin-top:16px;">';
    html += '<div style="cursor:pointer;color:var(--muted);font-size:0.85rem;margin-bottom:8px;" onclick="toggleEl(&quot;support-tasks&quot;)">';
    html += '&#9654; Supporting work (' + supportTasks.length + ' tasks)</div>';
    html += '<div id="support-tasks" style="display:none;">';
    for (var st = 0; st < supportTasks.length; st++) {
      var t = supportTasks[st];
      var p = parseResult(t.result);
      var brief = extractBrief(p);
      html += '<div class="task-summary-card" style="margin-bottom:8px;">';
      html += '<div class="task-title"><span>' + esc(t.title) + '</span>';
      html += '<span class="role-badge">' + esc(t.required_role) + '</span></div>';
      html += '<div class="task-content" style="font-size:0.82rem;color:var(--muted);">' + esc(brief).replace(/\\n/g, '<br>') + '</div>';
      html += '</div>';
    }
    html += '</div></div>';
  }

  container.innerHTML = html;
}

// -- History management --
function archiveCurrentResult(lr) {
  if (!lr || !lr.objective) return;
  const id = lr.objective + '_' + (lr.iterations || 0);
  if (runHistory.some(h => h.id === id)) return;

  const workerTasks = (lr.tasks || []).filter(t => t.result && !['planner','chat'].includes(t.required_role));
  runHistory.unshift({
    id: id,
    objective: lr.objective,
    summary: lr.summary?.summary || '',
    tasks: lr.tasks || [],
    taskCount: workerTasks.length,
    time: new Date().toLocaleTimeString(),
  });
  if (runHistory.length > 10) runHistory.pop();
  renderHistory();
}

function renderHistory() {
  const section = document.getElementById('history-section');
  const list = document.getElementById('history-list');
  if (runHistory.length === 0) { section.style.display = 'none'; return; }

  section.style.display = '';
  list.innerHTML = runHistory.map((h, i) => {
    const workerTasks = (h.tasks || []).filter(t => t.result && !['planner','chat'].includes(t.required_role));
    const taskPreviews = workerTasks.map(t => {
      const parsed = parseResult(t.result);
      // Grab a short text preview
      let preview = '';
      if (typeof parsed === 'string') { preview = parsed; }
      else if (parsed?.title) { preview = parsed.title; }
      else if (parsed?.research_topic) { preview = parsed.research_topic; }
      else { preview = JSON.stringify(parsed).slice(0, 120); }
      if (preview.length > 150) preview = preview.slice(0, 150) + '...';
      return `<div style="margin-bottom:6px;"><strong style="color:var(--blue);font-size:0.82rem;">${esc(t.title)}</strong>
        <span class="role-badge" style="margin-left:6px;">${esc(t.required_role)}</span>
        <div style="font-size:0.8rem;color:var(--muted);margin-top:2px;">${esc(preview)}</div></div>`;
    }).join('');

    return `
      <div class="history-run">
        <div class="history-run-header" onclick="toggleEl('history-body-${i}')">
          <span>${esc(h.objective)}</span>
          <span>
            <span class="run-stats">${h.taskCount} tasks</span>
            <span class="run-time">${h.time}</span>
          </span>
        </div>
        <div class="history-run-body" id="history-body-${i}">
          ${h.summary ? '<div class="results-summary">' + esc(h.summary) + '</div>' : ''}
          ${taskPreviews}
        </div>
      </div>`;
  }).join('');
}

function clearHistory() {
  runHistory = [];
  renderHistory();
}

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

    // Results — render ONCE when complete, never re-render (preserves UI state)
    const lr = d.last_result;
    const resultsSection = document.getElementById('results-section');
    if (lr && !d.running) {
      if (!resultsRendered) {
        resultsRendered = true;
        resultsSection.style.display = '';
        document.getElementById('results-objective').textContent = 'Objective: ' + (lr.objective || '');

        const summaryEl = document.getElementById('results-summary');
        const summaryText = lr.summary?.summary;
        if (summaryText) {
          summaryEl.style.display = '';
          summaryEl.textContent = summaryText;
        } else {
          summaryEl.style.display = 'none';
        }

        try {
          renderSummaryCards(lr);
        } catch(renderErr) {
          console.error('Render error:', renderErr);
          // Fallback: show raw JSON
          document.getElementById('task-results').innerHTML =
            '<pre style="white-space:pre-wrap;font-size:0.8rem;">' + esc(JSON.stringify(lr.tasks, null, 2)) + '</pre>';
        }
      }
    } else if (d.running) {
      resultsRendered = false;
    } else if (!lr) {
      resultsSection.style.display = 'none';
    }
  } catch(e) { console.error('Poll error:', e); }
}

function toggleEventLog() {
  const el = document.getElementById('event-log');
  const visible = el.style.display !== 'none';
  el.style.display = visible ? 'none' : '';
  el.parentElement.querySelector('.section-toggle').textContent = visible ? 'Show' : 'Hide';
}

function toggleEl(id) {
  const body = document.getElementById(id);
  if (!body) return;
  const visible = body.style.display !== 'none';
  body.style.display = visible ? 'none' : '';
}

function esc(s) { const d = document.createElement('div'); d.textContent = s || ''; return d.innerHTML; }

async function runObjective() {
  const obj = document.getElementById('objective').value.trim();
  if (!obj) return;

  // Archive current result to history
  if (document.getElementById('results-section').style.display !== 'none') {
    try {
      const statusRes = await fetch('/api/status');
      const statusData = await statusRes.json();
      if (statusData.last_result) archiveCurrentResult(statusData.last_result);
    } catch(e) {}
  }

  // Clear current results
  resultsRendered = false;
  document.getElementById('results-section').style.display = 'none';
  document.getElementById('task-results').innerHTML = '';
  document.getElementById('results-summary').style.display = 'none';

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

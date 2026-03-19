# OA CLI 1.2.8 — manual test plan

**Purpose:** Step-by-step checks for **open-agent-spec** CLI (**`oa`** command) before sign-off.

**Environment:** Python 3.10+; clean shell; optional `OPENAI_API_KEY` (or other engine key) for `oa run`.

---

## 0. Prerequisites

| Step | Action | Pass if |
|------|--------|----------|
| 0.1 | `python --version` | Shows 3.10 or newer |
| 0.2 | Install the build under test | e.g. `pip install open-agent-spec==1.2.8` or `pip install .` from repo |
| 0.3 | `oa --version` | Prints **1.2.8** (or `python -m oas_cli --version` if `oa` not on PATH) |

**Note:** The console script is **`oa`** (not `oas`). If `oa` is not found, use:

```bash
python -m oas_cli --version
```

---

## 1. Version & help

| # | Command | What to check |
|---|---------|----------------|
| 1.1 | `oa --version` | Exits 0; shows version **1.2.8** |
| 1.2 | `oa version` | Same as 1.1 (dedicated version command) |
| 1.3 | `oa --help` | Lists commands: `init`, `run`, `update`, `version` |
| 1.4 | `oa init --help` | Shows `--spec`, `--output`, `--template`, `--dry-run`, `--verbose` |
| 1.5 | `oa run --help` | Shows `--spec`, `--task`, `--input`, `--quiet`, `--verbose` |
| 1.6 | `oa update --help` | Shows `--spec`, `--output`, `--dry-run`, `--verbose` |

---

## 2. `oa init` without args (error path)

| # | Command | What to check |
|---|---------|----------------|
| 2.1 | `oa init` (no options) | Exits non-zero; prints hint to use `--output` and mentions `oa init aac` |

---

## 3. `oa init aac` (agent-as-code layout)

| # | Command | What to check |
|---|---------|----------------|
| 3.1 | `cd /tmp && mkdir -p oa-test && cd oa-test` | Clean dir |
| 3.2 | `oa init aac` | Creates `.agents/example.yaml`, `.agents/review.yaml`, and `.agents/README.md` |
| 3.3 | `test -f .agents/example.yaml && test -f .agents/review.yaml` | Both files exist |
| 3.4 | `oa init aac` again (no force) | Exits non-zero; says file exists; suggests `--force` |
| 3.5 | `oa init aac --force` | Overwrites; still succeeds |
| 3.6 | `oa init aac -q` | Quiet: prints only the paths to `example.yaml` and `review.yaml` |

---

## 4. `oa run` (spec → model, no codegen)

**Needs a valid API key** for the engine in the spec (e.g. `export OPENAI_API_KEY=...`).

| # | Command | What to check |
|---|---------|----------------|
| 4.1 | After `oa init aac`, `oa run --spec .agents/example.yaml --task greet --input '{"name": "Tester"}' --quiet` | Exit 0; **stdout** is task output JSON only (no banner), suitable for piping to `jq` |
| 4.2 | Same without `--quiet` | Banner + pretty JSON (Rich) |
| 4.3 | `oa run --spec .agents/example.yaml --input '{"name": "X"}'` | Omitting `--task` uses default task; still runs or clear error |
| 4.4 | `oa run --spec missing.yaml` | Non-zero; error on stderr |
| 4.5 | `oa run --spec .agents/example.yaml --input 'not-json'` | Non-zero; bad parameter message |

---

## 5. `oa init --spec … --output …` (full scaffold)

| # | Command | What to check |
|---|---------|----------------|
| 5.1 | Use path to bundled minimal spec, e.g. from repo:  
`oa init --spec <path-to>/oas_cli/templates/minimal-agent.yaml --output /tmp/oa-agent` | Success message; no traceback |
| 5.2 | `ls /tmp/oa-agent` | Contains `agent.py`, `requirements.txt`, `.env.example`, `README.md`, `prompts/` |
| 5.3 | `ls /tmp/oa-agent/prompts` | At least `agent_prompt.jinja2`; task prompts if spec has tasks |
| 5.4 | `oa init --spec <same-spec> --output /tmp/oa-agent --dry-run` | No files changed; lists what would be created |
| 5.5 | `oa init --spec /nonexistent.yaml --output /tmp/x` | Non-zero; validation/path error |
| 5.6 | `oa init --spec <valid.yaml> --output /tmp/oa-agent --verbose` | More log output; still succeeds |

---

## 7. `oa update`

| # | Command | What to check |
|---|---------|----------------|
| 7.1 | `oa update --spec <same-spec> --output /tmp/oa-agent --dry-run` | No file changes; lists files that would update |
| 7.2 | `oa update --spec <same-spec> --output /tmp/oa-agent` | Success; regenerates files |
| 7.3 | `oa update --spec x.yaml --output /nonexistent` | Non-zero; message says run `oa init` first |

---

## 8. Regression: old command name

| # | Command | What to check |
|---|---------|----------------|
| 8.1 | `oas --version` (if typed by habit) | **Should fail** (command not found) unless you kept an alias — confirms migration to `oa` |

---

## 9. Optional: generated project smoke test

| # | Command | What to check |
|---|---------|----------------|
| 9.1 | `cd /tmp/oa-agent && cp .env.example .env` | Add a real API key in `.env` |
| 9.2 | `pip install -r requirements.txt` | Installs without error |
| 9.3 | `python -c "import agent"` or run `python agent.py` per generated README | No import error (runtime may need network for full run) |

---

## Pass criteria

- **§1** help/version all succeed; version reads **1.2.8**.
- **§3** `oa init aac` creates `.agents/` as documented.
- **§5–6** init produces expected tree; `--dry-run` does not write.
- **§7** update works on existing dir; fails cleanly on missing dir.
- **§4** if API key set, `oa run --quiet` gives JSON on stdout only.

---

## Quick copy-paste block (after install)

```bash
oa --version
oa --help
oa init || true
cd /tmp && rm -rf oa-smoke && mkdir oa-smoke && cd oa-smoke
oa init aac
oa run --spec .agents/example.yaml --task greet --input '{"name": "QA"}' --quiet
```

If `oa run` fails without a key, that’s expected; still check exit code and stderr message are sensible.

---

## Reference

- [CHANGELOG.md](../CHANGELOG.md) — breaking changes (e.g. `oas` → `oa`)
- [README.md](../README.md) — user-facing commands
- [REFERENCE.md](REFERENCE.md) — spec shape and engines


# Workflows

| Workflow   | Trigger              | Purpose |
|-----------|----------------------|--------|
| **ci.yml** | Push / PR to `main`, `develop` | Run tests (pytest), Ruff check/format, mypy, integration tests; upload artifacts (results, coverage). |
| **publish.yml** | Push of tag `v*` (e.g. `v1.0.9`) | Verify tag matches `pyproject.toml` version, build package, publish to PyPI. |

Run tests locally: `pytest tests/` (see repo root README and CONTRIBUTING).

# CI/CD Workflows

## Main CI (single job)

| Workflow | Trigger | Purpose |
|----------|---------|---------|
| **ci.yml** | PRs and pushes to `main` / `develop` | One job: Python 3.12, pytest + ruff (check + format) + mypy + integration tests (`test_templates.py`). Single green check per run. |

All testing and linting for PRs and main/develop runs in this one workflow. No matrix, no separate “Enhanced” or “PR Tests” jobs.

## Other workflows

| Workflow | Trigger | Purpose |
|----------|---------|---------|
| **publish.yml** | Release | Publish package. |
| **ci-error-analysis.yaml** | — | Error analysis (if used). |

## Removed (consolidated into ci.yml)

- **pr-test.yml** — merged into ci.yml
- **test-enhanced.yml** — matrix and reporting removed; single Python 3.12 in ci.yml
- **test.yml** (Push Tests) — same steps now in ci.yml on push
- **integration-tests.yml** — integration step runs inside ci.yml
- **feature-test.yml** — removed; open a PR to run CI on feature branches

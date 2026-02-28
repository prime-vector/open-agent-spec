# CI/CD Workflows

| Workflow | Trigger | Purpose |
|----------|---------|---------|
| **pr-test.yml** | PRs to `main`/`develop` | Fast feedback: pytest (single version), ruff, mypy. One run per branch (new pushes cancel previous). |
| **test-enhanced.yml** | Push/PR to `main` | Full test matrix (Python 3.10 + 3.12), coverage, JUnit/HTML reports, Codecov, PR comment with results. |
| **test.yml** | Push to `main`/`develop` | Push Tests: pytest, ruff, mypy (single Python 3.12). |
| **integration-tests.yml** | Push/PR to `main`, manual | Runs `tests/integration/test_templates.py` (scaffold from templates, install deps, import/run agents). |
| **feature-test.yml** | Feature branches | Optional feature-branch testing. |
| **publish.yml** | Release/publish | Package publish. |
| **ci-error-analysis.yaml** | — | Error analysis. |

**Python versions:** The project supports **Python ≥3.10**. We run the main test suite on **3.10 and 3.12** only (3.11 skipped to keep CI faster and reduce redundant runs). If you need to support 3.11 explicitly, add it back to the matrix in `test-enhanced.yml`.

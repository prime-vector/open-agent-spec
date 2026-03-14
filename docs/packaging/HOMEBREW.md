# Installing Open Agent Spec (OA) CLI via Homebrew

The **open-agent-spec** package is the **Open Agent Spec (OA)** Python CLI (`oa`). Homebrew can expose it in several ways.

---

## 0. **prime-vector tap** (official tap install)

If the formula is published in the **prime-vector** tap:

```bash
brew tap prime-vector/homebrew-prime-vector
brew install open-agent-spec
oa --version
```

---

## 1. **pipx** (recommended when not using the tap)

Homebrew ships **pipx**; pipx isolates the CLI in its own venv (like `brew` for Python tools).

```bash
brew install pipx
pipx ensurepath
pipx install open-agent-spec
oa --version
```

**Pros:** Always matches PyPI; no sha256 bumps; no resource blocks.  
**Cons:** Not a single `brew install open-agent-spec` for users who refuse pipx.

---

## 2. **Official Homebrew/core** (best UX, more work)

To get `brew install open-agent-spec`:

1. **Fork** [Homebrew/homebrew-core](https://github.com/Homebrew/homebrew-core).
2. **Create a formula** that uses the PyPI **sdist** URL and sha256 from [pypi.org/project/open-agent-spec/#files](https://pypi.org/project/open-agent-spec/#files).
3. **Generate Python resources** (required — Brew won’t run `pip install` online during install):
   ```bash
   brew create --set-name open-agent-spec https://files.pythonhosted.org/packages/source/o/open-agent-spec/open_agent_spec-1.2.5.tar.gz
   # Then in the formula directory:
   brew update-python-resources open-agent-spec
   ```
4. **Test** locally: `brew install --build-from-source ./open-agent-spec.rb` and `oa --version`.
5. **Open a PR** to homebrew-core; follow their Python app guidelines:  
   https://docs.brew.sh/Python-for-Formula-Authors

After merge, users run:

```bash
brew install open-agent-spec
oa --version
```

---

## 3. **Your own tap** (middle ground)

If you don’t want to wait for core:

1. Create a repo `homebrew-open-agent-spec` (or `homebrew-tap`) with:
   ```
   Formula/open-agent-spec.rb
   ```
2. Use the same approach as core: **sdist URL + `brew update-python-resources`** to fill `resource` blocks, or maintain them by hand after each release.
3. Users tap and install:
   ```bash
   brew tap YOUR_GITHUB_USER/open-agent-spec
   brew install open-agent-spec
   ```

Each **PyPI release** requires updating the formula **url**, **sha256**, and usually re-running **`brew update-python-resources`** so dependency resources stay in sync.

---

## NPM

**No** — publishing to **npm** would **not** reuse the Python wheel. You’d need either:

- A **small Node wrapper** that shells out to `oa` (still requires Python/pipx/Homebrew on the machine), or  
- A **full JS/TS reimplementation** of the CLI (large effort).

So: **Homebrew = package the Python app** (via pipx or a resource-heavy formula). **NPM = separate project** unless you only publish a thin installer script.

---

## Summary

| Method              | Command                          | Maintenance      |
|---------------------|----------------------------------|------------------|
| prime-vector tap    | `brew tap prime-vector/homebrew-prime-vector && brew install open-agent-spec` | Tap maintainer   |
| pipx                | `pipx install open-agent-spec`   | None             |
| Homebrew/core       | `brew install open-agent-spec`   | PR + resources   |
| Custom tap          | `brew tap … && brew install …`  | Per-release bump |

For a pre-launch launch, **document pipx** in the README; add **Homebrew/core** when you’re ready to maintain the formula or get a contributor to run `update-python-resources` each release.

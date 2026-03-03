# Open Agent Spec — Website & Playground

A modern `.dev` site and **interactive playground** for [Open Agent Spec](https://openagents.org): a declarative standard for defining AI agents.

## What’s in here

- **Split-screen playground**: YAML spec editor (left) and result tabs (right): **Generated Code** | **Logs** | **Output**.
- **Layers** (no demo logic in UI):
  - **Spec**: YAML parsing + validation (JSON schema).
  - **Codegen**: Python/TypeScript scaffold from validated spec.
  - **Runtime**: Mock execution pipeline (prompt build, memory, tools, response) with structured logs.
- **UX**: Minimal, calm UI; subtle indicators for engine, memory, and tool usage.

## Tech stack

- **Next.js** (App Router), **TypeScript**, **Tailwind**
- **Monaco Editor** for YAML
- **ajv** + bundled Open Agent Spec JSON schema for validation
- Mock runtime only (no real LLM calls)

## Develop

```bash
cd Website
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000).

## Scripts

- `npm run dev` — dev server
- `npm run build` — production build
- `npm run start` — run production build
- `npm run lint` — ESLint

## Project layout

```
Website/
├── app/                    # Next App Router
│   ├── layout.tsx
│   ├── page.tsx            # Playground page
│   └── globals.css
├── components/
│   ├── layout/             # SplitScreen, ResultTabs
│   ├── editor/             # YamlEditor (Monaco)
│   ├── playground/         # GenerateButton, RunButton
│   └── output/             # GeneratedCodeTab, LogsTab, OutputTab
├── lib/
│   ├── spec/               # types, schema.ts, validate, parse
│   ├── codegen/             # generate (Python/TS scaffold)
│   └── runtime/             # mockRuntime, types (logs, execution result)
├── content/                # defaultSpec (YAML + TS export)
└── public/
```

## Extending

- **Validation**: Edit `lib/spec/schema.ts` (and/or `validate.ts`) to match the canonical Open Agent Spec schema.
- **Codegen**: The playground uses the **real** Open Agent Spec Python generator when available:
  - **Generate Agent** (Python): `POST /api/generate` with `{ yaml }` runs `Website/scripts/invoke_generator.py`, which calls `oas_cli` to produce full `agent.py`, README, requirements, prompts, etc. Requires the repo root to have Python and `pip install -e .` (or `open-agent-spec`). If the API is unavailable (e.g. deployed without Python), the UI falls back to the in-browser scaffold.
  - Set `PYTHON_PATH` (e.g. to a venv’s `python`) when running the dev server if your default `python3` is not the right one.
- **Try with OpenAI**: **Try with OpenAI** runs the first task once via `POST /api/run-demo` with optional `apiKey`. Rate limit: **1 run per IP per calendar day**. Leave the key blank for a mock result. For production, replace the in-memory rate limit with Redis or Vercel KV.
- **Runtime**: **Run Agent** uses the mock runtime (no LLM). Replace or extend `lib/runtime/mockRuntime.ts` for real inference.

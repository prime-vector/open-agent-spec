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
- **Codegen**:
  - In **production on Vercel**, the playground uses a **Python Vercel Function**: `POST /api/generate` with `{ yaml }` proxies to `api/cli_generate.py`, which imports `open-agent-spec` from PyPI and produces full `agent.py`, README, requirements, prompts, etc. If that function fails, the UI falls back to the in-browser scaffold.
  - In **local dev**, `npm run dev` does not run the Python function; `/api/generate` will return a fallback error and the UI will use the in-browser scaffold by design. To exercise the real generator locally, run with `vercel dev` so the Python function is available.
- **Try with OpenAI**: **Try with OpenAI** runs the first task once via `POST /api/run-demo` with optional `apiKey`, executing **directly from the Open Agent Spec YAML** (no generated agent code required). Rate limit: **1 run per IP per calendar day** (can be temporarily disabled for demos with `DISABLE_DEMO_RATE_LIMIT=1`). Leave the key blank for a mock result. For production, replace the in-memory rate limit with Redis or Vercel KV.
- **Runtime**: **Run Agent** uses the mock runtime (no LLM). Replace or extend `lib/runtime/mockRuntime.ts` for real inference.


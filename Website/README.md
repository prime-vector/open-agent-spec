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
- **Codegen**: Adjust `lib/codegen/generate.ts` to match the real CLI output (or call a backend).
- **Runtime**: Replace or extend `lib/runtime/mockRuntime.ts` with a real inference engine or proxy.

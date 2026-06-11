#!/usr/bin/env node
// Conformance adapter for the npm runtime (@prime-vector/open-agent-spec).
//
// Speaks the OA Conformance Adapter Protocol v1 (see ../../PROTOCOL.md):
//
//     adapter.mjs --capabilities     → capability manifest on stdout
//     adapter.mjs  (case on stdin)   → result JSON on stdout
//
// Requires the npm package to be built first (cd npm && npm ci && npm run build).
// Mocking strategy: injects a canned invokeFn via runTaskFromSpec options.

import { readFileSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const PROTOCOL_VERSION = 1;
const HERE = dirname(fileURLToPath(import.meta.url));
const NPM_DIST = join(HERE, "..", "..", "..", "..", "npm", "dist");

async function loadRuntime() {
  try {
    return await import(pathToFileURL(join(NPM_DIST, "index.js")).href);
  } catch (err) {
    process.stderr.write(
      `Cannot load npm runtime from ${NPM_DIST} — run 'cd npm && npm ci && npm run build' first.\n${err}\n`,
    );
    process.exit(3);
  }
}

function capabilities() {
  let version = "dev";
  try {
    const pkg = JSON.parse(
      readFileSync(join(HERE, "..", "..", "..", "..", "npm", "package.json"), "utf8"),
    );
    version = pkg.version;
  } catch {
    /* keep dev */
  }
  return {
    protocol: PROTOCOL_VERSION,
    runtime: "npm",
    version,
    capabilities: [
      "core",
      "depends-on",
      "delegation",
      "registry",
      "response-format-text",
      "history",
    ],
  };
}

// Mock matching rule per PROTOCOL.md: task name in rendered prompt wins,
// otherwise responses are consumed in declaration order; each at most once.
function buildMock(mockResponses) {
  const remaining = new Map(Object.entries(mockResponses ?? {}));
  const order = Object.keys(mockResponses ?? {});
  let orderIdx = 0;

  return async (system, user) => {
    for (const taskName of remaining.keys()) {
      if (user.includes(taskName) || system.includes(taskName)) {
        const response = remaining.get(taskName);
        remaining.delete(taskName);
        return response;
      }
    }
    while (orderIdx < order.length) {
      const key = order[orderIdx++];
      if (remaining.has(key)) {
        const response = remaining.get(key);
        remaining.delete(key);
        return response;
      }
    }
    return "{}";
  };
}

async function runCase(caseData, runtime) {
  const { parseSpec, runTaskFromSpec, OAError } = runtime;
  const invoke = caseData.invoke ?? {};
  const filesDir = caseData.files_dir ?? null;

  try {
    const specPath = filesDir ? join(filesDir, "main.yaml") : null;
    const spec = parseSpec(caseData.spec, specPath ?? "<case>");

    const result = await runTaskFromSpec(
      spec,
      invoke.task ?? undefined,
      invoke.input ?? {},
      specPath,
      {
        systemPromptOverride: invoke.override_system ?? null,
        userPromptOverride: invoke.override_user ?? null,
        invokeFn: buildMock(caseData.mock_responses),
      },
    );
    return { protocol: PROTOCOL_VERSION, status: "ok", result };
  } catch (err) {
    if (err instanceof OAError) {
      return { protocol: PROTOCOL_VERSION, status: "error", error: err.toJSON() };
    }
    return {
      protocol: PROTOCOL_VERSION,
      status: "error",
      error: { error: String(err?.message ?? err), code: "RUN_ERROR", stage: "run" },
    };
  }
}

async function main() {
  if (process.argv.includes("--capabilities")) {
    process.stdout.write(JSON.stringify(capabilities()) + "\n");
    return;
  }

  const runtime = await loadRuntime();
  const chunks = [];
  for await (const chunk of process.stdin) chunks.push(chunk);
  const caseData = JSON.parse(Buffer.concat(chunks).toString("utf8"));
  const response = await runCase(caseData, runtime);
  process.stdout.write(JSON.stringify(response) + "\n");
}

main();

import { resolve, dirname } from "node:path";
import { loadSpecFromFile, chooseTask, OAError } from "./loader.js";
import { resolvePrompts } from "./prompts.js";
import { invokeProvider } from "./providers/index.js";
import { resolveSpecUrl, isRemoteRef, fetchRemoteSpec } from "./registry.js";
import type {
  OASpec,
  RunInput,
  TaskResult,
  DelegatedTask,
  InlineTask,
  InvokeFn,
  ChatMessage,
} from "./types.js";

// ── JSON extraction ────────────────────────────────────────────────────────

function extractJson(raw: string): unknown {
  const trimmed = raw.trim();

  // Try direct parse first.
  try {
    const parsed = JSON.parse(trimmed);
    if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
      return parsed;
    }
  } catch {
    /* fall through */
  }

  // Strip markdown code fences: ```json ... ``` or ``` ... ```
  const fenced = trimmed.match(/```(?:json)?\s*([\s\S]+?)```/i);
  if (fenced?.[1]) {
    try {
      const parsed = JSON.parse(fenced[1].trim());
      if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
        return parsed;
      }
    } catch {
      /* fall through */
    }
  }

  // Last resort: find the first {...} block.
  const braceStart = trimmed.indexOf("{");
  const braceEnd = trimmed.lastIndexOf("}");
  if (braceStart !== -1 && braceEnd > braceStart) {
    try {
      const parsed = JSON.parse(trimmed.slice(braceStart, braceEnd + 1));
      if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
        return parsed;
      }
    } catch {
      /* fall through */
    }
  }

  // Mirror the reference runtime: unparseable output is returned raw.
  return raw;
}

// ── Execution context ──────────────────────────────────────────────────────

interface ExecContext {
  invoke: InvokeFn;
  overrideSystem?: string | null;
  overrideUser?: string | null;
}

// ── Single task execution ──────────────────────────────────────────────────

async function runSingleTask(
  spec: OASpec,
  taskName: string,
  input: RunInput,
  specPath: string | null,
  visitedSpecs: ReadonlySet<string>,
  ctx: ExecContext,
): Promise<TaskResult> {
  // Immutable input snapshot — every task boundary gets its own copy.
  input = structuredClone(input);

  const taskDef = spec.tasks[taskName];
  if (!taskDef) {
    const available = Object.keys(spec.tasks).join(", ");
    throw new OAError(
      `Task '${taskName}' not found. Available: ${available}`,
      "TASK_NOT_FOUND",
      "routing",
    );
  }

  // ── Delegation ────────────────────────────────────────────────────────
  const delegated = taskDef as DelegatedTask;
  if (delegated.spec) {
    const rawRef = delegated.spec.trim();
    let canonicalKey: string;
    let delegatedSpec: OASpec;
    let nextSpecPath: string | null;

    if (isRemoteRef(rawRef)) {
      const url = resolveSpecUrl(rawRef);
      canonicalKey = url;

      if (visitedSpecs.has(canonicalKey)) {
        throw new OAError(
          `Circular spec delegation detected: '${url}' is already in the delegation stack`,
          "DELEGATION_CYCLE_ERROR",
          "delegation",
          taskName,
        );
      }

      delegatedSpec = await fetchRemoteSpec(url);
      nextSpecPath = null;
    } else {
      // Local path — resolve relative to calling spec's directory.
      const resolved = specPath
        ? resolve(dirname(specPath), rawRef)
        : resolve(rawRef);

      canonicalKey = resolved;

      if (visitedSpecs.has(canonicalKey)) {
        throw new OAError(
          `Circular spec delegation detected: '${resolved}' is already in the delegation stack`,
          "DELEGATION_CYCLE_ERROR",
          "delegation",
          taskName,
        );
      }

      delegatedSpec = loadSpecFromFile(resolved);
      nextSpecPath = resolved;
    }

    const newVisited = new Set(visitedSpecs).add(canonicalKey);
    const delegatedTaskName = delegated.task ?? taskName;

    if (!(delegatedTaskName in delegatedSpec.tasks)) {
      const available = Object.keys(delegatedSpec.tasks).join(", ");
      throw new OAError(
        `Task '${delegatedTaskName}' not found in delegated spec '${canonicalKey}'. Available: ${available}`,
        "TASK_NOT_FOUND",
        "delegation",
        taskName,
      );
    }

    const result = await runSingleTask(
      delegatedSpec,
      delegatedTaskName,
      input,
      nextSpecPath,
      newVisited,
      ctx,
    );

    return {
      ...result,
      task: taskName,
      delegated_to: `${canonicalKey}#${delegatedTaskName}`,
    };
  }

  // ── Inline task ──────────────────────────────────────────────────────
  const inline = taskDef as InlineTask;

  // Validate required inputs before touching the model.
  const requiredFields = inline.input?.required ?? [];
  const missing = requiredFields.filter((f) => !(f in input));
  if (missing.length > 0) {
    throw new OAError(
      `Missing required input field(s) for task '${taskName}': ${missing.join(", ")}`,
      "CHAIN_INPUT_MISSING",
      "input_validation",
      taskName,
    );
  }

  const prompts = resolvePrompts(spec, taskName, taskDef, input, {
    system: ctx.overrideSystem,
    user: ctx.overrideUser,
  });

  const providerConfig = {
    engine: spec.intelligence.engine,
    model: spec.intelligence.model,
    config: spec.intelligence.config,
  };

  // history is a reserved input convention — never stored by OA, just forwarded.
  const history = Array.isArray(input["history"])
    ? (input["history"] as ChatMessage[])
    : undefined;

  const raw = await ctx.invoke(prompts.system, prompts.user, providerConfig, history);

  // response_format: text → raw passthrough, no JSON parsing.
  const output = inline.response_format === "text" ? raw : extractJson(raw);

  return {
    task: taskName,
    input,
    prompt: `${prompts.system}\n\n${prompts.user}`.trim(),
    engine: spec.intelligence.engine,
    model: spec.intelligence.model,
    raw_output: raw,
    output,
  };
}

// ── Dependency chain resolution ────────────────────────────────────────────

/**
 * Transitive cycle check over the depends_on graph (spec §7.2).
 * Mirrors the reference runtime: detection is transitive even though
 * execution only runs *direct* dependencies.
 */
function checkChainCycles(spec: OASpec, taskName: string): void {
  const seen = new Set<string>([taskName]);

  const walk = (name: string): void => {
    const deps = (spec.tasks[name] as { depends_on?: string[] })?.depends_on ?? [];
    for (const dep of deps) {
      if (seen.has(dep)) {
        throw new OAError(
          `Circular dependency detected: '${dep}' is already in the chain`,
          "CHAIN_CYCLE_ERROR",
          "routing",
          name,
        );
      }
      seen.add(dep);
      walk(dep);
    }
  };

  walk(taskName);
}

/**
 * Resolve the depends_on chain for a task (spec §7.2).
 *
 * Direct dependencies run in listed order; each dependency's output is merged
 * into the accumulating input (later entries win on key collision). Returns
 * the merged input plus the chain of intermediate result envelopes.
 */
async function resolveChain(
  spec: OASpec,
  taskName: string,
  baseInput: RunInput,
  specPath: string | null,
  ctx: ExecContext,
): Promise<{ merged: RunInput; chain: Record<string, TaskResult> }> {
  const taskDef = spec.tasks[taskName];
  const deps: string[] =
    (taskDef as { depends_on?: string[] } | undefined)?.depends_on ?? [];

  if (deps.length === 0) {
    return { merged: structuredClone(baseInput), chain: {} };
  }

  checkChainCycles(spec, taskName);

  const merged: RunInput = structuredClone(baseInput);
  const chain: Record<string, TaskResult> = {};

  for (const dep of deps) {
    if (!(dep in spec.tasks)) {
      throw new OAError(
        `depends_on references unknown task '${dep}'`,
        "TASK_NOT_FOUND",
        "routing",
        dep,
      );
    }
    const depResult = await runSingleTask(
      spec,
      dep,
      structuredClone(merged),
      specPath,
      new Set(),
      // Dependencies never receive caller prompt overrides.
      { invoke: ctx.invoke },
    );
    chain[dep] = depResult;
    if (depResult.output && typeof depResult.output === "object" && !Array.isArray(depResult.output)) {
      Object.assign(merged, depResult.output);
    }
  }

  return { merged, chain };
}

// ── Public API ─────────────────────────────────────────────────────────────

export interface RunOptions {
  specPath: string;
  taskName?: string;
  input?: RunInput;
  systemPromptOverride?: string | null;
  userPromptOverride?: string | null;
  /** Custom provider invocation — for embedding, testing, and conformance. */
  invokeFn?: InvokeFn;
}

/**
 * Run a task from a spec file path.
 *
 * @example
 * const result = await runTask({
 *   specPath: "./agents/summariser.yaml",
 *   taskName: "summarise",
 *   input: { text: "..." },
 * });
 */
export async function runTask(options: RunOptions): Promise<TaskResult> {
  const { specPath, input = {} } = options;
  const spec = loadSpecFromFile(specPath);
  return runTaskFromSpec(spec, options.taskName, input, specPath, {
    systemPromptOverride: options.systemPromptOverride,
    userPromptOverride: options.userPromptOverride,
    invokeFn: options.invokeFn,
  });
}

export interface RunFromSpecOptions {
  systemPromptOverride?: string | null;
  userPromptOverride?: string | null;
  invokeFn?: InvokeFn;
}

/**
 * Run a task from a pre-parsed spec object (useful for testing and embedding).
 */
export async function runTaskFromSpec(
  spec: OASpec,
  taskName?: string,
  input: RunInput = {},
  specPath: string | null = null,
  options: RunFromSpecOptions = {},
): Promise<TaskResult> {
  const [resolvedTaskName] = chooseTask(spec, taskName);

  const ctx: ExecContext = {
    invoke: options.invokeFn ?? invokeProvider,
    overrideSystem: options.systemPromptOverride,
    overrideUser: options.userPromptOverride,
  };

  // Deep copy at the public entry point so the caller's object is never mutated.
  const baseInput = structuredClone(input);

  const { merged, chain } = await resolveChain(
    spec,
    resolvedTaskName,
    baseInput,
    specPath,
    ctx,
  );

  const visited = new Set<string>();
  if (specPath) visited.add(resolve(specPath));

  const result = await runSingleTask(
    spec,
    resolvedTaskName,
    merged,
    specPath,
    visited,
    ctx,
  );

  if (Object.keys(chain).length > 0) {
    result.chain = chain;
  }

  return result;
}

import { resolve, dirname } from "node:path";
import { loadSpecFromFile, parseSpec, chooseTask, OAError } from "./loader.js";
import { resolvePrompts } from "./prompts.js";
import { invokeProvider } from "./providers/index.js";
import { resolveSpecUrl, isRemoteRef, fetchRemoteSpec } from "./registry.js";
import type { OASpec, RunInput, TaskResult, DelegatedTask } from "./types.js";

// ── JSON extraction ────────────────────────────────────────────────────────

function extractJson(raw: string): Record<string, unknown> {
  const trimmed = raw.trim();

  // Try direct parse first.
  try {
    const parsed = JSON.parse(trimmed);
    if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
      return parsed as Record<string, unknown>;
    }
  } catch {
    /* fall through */
  }

  // Strip markdown code fences: ```json ... ``` or ``` ... ```
  const fenced = trimmed.match(/```(?:json)?\s*([\s\S]+?)```/);
  if (fenced?.[1]) {
    try {
      const parsed = JSON.parse(fenced[1].trim());
      if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
        return parsed as Record<string, unknown>;
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
        return parsed as Record<string, unknown>;
      }
    } catch {
      /* fall through */
    }
  }

  throw new OAError(
    `Model response did not contain a valid JSON object: ${trimmed.slice(0, 200)}`,
    "OUTPUT_PARSE_ERROR",
    "output_parsing",
  );
}

// ── Single task execution ──────────────────────────────────────────────────

async function runSingleTask(
  spec: OASpec,
  taskName: string,
  input: RunInput,
  specPath: string | null,
  visitedSpecs: ReadonlySet<string>,
): Promise<TaskResult> {
  const taskDef = spec.tasks[taskName];
  if (!taskDef) {
    const available = Object.keys(spec.tasks).join(", ");
    throw new OAError(
      `Task '${taskName}' not found. Available: ${available}`,
      "TASK_NOT_FOUND",
      "task_selection",
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
    );

    return {
      ...result,
      task: taskName,
      delegated_to: `${canonicalKey}#${delegatedTaskName}`,
    };
  }

  // ── Inline task ──────────────────────────────────────────────────────
  const prompts = resolvePrompts(spec, taskName, taskDef, input);

  const providerConfig = {
    engine: spec.intelligence.engine,
    model: spec.intelligence.model,
    config: spec.intelligence.config,
  };

  const raw = await invokeProvider(prompts.system, prompts.user, providerConfig);
  const output = extractJson(raw);

  return {
    task: taskName,
    output,
    provider: spec.intelligence.engine,
    model: spec.intelligence.model,
  };
}

// ── Dependency chain resolution ────────────────────────────────────────────

async function resolveChain(
  spec: OASpec,
  taskName: string,
  baseInput: RunInput,
  specPath: string | null,
): Promise<TaskResult> {
  const taskDef = spec.tasks[taskName];
  if (!taskDef) {
    throw new OAError(
      `Task '${taskName}' not found`,
      "TASK_NOT_FOUND",
      "task_selection",
    );
  }

  const dependsOn: string[] = (taskDef as { depends_on?: string[] }).depends_on ?? [];
  const mergedInput: RunInput = { ...baseInput };

  // Resolve each upstream dependency first (sequential — preserves data contract).
  for (const dep of dependsOn) {
    const depResult = await resolveChain(spec, dep, baseInput, specPath);
    // Merge upstream output fields into this task's input.
    Object.assign(mergedInput, depResult.output);
  }

  return runSingleTask(spec, taskName, mergedInput, specPath, new Set());
}

// ── Public API ─────────────────────────────────────────────────────────────

export interface RunOptions {
  specPath: string;
  taskName?: string;
  input?: RunInput;
  systemPromptOverride?: string;
  userPromptOverride?: string;
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
  const [resolvedTaskName] = chooseTask(spec, options.taskName);
  return resolveChain(spec, resolvedTaskName, input, specPath);
}

/**
 * Run a task from a pre-parsed spec object (useful for testing and embedding).
 */
export async function runTaskFromSpec(
  spec: OASpec,
  taskName: string,
  input: RunInput = {},
  specPath: string | null = null,
): Promise<TaskResult> {
  const [resolvedTaskName] = chooseTask(spec, taskName);
  return resolveChain(spec, resolvedTaskName, input, specPath);
}

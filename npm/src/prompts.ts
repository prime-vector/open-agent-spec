import { OAError } from "./loader.js";
import type { OASpec, TaskDef, TaskPrompts, RunInput } from "./types.js";

/**
 * Resolve `{key}` and `{{ key }}` placeholders in a template string.
 * Missing keys are left as-is (matching Python runner behaviour).
 */
export function interpolate(template: string, vars: RunInput): string {
  return template
    .replace(/\{\{\s*(\w+)\s*\}\}/g, (_, key: string) =>
      key in vars ? String(vars[key]) : `{{ ${key} }}`
    )
    .replace(/\{(\w+)\}/g, (match, key: string) =>
      key in vars ? String(vars[key]) : match
    );
}

export interface ResolvedPrompts {
  system: string;
  user: string;
}

export interface PromptOverrides {
  system?: string | null;
  user?: string | null;
}

/**
 * Layered prompt resolution, matching the reference runtime (spec §9.1).
 *
 * Priority (highest → lowest):
 *   1. CLI / caller overrides
 *   2. Per-task inline prompts   tasks.<name>.prompts   (Style A)
 *   3. Legacy per-task map       prompts.<name>.system  (Style B)
 *   4. Global fallback           prompts.system / prompts.user
 */
export function resolvePrompts(
  spec: OASpec,
  taskName: string,
  taskDef: TaskDef,
  input: RunInput,
  overrides: PromptOverrides = {},
): ResolvedPrompts {
  const globalPrompts = spec.prompts ?? {};
  const globalSystem = typeof globalPrompts.system === "string" ? globalPrompts.system : "";
  const globalUser = typeof globalPrompts.user === "string" ? globalPrompts.user : "";

  // Style B — legacy task-keyed map under the global prompts block.
  const legacyEntry = globalPrompts[taskName];
  const legacy: TaskPrompts =
    legacyEntry && typeof legacyEntry === "object" ? legacyEntry : {};

  // Style A — prompts co-located inside the task definition.
  const inlineTask = taskDef as { prompts?: TaskPrompts };
  const inline: TaskPrompts = inlineTask.prompts ?? {};

  let rawSystem: string;
  if (overrides.system != null) {
    rawSystem = overrides.system;
  } else if (inline.system) {
    rawSystem = inline.system;
  } else if (legacy.system) {
    rawSystem = legacy.system;
  } else {
    rawSystem = globalSystem;
  }

  let rawUser: string;
  if (overrides.user != null) {
    rawUser = overrides.user;
  } else if (inline.user) {
    rawUser = inline.user;
  } else if (legacy.user) {
    rawUser = legacy.user;
  } else {
    rawUser = globalUser;
  }

  if (!rawUser) {
    throw new OAError(
      `Task '${taskName}' has no user prompt (set prompts.user at task or spec level)`,
      "PROMPT_RESOLUTION_ERROR",
      "prompt_resolution",
      taskName,
    );
  }

  return {
    system: interpolate(rawSystem, input),
    user: interpolate(rawUser, input),
  };
}

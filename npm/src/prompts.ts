import type { OASpec, TaskDef, RunInput } from "./types.js";

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

export function resolvePrompts(
  spec: OASpec,
  taskName: string,
  taskDef: TaskDef,
  input: RunInput,
): ResolvedPrompts {
  // Task-level prompts take precedence over global spec-level prompts.
  const globalSystem = spec.prompts?.system ?? "";
  const globalUser = spec.prompts?.user ?? "";

  const inlineTask = taskDef as { prompts?: { system?: string; user?: string } };
  const taskSystem = inlineTask.prompts?.system ?? "";
  const taskUser = inlineTask.prompts?.user ?? "";

  const rawSystem = taskSystem || globalSystem;
  const rawUser = taskUser || globalUser;

  if (!rawUser) {
    throw new Error(
      `Task '${taskName}' has no user prompt (set prompts.user at task or spec level)`,
    );
  }

  return {
    system: interpolate(rawSystem, input),
    user: interpolate(rawUser, input),
  };
}

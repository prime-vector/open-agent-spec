import { readFileSync } from "node:fs";
import { load as yamlLoad } from "js-yaml";
import type { OASpec } from "./types.js";

export class OAError extends Error {
  constructor(
    message: string,
    public readonly code: string,
    public readonly stage: string,
    public readonly task?: string,
  ) {
    super(message);
    this.name = "OAError";
  }
}

export function loadSpecFromFile(specPath: string): OASpec {
  let raw: string;
  try {
    raw = readFileSync(specPath, "utf8");
  } catch (err) {
    throw new OAError(
      `Spec file not found: ${specPath}`,
      "SPEC_LOAD_ERROR",
      "load",
    );
  }
  return parseSpec(raw, specPath);
}

export function parseSpec(raw: string, source = "<string>"): OASpec {
  let data: unknown;
  try {
    data = yamlLoad(raw);
  } catch (err) {
    throw new OAError(
      `Invalid YAML in spec ${source}: ${String(err)}`,
      "SPEC_LOAD_ERROR",
      "load",
    );
  }

  if (!data || typeof data !== "object" || Array.isArray(data)) {
    throw new OAError(
      `Spec ${source} must decode to a YAML mapping`,
      "SPEC_LOAD_ERROR",
      "load",
    );
  }

  validateSpec(data as Record<string, unknown>, source);
  return data as OASpec;
}

function validateSpec(data: Record<string, unknown>, source: string): void {
  if (!data["open_agent_spec"]) {
    throw new OAError(
      `${source}: missing required field 'open_agent_spec'`,
      "VALIDATION_ERROR",
      "validate",
    );
  }
  if (!data["agent"] || typeof data["agent"] !== "object") {
    throw new OAError(
      `${source}: missing required field 'agent'`,
      "VALIDATION_ERROR",
      "validate",
    );
  }
  if (!data["intelligence"] || typeof data["intelligence"] !== "object") {
    throw new OAError(
      `${source}: missing required field 'intelligence'`,
      "VALIDATION_ERROR",
      "validate",
    );
  }
  if (!data["tasks"] || typeof data["tasks"] !== "object") {
    throw new OAError(
      `${source}: missing required field 'tasks'`,
      "VALIDATION_ERROR",
      "validate",
    );
  }
  const tasks = data["tasks"] as Record<string, unknown>;
  if (Object.keys(tasks).length === 0) {
    throw new OAError(
      `${source}: 'tasks' must contain at least one task`,
      "VALIDATION_ERROR",
      "validate",
    );
  }
}

export function chooseTask(
  spec: OASpec,
  taskName: string | undefined,
): [string, OASpec["tasks"][string]] {
  const tasks = spec.tasks;
  const names = Object.keys(tasks);

  if (!taskName) {
    if (names.length === 1) return [names[0]!, tasks[names[0]!]!];
    throw new OAError(
      `Spec has multiple tasks — specify --task. Available: ${names.join(", ")}`,
      "TASK_NOT_FOUND",
      "task_selection",
    );
  }

  if (!(taskName in tasks)) {
    throw new OAError(
      `Task '${taskName}' not found. Available: ${names.join(", ")}`,
      "TASK_NOT_FOUND",
      "task_selection",
    );
  }

  return [taskName, tasks[taskName]!];
}

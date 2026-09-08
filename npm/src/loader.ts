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

  toJSON(): Record<string, unknown> {
    const obj: Record<string, unknown> = {
      error: this.message,
      code: this.code,
      stage: this.stage,
    };
    if (this.task !== undefined) obj["task"] = this.task;
    return obj;
  }
}

// Mirrors the engine enum in the canonical JSON schema.
const VALID_ENGINES = new Set([
  "openai",
  "anthropic",
  "grok",
  "xai",
  "cortex",
  "codex",
  "local",
  "custom",
]);

// Mirrors the open_agent_spec version pattern in the canonical JSON schema.
const VERSION_PATTERN = /^(1\.(0\.[4-9]|[1-9]\.[0-9]+)|[2-9]\.[0-9]+\.[0-9]+)$/;

// Spec features this runtime does not implement. Per the conformance honesty
// rule, specs declaring them are REFUSED rather than silently degraded —
// particularly important for sandbox, which is a security feature.
const UNSUPPORTED_ROOT_KEYS = ["tools", "sandbox"] as const;
const UNSUPPORTED_TASK_KEYS = ["tools", "sandbox"] as const;
const ALLOW_DOMAIN_PATTERN = /^(?![A-Za-z][A-Za-z0-9+.-]*:\/\/)(?:\[[^\]]+\]|[^:/\s]+)(?::[0-9]{1,5})?$/;

function validateAllowDomains(sandbox: unknown, source: string): void {
  if (!sandbox || typeof sandbox !== "object" || Array.isArray(sandbox)) return;
  const http = (sandbox as Record<string, unknown>)["http"];
  if (!http || typeof http !== "object" || Array.isArray(http)) return;
  const rules = (http as Record<string, unknown>)["allow_domains"];
  if (rules === undefined) return;
  if (!Array.isArray(rules)) {
    fail(source, "sandbox.http.allow_domains must be an array");
  }
  for (const [index, rule] of rules.entries()) {
    if (typeof rule !== "string" || !ALLOW_DOMAIN_PATTERN.test(rule)) {
      fail(source, `sandbox.http.allow_domains[${index}] must be a host or host:port`);
    }
    const portMatch = rule.match(/:([0-9]{1,5})$/);
    if (portMatch) {
      const port = Number(portMatch[1]);
      if (port < 1 || port > 65535) {
        fail(source, `sandbox.http.allow_domains[${index}] port must be between 1 and 65535`);
      }
    }
  }
}

export function loadSpecFromFile(specPath: string): OASpec {
  let raw: string;
  try {
    raw = readFileSync(specPath, "utf8");
  } catch {
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

function fail(source: string, message: string): never {
  throw new OAError(`${source}: ${message}`, "SPEC_LOAD_ERROR", "load");
}

function validateSpec(data: Record<string, unknown>, source: string): void {
  const version = data["open_agent_spec"];
  if (!version || typeof version !== "string") {
    fail(source, "missing required field 'open_agent_spec'");
  }
  if (!VERSION_PATTERN.test(version)) {
    fail(source, `'open_agent_spec' version '${version}' does not match the required pattern`);
  }

  const agent = data["agent"];
  if (!agent || typeof agent !== "object" || Array.isArray(agent)) {
    fail(source, "missing required field 'agent'");
  }
  const agentObj = agent as Record<string, unknown>;
  if (typeof agentObj["name"] !== "string") {
    fail(source, "'agent.name' is required and must be a string");
  }
  if (typeof agentObj["description"] !== "string") {
    fail(source, "'agent.description' is required and must be a string");
  }

  const intelligence = data["intelligence"];
  if (!intelligence || typeof intelligence !== "object" || Array.isArray(intelligence)) {
    fail(source, "missing required field 'intelligence'");
  }
  const intel = intelligence as Record<string, unknown>;
  if (intel["type"] !== "llm") {
    fail(source, "'intelligence.type' must be 'llm'");
  }
  if (typeof intel["engine"] !== "string" || !VALID_ENGINES.has(intel["engine"])) {
    fail(
      source,
      `'intelligence.engine' must be one of: ${[...VALID_ENGINES].join(", ")}`,
    );
  }
  if (typeof intel["model"] !== "string") {
    fail(source, "'intelligence.model' is required and must be a string");
  }

  const tasks = data["tasks"];
  if (!tasks || typeof tasks !== "object" || Array.isArray(tasks)) {
    fail(source, "missing required field 'tasks'");
  }
  const taskMap = tasks as Record<string, unknown>;
  if (Object.keys(taskMap).length === 0) {
    fail(source, "'tasks' must contain at least one task");
  }

  validateAllowDomains(data["sandbox"], source);
  for (const [taskName, taskDef] of Object.entries(taskMap)) {
    if (taskDef && typeof taskDef === "object") {
      validateAllowDomains(
        (taskDef as Record<string, unknown>)["sandbox"],
        `${source}: task '${taskName}'`,
      );
    }
  }

  if ("behavioural_contract" in data) {
    throw new OAError(
      `${source}: spec declares 'behavioural_contract:' but this runtime cannot enforce contracts.`,
      "CONTRACTS_UNAVAILABLE",
      "contract",
    );
  }
  for (const [taskName, taskDef] of Object.entries(taskMap)) {
    if (
      taskDef &&
      typeof taskDef === "object" &&
      "behavioural_contract" in (taskDef as Record<string, unknown>)
    ) {
      throw new OAError(
        `${source}: task '${taskName}' declares 'behavioural_contract:' but this runtime cannot enforce contracts.`,
        "CONTRACTS_UNAVAILABLE",
        "contract",
        taskName,
      );
    }
  }

  // ── Unsupported feature guard (conformance honesty rule) ───────────────
  for (const key of UNSUPPORTED_ROOT_KEYS) {
    if (key in data) {
      throw new OAError(
        `${source}: spec declares '${key}:' which this runtime does not implement. ` +
          `Refusing to run rather than silently ignoring it. ` +
          `Use the Python reference runtime (pipx install open-agent-spec) for ${key} support.`,
        "UNSUPPORTED_FEATURE",
        "load",
      );
    }
  }
  for (const [taskName, taskDef] of Object.entries(taskMap)) {
    if (!taskDef || typeof taskDef !== "object") continue;
    for (const key of UNSUPPORTED_TASK_KEYS) {
      if (key in (taskDef as Record<string, unknown>)) {
        throw new OAError(
          `${source}: task '${taskName}' declares '${key}:' which this runtime does not implement. ` +
            `Refusing to run rather than silently ignoring it.`,
          "UNSUPPORTED_FEATURE",
          "load",
          taskName,
        );
      }
    }
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
      "routing",
    );
  }

  if (!(taskName in tasks)) {
    throw new OAError(
      `Task '${taskName}' not found. Available: ${names.join(", ")}`,
      "TASK_NOT_FOUND",
      "routing",
    );
  }

  return [taskName, tasks[taskName]!];
}

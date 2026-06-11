// TypeScript types mirroring the Open Agent Spec 1.5.1 YAML schema.

export interface OASpec {
  open_agent_spec: string;
  agent: AgentMeta;
  intelligence: Intelligence;
  prompts?: GlobalPrompts;
  tasks: Record<string, TaskDef>;
}

export interface AgentMeta {
  name: string;
  description: string;
  role?: string;
}

export interface Intelligence {
  type: "llm";
  engine: string;
  model: string;
  config?: Record<string, unknown>;
}

/**
 * Global prompts block. Besides the `system` / `user` fallbacks it may carry
 * legacy per-task prompt maps keyed by task name (Style B):
 *
 *   prompts:
 *     system: "global fallback"
 *     greet:
 *       system: "task-specific"
 */
export interface GlobalPrompts {
  system?: string;
  user?: string;
  [taskName: string]: string | TaskPrompts | undefined;
}

// A task either has inline implementation (prompts) OR delegates to another spec.
export type TaskDef = InlineTask | DelegatedTask;

export interface InlineTask {
  description?: string;
  input?: JsonSchema;
  output?: JsonSchema;
  prompts?: TaskPrompts;
  depends_on?: string[];
  response_format?: "json" | "text";
  // spec / task are absent for inline tasks
  spec?: never;
  task?: never;
}

export interface DelegatedTask {
  description?: string;
  spec: string;       // path, oa://, or https:// reference
  task?: string;      // task name in the delegated spec; defaults to same name
  depends_on?: string[];
  // inline fields absent for delegated tasks
  input?: never;
  output?: never;
  prompts?: never;
  response_format?: never;
}

export interface TaskPrompts {
  system?: string;
  user?: string;
}

export interface JsonSchema {
  type?: string;
  properties?: Record<string, JsonSchema>;
  items?: JsonSchema;
  required?: string[];
  description?: string;
}

// ── Execution types ────────────────────────────────────────────────────────

export interface RunInput {
  [key: string]: unknown;
}

/**
 * The OA result envelope (spec §10). Matches the reference Python runtime
 * field-for-field so results are portable across runtimes.
 */
export interface TaskResult {
  task: string;
  input: RunInput;
  prompt: string;
  engine: string;
  model: string;
  raw_output: string;
  output: unknown;
  chain?: Record<string, TaskResult>;
  delegated_to?: string;
}

// ── Provider types ─────────────────────────────────────────────────────────

export interface ProviderConfig {
  engine: string;
  model: string;
  config?: Record<string, unknown>;
}

export interface ChatMessage {
  role: "system" | "user" | "assistant";
  content: string;
}

/**
 * Signature of the provider invocation function. Injectable via RunOptions
 * for embedding, testing, and conformance certification.
 */
export type InvokeFn = (
  system: string,
  user: string,
  config: ProviderConfig,
  history?: ChatMessage[],
) => Promise<string>;

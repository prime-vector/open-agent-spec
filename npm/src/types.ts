// TypeScript types mirroring the Open Agent Spec 1.5.0 YAML schema.

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
  role: string;
}

export interface Intelligence {
  type: "llm";
  engine: string;
  model: string;
  config?: Record<string, unknown>;
}

export interface GlobalPrompts {
  system?: string;
  user?: string;
}

// A task either has inline implementation (prompts) OR delegates to another spec.
export type TaskDef = InlineTask | DelegatedTask;

export interface InlineTask {
  description?: string;
  input?: JsonSchema;
  output?: JsonSchema;
  prompts?: TaskPrompts;
  depends_on?: string[];
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

export interface TaskResult {
  task: string;
  output: Record<string, unknown>;
  delegated_to?: string;
  provider?: string;
  model?: string;
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

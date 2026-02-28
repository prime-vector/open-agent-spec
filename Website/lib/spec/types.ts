/**
 * Open Agent Spec — type definitions.
 * Mirrors the canonical schema; used by validation and runtime.
 */

export interface AgentInfo {
  name: string;
  description: string;
  role?: string;
}

export interface IntelligenceConfig {
  temperature?: number;
  max_tokens?: number;
  top_p?: number;
  [key: string]: unknown;
}

export interface Intelligence {
  type: string;
  engine: string;
  model: string;
  endpoint?: string;
  config?: IntelligenceConfig;
}

export interface TaskDef {
  description: string;
  timeout?: number;
  input?: { type?: string; properties?: Record<string, unknown>; required?: string[] };
  output: { type?: string; properties?: Record<string, unknown>; required?: string[] };
  tool?: string;
  multi_step?: boolean;
  steps?: Array<{ task: string; input_map?: Record<string, string> }>;
}

export interface Prompts {
  system: string;
  user: string;
  [taskName: string]: unknown;
}

export interface ToolDef {
  id: string;
  description: string;
  type: string;
  allowed_paths?: string[];
}

export interface IntegrationMemory {
  enabled?: boolean;
  [key: string]: unknown;
}

export interface OpenAgentSpec {
  open_agent_spec: string;
  agent: AgentInfo;
  intelligence: Intelligence;
  tasks: Record<string, TaskDef>;
  prompts: Prompts;
  tools?: ToolDef[];
  logging?: {
    enabled?: boolean;
    level?: string;
    [key: string]: unknown;
  };
  behavioural_contract?: Record<string, unknown>;
  integration?: { memory?: IntegrationMemory; [key: string]: unknown };
}

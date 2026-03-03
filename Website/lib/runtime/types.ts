/**
 * Execution simulation — structured log and result types.
 * Consumed by the Logs and Output tabs.
 */

export type LogKind =
  | "prompt"
  | "memory"
  | "tool"
  | "engine"
  | "response"
  | "step"
  | "info"
  | "error";

export interface LogEntry {
  kind: LogKind;
  message: string;
  timestamp: string; // ISO
  detail?: Record<string, unknown>;
}

export interface ToolCallSimulation {
  toolId: string;
  args: Record<string, unknown>;
  result?: unknown;
  durationMs: number;
}

export interface ExecutionResult {
  success: boolean;
  taskName: string;
  input: Record<string, unknown>;
  output?: Record<string, unknown>;
  logs: LogEntry[];
  toolCalls: ToolCallSimulation[];
  memoryUsed: boolean;
  engine: string;
  durationMs: number;
  error?: string;
  /** True when run-demo returned without calling OpenAI (e.g. no key or mock). */
  mock?: boolean;
}

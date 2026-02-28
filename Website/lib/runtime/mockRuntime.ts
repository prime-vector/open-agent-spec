/**
 * Mock agent execution pipeline.
 * Simulates: prompt construction → memory injection → tool invocation → response.
 * Produces structured logs for the UI. No real LLM calls.
 */

import type { OpenAgentSpec, TaskDef, Prompts } from "@/lib/spec/types";
import type { ExecutionResult, LogEntry, ToolCallSimulation } from "./types";

function isoNow(): string {
  return new Date().toISOString();
}

function interpolate(template: string, vars: Record<string, unknown>): string {
  let out = template;
  for (const [key, value] of Object.entries(vars)) {
    out = out.replace(new RegExp(`\\{\\{\\s*${key}\\s*\\}\\}`, "g"), String(value ?? ""));
    out = out.replace(new RegExp(`\\{\\{\\s*input\\.${key}\\s*\\}\\}`, "g"), String(value ?? ""));
  }
  out = out.replace(/\{\{\s*input\.(\w+)\s*\}\}/g, (_, k) => String(vars[k] ?? ""));
  return out;
}

function buildPrompt(spec: OpenAgentSpec, taskName: string, input: Record<string, unknown>): string {
  const prompts = spec.prompts as Prompts;
  const system = (prompts[taskName] as { system?: string })?.system ?? prompts.system;
  const userTemplate = (prompts[taskName] as { user?: string })?.user ?? prompts.user;
  const user = interpolate(userTemplate, input);
  return `[System]\n${system}\n\n[User]\n${user}`;
}

function getTask(spec: OpenAgentSpec, taskName: string): TaskDef | undefined {
  return spec.tasks[taskName];
}

function simulateToolCall(toolId: string, args: Record<string, unknown>): { result: unknown; durationMs: number } {
  const durationMs = 50 + Math.floor(Math.random() * 100);
  if (toolId === "file_writer") {
    return { result: { success: true, file_path: args.file_path, bytes_written: String(args.content).length }, durationMs };
  }
  return { result: { success: true, ...args }, durationMs };
}

/**
 * Run a single task with mock execution. Produces logs and optional tool calls.
 */
export function runMockExecution(
  spec: OpenAgentSpec,
  taskName: string,
  input: Record<string, unknown>
): ExecutionResult {
  const logs: LogEntry[] = [];
  const toolCalls: ToolCallSimulation[] = [];
  const start = Date.now();

  const task = getTask(spec, taskName);
  if (!task) {
    logs.push({ kind: "error", message: `Unknown task: ${taskName}`, timestamp: isoNow() });
    return {
      success: false,
      taskName,
      input,
      logs,
      toolCalls,
      memoryUsed: false,
      engine: spec.intelligence.engine,
      durationMs: Date.now() - start,
      error: `Task not found: ${taskName}`,
    };
  }

  const engine = spec.intelligence.engine;
  const model = spec.intelligence.model;

  logs.push({
    kind: "engine",
    message: `Using ${engine} / ${model}`,
    timestamp: isoNow(),
    detail: { engine, model },
  });

  const prompt = buildPrompt(spec, taskName, input);
  logs.push({
    kind: "prompt",
    message: "Prompt constructed",
    timestamp: isoNow(),
    detail: { preview: prompt.slice(0, 200) + (prompt.length > 200 ? "…" : "") },
  });

  const memoryEnabled = spec.integration?.memory?.enabled ?? false;
  if (memoryEnabled) {
    logs.push({
      kind: "memory",
      message: "Memory context injected (mock)",
      timestamp: isoNow(),
      detail: { entries: 3 },
    });
  }

  if (task.tool) {
    const args = { ...input, file_path: input.file_path ?? "/tmp/out.txt", content: input.content ?? "" };
    const { result, durationMs } = simulateToolCall(task.tool, args);
    toolCalls.push({ toolId: task.tool, args, result, durationMs });
    logs.push({
      kind: "tool",
      message: `Tool called: ${task.tool}`,
      timestamp: isoNow(),
      detail: { toolId: task.tool, args, result },
    });
  }

  // Mock structured output from "LLM"
  const output: Record<string, unknown> = {};
  const outputSchema = task.output?.properties ?? {};
  for (const key of Object.keys(outputSchema)) {
    if (taskName === "greet" && key === "response") output[key] = `Hello, ${input.name ?? "there"}!`;
    else if (key === "greeting") output[key] = `Hello, ${input.name ?? "world"}!`;
    else if (key === "success") output[key] = true;
    else if (key === "file_path") output[key] = (input as { file_path?: string }).file_path ?? "/tmp/out.txt";
    else if (key === "bytes_written") output[key] = 42;
    else output[key] = `[mock ${key}]`;
  }

  logs.push({
    kind: "response",
    message: "Response generated",
    timestamp: isoNow(),
    detail: { output },
  });

  return {
    success: true,
    taskName,
    input,
    output,
    logs,
    toolCalls,
    memoryUsed: memoryEnabled,
    engine,
    durationMs: Date.now() - start,
  };
}

/**
 * Run the first available task with sample input (for "Run Agent" demo).
 */
export function runFirstTaskWithSampleInput(spec: OpenAgentSpec): ExecutionResult | null {
  const taskNames = Object.keys(spec.tasks).filter((t) => !spec.tasks[t].multi_step);
  const first = taskNames[0];
  if (!first) return null;
  const task = spec.tasks[first];
  const input: Record<string, unknown> = {};
  const props = task.input?.properties ?? {};
  for (const key of Object.keys(props)) {
    if (key === "name") input[key] = "Playground";
    else if (key === "message") input[key] = "Hello from the playground";
    else if (key === "file_path") input[key] = "/tmp/playground.txt";
    else if (key === "content") input[key] = "Sample content";
    else input[key] = "sample";
  }
  return runMockExecution(spec, first, input);
}

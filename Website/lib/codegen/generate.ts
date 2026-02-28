/**
 * Code generation layer.
 * Produces a minimal agent scaffold (Python or TypeScript) from a validated spec.
 * For the playground we only generate display-only code; no file writes.
 */

import type { OpenAgentSpec } from "@/lib/spec/types";

export type TargetLanguage = "python" | "typescript";

function classNameFromName(name: string): string {
  return name
    .split(/[-_]/)
    .map((s) => s.charAt(0).toUpperCase() + s.slice(1))
    .join("") + "Agent";
}

function methodNameFromTask(taskName: string): string {
  return taskName.replace(/-/g, "_");
}

export function generateScaffold(spec: OpenAgentSpec, lang: TargetLanguage): string {
  const agentName = spec.agent.name;
  const className = classNameFromName(agentName);
  const engine = spec.intelligence.engine;
  const model = spec.intelligence.model;
  const taskNames = Object.keys(spec.tasks);

  if (lang === "python") {
    const methods = taskNames
      .map(
        (t) =>
          `    def ${methodNameFromTask(t)}(self, **inputs):\n        """${spec.tasks[t].description}"""\n        # TODO: call LLM (${engine}/${model}) and return structured output\n        pass`
      )
      .join("\n\n");
    return `# Generated from Open Agent Spec — ${agentName}
# Engine: ${engine} / ${model}

class ${className}:
    def __init__(self):
        self.engine = "${engine}"
        self.model = "${model}"

${methods}
`;
  }

  const methods = taskNames
    .map(
      (t) =>
        `  async ${methodNameFromTask(t)}(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {\n    // ${spec.tasks[t].description}\n    // TODO: call LLM (${engine}/${model})\n    return {};\n  }`
    )
    .join("\n\n");
  return `// Generated from Open Agent Spec — ${agentName}
// Engine: ${engine} / ${model}

export class ${className} {
  private engine = "${engine}";
  private model = "${model}";

${methods}
}
`;
}

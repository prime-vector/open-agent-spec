import { DEFAULT_SPEC_YAML } from "../defaultSpec";
import { RESEARCH_ASSISTANT_SPEC_YAML } from "../researchAssistantSpec";
import { CODEX_SPEC_YAML } from "../codexSpec";
import { TOOLS_SPEC_YAML } from "../toolsSpec";
import { MULTI_TASK_SPEC_YAML } from "../multiTaskSpec";

export interface ExampleSpec {
  id: string;
  label: string;
  yaml: string;
}

export const EXAMPLES: ExampleSpec[] = [
  {
    id: "hello-world",
    label: "Hello World",
    yaml: DEFAULT_SPEC_YAML,
  },
  {
    id: "multi-task",
    label: "Multi-task Pipeline",
    yaml: MULTI_TASK_SPEC_YAML,
  },
  {
    id: "tool-use",
    label: "Tool Use (file.read)",
    yaml: TOOLS_SPEC_YAML,
  },
  {
    id: "research-assistant",
    label: "Research Assistant",
    yaml: RESEARCH_ASSISTANT_SPEC_YAML,
  },
  {
    id: "codex-refactor",
    label: "Codex Repo Refactor",
    yaml: CODEX_SPEC_YAML,
  },
];

export const DEFAULT_EXAMPLE_ID = "hello-world";

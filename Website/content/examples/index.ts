import { DEFAULT_SPEC_YAML } from "../defaultSpec";
import { RESEARCH_ASSISTANT_SPEC_YAML } from "../researchAssistantSpec";

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
    id: "research-assistant",
    label: "Research Assistant",
    yaml: RESEARCH_ASSISTANT_SPEC_YAML,
  },
];

export const DEFAULT_EXAMPLE_ID = "hello-world";

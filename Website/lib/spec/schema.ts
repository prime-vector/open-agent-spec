/**
 * Open Agent Spec JSON schema (minimal for playground validation).
 * Canonical schema: oas_cli/schemas/oas-schema.json in the main repo.
 */
export const oasSchema = {
  $schema: "http://json-schema.org/draft-07/schema#",
  $id: "https://openagents.org/schemas/oas-schema.json",
  type: "object",
  required: ["open_agent_spec", "agent", "intelligence", "tasks", "prompts"],
  properties: {
    open_agent_spec: {
      type: "string",
      pattern: "^(1\\.(0\\.[4-9]|[1-9]\\.[0-9]+)|[2-9]\\.[0-9]+\\.[0-9]+)$",
    },
    agent: {
      type: "object",
      properties: {
        name: { type: "string" },
        description: { type: "string" },
        role: {
          type: "string",
          enum: ["analyst", "reviewer", "chat", "retriever", "planner", "executor"],
        },
      },
      required: ["name", "description"],
    },
    intelligence: {
      type: "object",
      properties: {
        type: { type: "string", enum: ["llm"] },
        engine: {
          type: "string",
          enum: ["openai", "anthropic", "grok", "cortex", "local", "custom"],
        },
        model: { type: "string" },
        endpoint: { type: "string", pattern: "^(https?://)[^\\s]+$" },
        config: { type: "object" },
      },
      required: ["type", "engine", "model"],
    },
    tasks: {
      type: "object",
      patternProperties: {
        "^[a-zA-Z0-9_-]+$": {
          type: "object",
          properties: {
            description: { type: "string" },
            timeout: { type: "integer", minimum: 1 },
            input: { type: "object" },
            output: { type: "object" },
            tool: { type: "string" },
            multi_step: { type: "boolean" },
            steps: { type: "array" },
          },
          required: ["description", "output"],
        },
      },
    },
    prompts: {
      type: "object",
      properties: {
        system: { type: "string" },
        user: { type: "string" },
      },
      required: ["system", "user"],
    },
    logging: { type: "object" },
    behavioural_contract: { type: "object" },
    tools: { type: "array" },
    integration: { type: "object" },
  },
} as const;

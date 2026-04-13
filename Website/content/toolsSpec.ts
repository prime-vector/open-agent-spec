export const TOOLS_SPEC_YAML = `open_agent_spec: "1.5.0"

agent:
  name: file-analyst
  description: >
    Reads a file from disk using a native tool, then summarises its contents.
    Demonstrates OA native tool use — no MCP server required.
  role: analyst

intelligence:
  type: llm
  engine: openai
  model: gpt-4o-mini
  config:
    temperature: 0.2

tools:
  reader:
    type: native
    native: file.read
    description: Read a file from the local filesystem

tasks:
  summarise_file:
    description: Read a file then summarise its contents
    tools: [reader]
    input:
      type: object
      properties:
        path:
          type: string
          description: Absolute path to the file to read
      required: [path]
    output:
      type: object
      properties:
        summary:
          type: string
          description: A concise summary of the file contents
        key_points:
          type: array
          items:
            type: string
      required: [summary]
    prompts:
      system: >
        You are a concise technical analyst. Use the file.read tool to read
        the file at the given path, then summarise its contents clearly.
        Always respond with valid JSON.
      user: |
        Read the file at path: {path}

        Then respond with JSON:
        {
          "summary": "<one or two sentences>",
          "key_points": ["point 1", "point 2"]
        }
`;

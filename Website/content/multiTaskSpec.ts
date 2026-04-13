export const MULTI_TASK_SPEC_YAML = `open_agent_spec: "1.5.0"

agent:
  name: research-pipeline
  description: >
    A multi-task research pipeline. Extracts key facts from a document,
    then produces a summary. Demonstrates depends_on data chaining —
    no framework, no orchestration engine, just YAML.
  role: analyst

intelligence:
  type: llm
  engine: openai
  model: gpt-4o-mini
  config:
    temperature: 0.2

tasks:
  extract:
    description: Extract the three most important facts from the text
    input:
      type: object
      properties:
        document:
          type: string
          description: The source text to analyse
      required: [document]
    output:
      type: object
      properties:
        facts:
          type: array
          items:
            type: string
          description: Up to three key facts extracted from the document
      required: [facts]
    prompts:
      system: >
        You are a precise analyst. Extract the three most important facts
        from the provided text. Respond with valid JSON only.
      user: |
        Extract the three most important facts from this text:

        {{ document }}

        Respond with JSON: { "facts": ["fact 1", "fact 2", "fact 3"] }

  summarise:
    description: >
      Summarise the extracted facts in one sentence.
      depends_on [extract] declares a data dependency — extract's output
      (the facts array) is merged into this task's input automatically.
    depends_on: [extract]
    input:
      type: object
      properties:
        document:
          type: string
      required: [document]
    output:
      type: object
      properties:
        summary:
          type: string
          description: A single-sentence summary of the key facts
      required: [summary]
    prompts:
      system: >
        You are a concise writer. Summarise the key facts provided in
        exactly one sentence. Respond with valid JSON only.
      user: |
        Summarise these facts in one sentence:

        {{ facts }}

        Respond with JSON: { "summary": "<one sentence>" }
`;

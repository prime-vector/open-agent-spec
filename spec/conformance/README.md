# OAS Conformance Tests

This directory will contain the conformance test suite for Open Agent Spec 1.4.0. Conformance tests validate **runtime behaviour**, not LLM output.

## Purpose

The spec at `../open-agent-spec-1.4.md` defines what a conforming runtime MUST do. These tests operationalise that definition — any runtime that passes the full suite can claim OAS 1.4.0 conformance.

Conformance tests differ from unit/integration tests in this repo:

| | Conformance tests (here) | Repo tests (`tests/`) |
|-|--------------------------|----------------------|
| Target | Any OAS runtime | The reference Python implementation |
| LLM calls | MUST be mocked / stubbed | May hit real APIs (with keys) |
| Inputs | Defined by the spec | Convenience-driven |
| Pass/fail authority | Spec document | Implementation behaviour |

## Scope

The conformance suite covers the normative MUST requirements from the spec:

### Schema Validation (Section 3)
- Valid documents are accepted
- Documents missing required top-level keys are rejected
- Invalid engine values are rejected
- Invalid `open_agent_spec` version strings are rejected

### Prompt Resolution (Section 9.1)
- CLI override beats all other sources
- Per-task inline prompts beat global fallback
- Legacy per-task map beats global fallback
- Global fallback is used when no other source provides a prompt
- System and user resolve independently

### Template Substitution (Section 9.2)
- `{{ key }}` is substituted from input data
- `{{ input.key }}` is substituted from input data
- `{key}` (single-brace) is substituted from input data
- Unknown placeholders are handled gracefully (not a hard error)

### Task Execution — Single Task (Section 7.1)
- Required input fields are validated before any LLM call
- Missing required fields raise `CHAIN_INPUT_MISSING`
- Output is parsed as JSON when `response_format` is `"json"`
- Markdown code fences are stripped before JSON parsing
- Raw string is returned when `response_format` is `"text"`
- JSON schema output validation is skipped for `response_format: text`

### Task Execution — `depends_on` Chains (Section 7.2)
- Dependency tasks run before the dependent task
- Dependency outputs are merged into the input map (later wins)
- Circular dependency chains raise `CHAIN_CYCLE_ERROR` before any LLM call
- Unknown dependency names raise `TASK_NOT_FOUND`
- Result envelope includes `chain` key with intermediate results
- Tasks without `depends_on` do NOT include a `chain` key

### Spec Delegation (Section 7.3)
- Local relative paths resolve from the calling spec's directory
- `oa://` references expand to the registry URL
- Target task defaults to the calling task's name when `task:` is omitted
- Missing delegated task raises `TASK_NOT_FOUND`
- Delegation cycles raise `DELEGATION_CYCLE_ERROR` before any model call
- Result envelope includes `delegated_to` field

### Result Envelope (Section 8.2)
- All required fields are present: `task`, `output`, `input`, `prompt`, `engine`, `model`, `raw_output`
- `chain` is included iff the task has `depends_on`
- `delegated_to` is included iff the task uses `spec:` delegation

### Error Model (Section 11)
- All eight error codes are raised for their documented triggers
- Error objects contain `error`, `code`, and `stage` fields
- `task` field is included when a task name is known

### Behavioural Contract Merge (Section 10.4)
- Array fields are unioned (not replaced)
- Scalar fields use per-task-wins semantics
- Contract is enforced after output parsing, before result is returned
- Contract enforcement is skipped for `response_format: text`

## Test File Structure (Planned)

```
spec/conformance/
├── README.md                     # this file
├── cases/
│   ├── schema/
│   │   ├── valid-minimal.yaml    # minimal valid spec → should validate
│   │   ├── missing-agent.yaml    # missing required key → schema error
│   │   └── ...
│   ├── prompt-resolution/
│   │   ├── cli-override.yaml
│   │   ├── per-task-inline.yaml
│   │   ├── global-fallback.yaml
│   │   └── ...
│   ├── depends-on/
│   │   ├── linear-chain.yaml
│   │   ├── cycle-detection.yaml
│   │   └── ...
│   ├── delegation/
│   │   ├── local-path.yaml
│   │   ├── oa-scheme.yaml
│   │   ├── cycle-detection.yaml
│   │   └── ...
│   ├── response-format/
│   │   ├── json-default.yaml
│   │   ├── text-mode.yaml
│   │   ├── fence-stripping.yaml
│   │   └── ...
│   └── errors/
│       ├── chain-input-missing.yaml
│       ├── task-not-found.yaml
│       ├── chain-cycle.yaml
│       └── ...
└── runner/
    └── conformance_runner.py     # harness that executes cases against any runtime
```

## Test Case Format (Planned)

Each test case will be a YAML file with this shape:

```yaml
# spec/conformance/cases/depends-on/linear-chain.yaml
description: "Dependency output is merged into calling task input"
spec: |
  open_agent_spec: "1.4.0"
  agent:
    name: test
    description: test
  intelligence:
    type: llm
    engine: openai
    model: gpt-4o
  tasks:
    extract:
      description: extract
      output:
        type: object
        properties:
          facts: { type: string }
      prompts:
        system: ""
        user: "extract"
    summarise:
      description: summarise
      depends_on: [extract]
      output:
        type: object
        properties:
          summary: { type: string }
      prompts:
        system: ""
        user: "{{ facts }}"

mock_responses:
  extract: '{"facts": "sky is blue"}'
  summarise: '{"summary": "sky is blue"}'

invoke:
  task: summarise
  input: { document: "The sky is blue." }

expect:
  result.task: summarise
  result.output.summary: "sky is blue"
  result.chain.extract.output.facts: "sky is blue"
  # merged input must contain both original input and dep output
  result.input.facts: "sky is blue"
  result.input.document: "The sky is blue."
```

## Contributing

To add a conformance test:

1. Identify the normative requirement (section + MUST/MUST NOT keyword).
2. Create a case file under `spec/conformance/cases/<category>/`.
3. The spec embedded in the case file MUST be self-contained and minimal.
4. Mock responses MUST be provided — conformance tests MUST NOT make real API calls.
5. Add the case path to the runner's manifest.

Conformance tests assert on observable runtime behaviour. They MUST NOT assert on LLM output content.

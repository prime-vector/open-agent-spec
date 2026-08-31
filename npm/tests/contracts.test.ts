import { describe, expect, test } from "@jest/globals";
import { parseSpec } from "../src/loader.js";

const base = `
open_agent_spec: "1.6.0"
agent:
  name: contract-test
  description: test
intelligence:
  type: llm
  engine: openai
  model: gpt-4o
tasks:
  run:
    description: run
    output: {type: object}
    prompts: {system: run, user: run}
`;

describe("contract capability honesty", () => {
  test("a root contract fails closed with CONTRACTS_UNAVAILABLE", () => {
    const spec = `${base}\nbehavioural_contract:\n  version: "1.0"\n`;
    expect(() => parseSpec(spec)).toThrow(
      expect.objectContaining({
        code: "CONTRACTS_UNAVAILABLE",
        stage: "contract",
      }),
    );
  });

  test("a task contract fails closed and identifies the task", () => {
    const spec = base.replace(
      "    output: {type: object}",
      '    behavioural_contract: {version: "1.0"}\n    output: {type: object}',
    );
    expect(() => parseSpec(spec)).toThrow(
      expect.objectContaining({
        code: "CONTRACTS_UNAVAILABLE",
        stage: "contract",
        task: "run",
      }),
    );
  });
});

describe("sandbox declaration validation", () => {
  test("a URL-shaped allowlist entry is a spec-load error before unsupported-feature refusal", () => {
    const spec = `${base}\nsandbox:\n  http:\n    allow_domains: [https://api.example.com]\n`;
    expect(() => parseSpec(spec)).toThrow(
      expect.objectContaining({
        code: "SPEC_LOAD_ERROR",
        stage: "load",
      }),
    );
  });
});

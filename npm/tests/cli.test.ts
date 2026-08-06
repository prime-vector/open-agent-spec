import { describe, expect, test } from "@jest/globals";
import { reconcileSpecArgs } from "../src/cli.js";

describe("reconcileSpecArgs (#100 — bare spec path parity with the Python CLI)", () => {
  test("bare .yaml path is accepted", () => {
    expect(reconcileSpecArgs("validate", "agent.yaml", undefined)).toEqual({ spec: "agent.yaml" });
  });

  test("bare .yml path is accepted, case-insensitively", () => {
    expect(reconcileSpecArgs("run", "Agent.YML", undefined)).toEqual({ spec: "Agent.YML" });
  });

  test("--spec alone still works", () => {
    expect(reconcileSpecArgs("validate", undefined, "agent.yaml")).toEqual({ spec: "agent.yaml" });
  });

  test("--spec is not suffix-gated (any path allowed, as before)", () => {
    expect(reconcileSpecArgs("run", undefined, "specs/agent.config")).toEqual({
      spec: "specs/agent.config",
    });
  });

  test("bare path and --spec together is an explicit error", () => {
    const result = reconcileSpecArgs("validate", "a.yaml", "b.yaml");
    expect(result).toHaveProperty("error");
    expect((result as { error: string }).error).toContain("not both");
  });

  test("non-YAML bare argument errors and names the valid forms", () => {
    const result = reconcileSpecArgs("run", "notes.txt", undefined);
    expect(result).toHaveProperty("error");
    const error = (result as { error: string }).error;
    expect(error).toContain("notes.txt");
    expect(error).toContain("oa run <spec.yaml>");
    expect(error).toContain("--spec");
  });

  test("neither form errors and names the valid forms", () => {
    const result = reconcileSpecArgs("validate", undefined, undefined);
    expect(result).toHaveProperty("error");
    expect((result as { error: string }).error).toContain("oa validate <spec.yaml>");
  });
});

/**
 * YAML parsing for Open Agent Spec.
 * Thin wrapper; validation is separate.
 */

import yaml from "js-yaml";
import type { OpenAgentSpec } from "./types";
import { validateSpecData, type ValidationResult } from "./validate";

export interface ParseResult extends ValidationResult {
  raw?: unknown;
  parseError?: string;
}

/**
 * Parse YAML string and validate. Returns parse errors or validation errors.
 */
export function parseAndValidateSpec(yamlString: string): ParseResult {
  let raw: unknown;
  try {
    raw = yaml.load(yamlString);
  } catch (e) {
    const message = e instanceof Error ? e.message : "Invalid YAML";
    return { success: false, errors: [], raw: undefined, parseError: message };
  }

  if (raw == null || typeof raw !== "object") {
    return {
      success: false,
      errors: [{ path: "/", message: "Spec must be a YAML object" }],
      raw,
      parseError: "Expected a YAML object",
    };
  }

  const result = validateSpecData(raw);
  return { ...result, raw };
}

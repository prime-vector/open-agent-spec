/**
 * Spec validation layer.
 * Parses YAML and validates against the Open Agent Spec JSON schema.
 * Returns structured errors for UI display.
 */

import Ajv, { type ErrorObject } from "ajv";
import addFormats from "ajv-formats";
import type { OpenAgentSpec } from "./types";
import { oasSchema } from "./schema";

export interface ValidationError {
  path: string;
  message: string;
  schemaPath?: string;
}

export interface ValidationResult {
  success: boolean;
  spec?: OpenAgentSpec;
  errors: ValidationError[];
}

const ajv = new Ajv({ allErrors: true, strict: false });
addFormats(ajv);
const validateSchema = ajv.compile(oasSchema as object);

function toValidationErrors(errors: ErrorObject[] | null): ValidationError[] {
  if (!errors?.length) return [];
  return errors.map((e) => ({
    path: e.instancePath || "/",
    message: e.message ?? "Validation failed",
    schemaPath: e.schemaPath,
  }));
}

/**
 * Validate spec data (already parsed) against the JSON schema.
 */
export function validateSpecData(data: unknown): ValidationResult {
  const valid = validateSchema(data);
  if (valid) {
    return { success: true, spec: data as OpenAgentSpec, errors: [] };
  }
  return {
    success: false,
    errors: toValidationErrors(validateSchema.errors ?? null),
  };
}

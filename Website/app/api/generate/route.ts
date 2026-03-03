/**
 * POST /api/generate
 * Body: { yaml: string }
 * Runs the real Open Agent Spec Python generator and returns generated files.
 * Requires Python + open-agent-spec to be available (e.g. from repo root when running locally).
 */

import { spawnSync } from "child_process";
import fs from "fs";
import os from "os";
import path from "path";

const REPO_ROOT = path.resolve(process.cwd(), "..");
const SCRIPT_PATH = path.join(process.cwd(), "scripts", "invoke_generator.py");
const PYTHON = process.env.PYTHON_PATH ?? "python3";

export interface GenerateResponse {
  agentPy?: string;
  readme?: string;
  requirementsTxt?: string;
  envExample?: string;
  prompts?: Record<string, string>;
  error?: string;
}

export async function POST(request: Request): Promise<Response> {
  let tmpDir: string | null = null;

  try {
    const body = await request.json();
    const yaml = typeof body?.yaml === "string" ? body.yaml : "";

    if (!yaml.trim()) {
      return Response.json(
        { error: "Missing or empty 'yaml' in request body" },
        { status: 400 }
      );
    }

    if (!fs.existsSync(SCRIPT_PATH)) {
      return Response.json(
        {
          error:
            "Generator script not found. Use the in-browser scaffold when running without the full repo.",
          fallback: true,
        },
        { status: 503 }
      );
    }

    tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), "oas-"));
    const specPath = path.join(tmpDir, "spec.yaml");
    const outputDir = path.join(tmpDir, "out");
    fs.writeFileSync(specPath, yaml, "utf-8");

    const proc = spawnSync(PYTHON, [SCRIPT_PATH, specPath, outputDir], {
      encoding: "utf-8",
      cwd: REPO_ROOT,
      maxBuffer: 4 * 1024 * 1024,
      input: undefined,
    });

    const stdout = (proc.stdout ?? "").trim();
    const stderr = (proc.stderr ?? "").trim();

    if (proc.status !== 0) {
      let errMsg = "Generation failed";
      try {
        const errJson = JSON.parse(stderr || "{}") as { error?: string };
        if (errJson.error) errMsg = errJson.error;
      } catch {
        if (stderr) errMsg = stderr.slice(0, 500);
      }
      return Response.json(
        { error: errMsg, fallback: true },
        { status: 422 }
      );
    }

    if (!stdout) {
      return Response.json(
        { error: "Generator produced no output", fallback: true },
        { status: 502 }
      );
    }

    let data: GenerateResponse;
    try {
      data = JSON.parse(stdout) as GenerateResponse;
    } catch {
      return Response.json(
        { error: "Invalid generator output", fallback: true },
        { status: 502 }
      );
    }

    if (data.error) {
      return Response.json(
        { error: data.error, fallback: true },
        { status: 422 }
      );
    }

    return Response.json(data);
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    if (
      message.includes("spawn") ||
      message.includes("ENOENT") ||
      message.includes("Python")
    ) {
      return Response.json(
        {
          error:
            "Python generator unavailable. Ensure Python 3 and open-agent-spec are installed from the repo root.",
          fallback: true,
        },
        { status: 503 }
      );
    }
    return Response.json(
      { error: message || "Generation failed", fallback: true },
      { status: 500 }
    );
  } finally {
    try {
      if (tmpDir && fs.existsSync(tmpDir)) {
        fs.rmSync(tmpDir, { recursive: true });
      }
    } catch {
      // ignore cleanup errors
    }
  }
}

/**
 * POST /api/run-demo
 * Body: { yaml: string, apiKey?: string, input?: Record<string, unknown> }
 * Runs the first task with a real OpenAI call (when apiKey provided).
 * Rate limit: 1 successful run per IP per calendar day.
 *
 * To temporarily disable the rate limit (e.g. for recording a demo), set
 * DISABLE_DEMO_RATE_LIMIT=1 in the environment. Remove it when done.
 */

import { NextRequest } from "next/server";
import yaml from "js-yaml";

// In-memory rate limit: IP -> { date (YYYY-MM-DD), count }
// For production with multiple instances use Redis or Vercel KV
const rateLimitMap = new Map<string, { date: string; count: number }>();
const MAX_RUNS_PER_IP_PER_DAY = 1;

const RATE_LIMIT_DISABLED =
  process.env.DISABLE_DEMO_RATE_LIMIT === "1" ||
  process.env.DISABLE_DEMO_RATE_LIMIT === "true";

function getClientIp(request: NextRequest): string {
  const forwarded = request.headers.get("x-forwarded-for");
  if (forwarded) return forwarded.split(",")[0].trim();
  const real = request.headers.get("x-real-ip");
  if (real) return real;
  return "unknown";
}

function getToday(): string {
  return new Date().toISOString().slice(0, 10);
}

function isRateLimited(ip: string): boolean {
  const today = getToday();
  const entry = rateLimitMap.get(ip);
  if (!entry) return false;
  if (entry.date !== today) return false;
  return entry.count >= MAX_RUNS_PER_IP_PER_DAY;
}

function recordRun(ip: string): void {
  const today = getToday();
  const entry = rateLimitMap.get(ip);
  if (!entry || entry.date !== today) {
    rateLimitMap.set(ip, { date: today, count: 1 });
    return;
  }
  entry.count += 1;
  // Prune old entries occasionally (simple: keep map under 10k entries)
  if (rateLimitMap.size > 10000) {
    const toDelete: string[] = [];
    rateLimitMap.forEach((v, k) => {
      if (v.date !== today) toDelete.push(k);
    });
    toDelete.forEach((k) => rateLimitMap.delete(k));
  }
}

interface SpecTask {
  description: string;
  input?: { properties?: Record<string, unknown>; required?: string[] };
  output?: { properties?: Record<string, unknown> };
}

interface SpecPrompts {
  system?: string;
  user?: string;
  [task: string]: unknown;
}

interface ParsedSpec {
  tasks?: Record<string, SpecTask>;
  prompts?: SpecPrompts;
  intelligence?: { engine?: string; model?: string };
}

function buildPrompt(
  prompts: SpecPrompts,
  taskName: string,
  input: Record<string, unknown>
): string {
  const system =
    (prompts[taskName] as { system?: string })?.system ?? prompts.system ?? "";
  const userTemplate =
    (prompts[taskName] as { user?: string })?.user ?? prompts.user ?? "{{ name }}";
  let user = userTemplate;
  for (const [key, value] of Object.entries(input)) {
    user = user.replace(
      new RegExp(`\\{\\{\\s*${key}\\s*\\}\\}`, "g"),
      String(value ?? "")
    );
    user = user.replace(
      new RegExp(`\\{\\{\\s*input\\.${key}\\s*\\}\\}`, "g"),
      String(value ?? "")
    );
  }
  return `${system}\n\n${user}`;
}

export async function POST(request: NextRequest): Promise<Response> {
  const ip = getClientIp(request);

  if (!RATE_LIMIT_DISABLED && isRateLimited(ip)) {
    return Response.json(
      {
        error:
          "Demo run limit reached (1 per IP per day). Try again tomorrow or run the generated code locally with your own key.",
        rateLimited: true,
      },
      { status: 429 }
    );
  }

  try {
    const body = await request.json();
    const yamlStr = typeof body?.yaml === "string" ? body.yaml : "";
    const apiKey =
      typeof body?.apiKey === "string" ? body.apiKey.trim() : undefined;
    const inputOverride =
      body?.input && typeof body.input === "object" ? body.input : undefined;

    if (!yamlStr.trim()) {
      return Response.json(
        { error: "Missing or empty 'yaml' in request body" },
        { status: 400 }
      );
    }

    let spec: ParsedSpec;
    try {
      spec = (yaml.load(yamlStr) as ParsedSpec) ?? {};
    } catch (e) {
      const msg = e instanceof Error ? e.message : "Invalid YAML";
      return Response.json({ error: `Invalid YAML: ${msg}` }, { status: 400 });
    }

    const tasks = spec.tasks ?? {};
    const taskNames = Object.keys(tasks).filter((n) => !(tasks[n] as SpecTask & { multi_step?: boolean }).multi_step);
    const firstTask = taskNames[0];
    if (!firstTask) {
      return Response.json(
        { error: "No runnable task found in spec" },
        { status: 400 }
      );
    }

    const task = tasks[firstTask];
    const prompts = spec.prompts ?? { system: "", user: "" };

    // Build sample input
    const input: Record<string, unknown> = inputOverride ?? {};
    const props = task.input?.properties ?? {};
    for (const key of Object.keys(props)) {
      if (!(key in input)) {
        if (key === "name") input[key] = "Playground";
        else if (key === "message") input[key] = "Hello from the playground";
        else if (key === "topic") input[key] = "Renewable energy trends in 2024";
        else input[key] = "sample";
      }
    }

    const engine = spec.intelligence?.engine ?? "openai";
    const model = spec.intelligence?.model ?? "gpt-4";

    // If no API key: return mock result (no rate limit consumed)
    if (!apiKey) {
      const promptPreview = buildPrompt(prompts, firstTask, input);
      const outputProps = task.output?.properties ?? {};
      const hasSummary = "summary" in outputProps;
      const hasKeyPoints = "key_points" in outputProps;
      const mockOutput: Record<string, unknown> =
        hasSummary && hasKeyPoints
          ? {
              summary: "Mock summary (add your OpenAI key to run for real).",
              key_points: [],
            }
          : { response: `Hello, ${input.name ?? "there"}! (mock — add your OpenAI key to run for real)` };
      return Response.json({
        success: true,
        taskName: firstTask,
        input,
        output: mockOutput,
        logs: [
          {
            kind: "info",
            message: "No API key provided — mock result. Add your OpenAI key to run a real demo (1 per IP per day).",
            timestamp: new Date().toISOString(),
          },
          {
            kind: "prompt",
            message: "Prompt constructed",
            timestamp: new Date().toISOString(),
            detail: { preview: promptPreview.slice(0, 200) + (promptPreview.length > 200 ? "…" : "") },
          },
        ],
        toolCalls: [],
        memoryUsed: false,
        engine,
        durationMs: 0,
        mock: true,
      });
    }

    // Real OpenAI call (only for openai engine)
    if (engine !== "openai") {
      return Response.json(
        {
          error: `Demo run supports OpenAI only; spec uses engine "${engine}". Use OpenAI in the spec to try the demo.`,
        },
        { status: 400 }
      );
    }

    const prompt = buildPrompt(prompts, firstTask, input);
    const start = Date.now();

    const openaiRes = await fetch("https://api.openai.com/v1/chat/completions", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${apiKey}`,
      },
      body: JSON.stringify({
        model: model || "gpt-4o-mini",
        messages: [
          { role: "system", content: "You are a helpful agent. Respond only with valid JSON matching the requested output schema." },
          { role: "user", content: prompt },
        ],
        temperature: 0.3,
        max_tokens: 500,
      }),
    });

    const durationMs = Date.now() - start;

    if (!openaiRes.ok) {
      const errBody = await openaiRes.text();
      let errMsg = `OpenAI API error: ${openaiRes.status}`;
      try {
        const j = JSON.parse(errBody) as { error?: { message?: string } };
        if (j.error?.message) errMsg = j.error.message;
      } catch {
        if (errBody) errMsg = errBody.slice(0, 300);
      }
      return Response.json(
        { error: errMsg },
        { status: openaiRes.status >= 500 ? 502 : 400 }
      );
    }

    const data = (await openaiRes.json()) as {
      choices?: Array<{ message?: { content?: string } }>;
    };
    const content = data.choices?.[0]?.message?.content?.trim() ?? "";

    // Parse JSON from response (handle markdown code blocks)
    let output: Record<string, unknown> = {};
    let raw = content;
    const codeMatch = content.match(/```(?:json)?\s*([\s\S]*?)```/);
    if (codeMatch) raw = codeMatch[1].trim();
    try {
      output = JSON.parse(raw) as Record<string, unknown>;
    } catch {
      output = { response: content || "(no structured output)" };
    }

    if (!RATE_LIMIT_DISABLED) recordRun(ip);

    return Response.json({
      success: true,
      taskName: firstTask,
      input,
      output,
      logs: [
        { kind: "engine", message: `Using ${engine} / ${model}`, timestamp: new Date().toISOString(), detail: { engine, model } },
        { kind: "prompt", message: "Prompt constructed", timestamp: new Date().toISOString(), detail: { preview: prompt.slice(0, 200) + "…" } },
        { kind: "response", message: "Response received", timestamp: new Date().toISOString(), detail: { output } },
      ],
      toolCalls: [],
      memoryUsed: false,
      engine,
      durationMs,
      mock: false,
    });
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    return Response.json(
      { error: message || "Run failed" },
      { status: 500 }
    );
  }
}

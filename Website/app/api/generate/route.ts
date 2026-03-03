import { NextRequest } from "next/server";

/**
 * POST /api/generate
 * Body: { yaml: string }
 * Runs the real Open Agent Spec Python generator (if available) and returns generated files.
 * In production on Vercel, this proxies to the Python function at /api/cli_generate.
 * When that function is unavailable (e.g. local dev without Vercel), it falls back
 * to the in-browser scaffold in the client.
 */

// Simple per-IP daily rate limit to prevent abuse of the generator.
// For production with many instances, back this with Redis / Vercel KV.
const rateLimitMap = new Map<string, { date: string; count: number }>();
const MAX_GENERATE_PER_IP_PER_DAY = 100;

function getToday(): string {
  return new Date().toISOString().slice(0, 10);
}

function getClientIp(request: NextRequest): string {
  const forwarded = request.headers.get("x-forwarded-for");
  if (forwarded) return forwarded.split(",")[0]!.trim();
  const real = request.headers.get("x-real-ip");
  if (real) return real;
  return "unknown";
}

function isRateLimited(ip: string): boolean {
  const today = getToday();
  const entry = rateLimitMap.get(ip);
  if (!entry) return false;
  if (entry.date !== today) return false;
  return entry.count >= MAX_GENERATE_PER_IP_PER_DAY;
}

function recordGenerate(ip: string): void {
  const today = getToday();
  const entry = rateLimitMap.get(ip);
  if (!entry || entry.date !== today) {
    rateLimitMap.set(ip, { date: today, count: 1 });
    return;
  }
  entry.count += 1;
  if (rateLimitMap.size > 10000) {
    const toDelete: string[] = [];
    rateLimitMap.forEach((v, k) => {
      if (v.date !== today) toDelete.push(k);
    });
    toDelete.forEach((k) => rateLimitMap.delete(k));
  }
}

export interface GenerateResponse {
  agentPy?: string;
  readme?: string;
  requirementsTxt?: string;
  envExample?: string;
  prompts?: Record<string, string>;
  error?: string;
  fallback?: boolean;
}

export async function POST(request: NextRequest): Promise<Response> {
  const ip = getClientIp(request);

  if (isRateLimited(ip)) {
    return Response.json(
      {
        error:
          "Generate limit reached (100 per IP per day). Please try again tomorrow or run the CLI locally.",
        fallback: true,
      },
      { status: 429 }
    );
  }

  try {
    const body = await request.json();
    const yaml = typeof body?.yaml === "string" ? body.yaml : "";

    if (!yaml.trim()) {
      return Response.json(
        { error: "Missing or empty 'yaml' in request body" },
        { status: 400 }
      );
    }

    // Call the Python Vercel Function at /api/cli_generate on the same host.
    const url = new URL("/api/cli_generate", request.url);
    let res: Response;
    try {
      res = await fetch(url.toString(), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ yaml }),
      });
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      return Response.json(
        {
          error:
            message ||
            "Generator function not reachable; using in-browser scaffold instead.",
          fallback: true,
        },
        { status: 503 }
      );
    }

    let data: GenerateResponse;
    try {
      data = (await res.json()) as GenerateResponse;
    } catch {
      return Response.json(
        {
          error: "Generator function returned invalid JSON; using scaffold instead.",
          fallback: true,
        },
        { status: 502 }
      );
    }

    if (!res.ok) {
      return Response.json(
        {
          error:
            data.error ||
            `Generator function failed with status ${res.status}; using scaffold instead.`,
          fallback: true,
        },
        { status: res.status }
      );
    }

    if (data.error) {
      return Response.json(
        { error: data.error, fallback: true },
        { status: 422 }
      );
    }

    recordGenerate(ip);
    return Response.json(data);
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    return Response.json(
      { error: message || "Generation failed", fallback: true },
      { status: 500 }
    );
  }
}


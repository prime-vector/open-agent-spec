/**
 * POST /api/generate
 * Body: { yaml: string }
 * Runs the real Open Agent Spec Python generator (if available) and returns generated files.
 * In production on Vercel, this proxies to the Python function at /api/cli_generate.
 * When that function is unavailable (e.g. local dev without Vercel), it falls back
 * to the in-browser scaffold in the client.
 */

export interface GenerateResponse {
  agentPy?: string;
  readme?: string;
  requirementsTxt?: string;
  envExample?: string;
  prompts?: Record<string, string>;
  error?: string;
  fallback?: boolean;
}

export async function POST(request: Request): Promise<Response> {
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

    return Response.json(data);
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    return Response.json(
      { error: message || "Generation failed", fallback: true },
      { status: 500 }
    );
  }
}


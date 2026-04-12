import type { ProviderConfig, ChatMessage } from "../types.js";
import { OAError } from "../loader.js";

const DEFAULT_BASE = "https://api.openai.com/v1";

// xAI / Grok use the same shape as OpenAI — different base URL only.
const ENGINE_BASES: Record<string, string> = {
  openai: DEFAULT_BASE,
  grok: "https://api.x.ai/v1",
  "x-ai": "https://api.x.ai/v1",
};

const ENGINE_ENV_KEYS: Record<string, string> = {
  openai: "OPENAI_API_KEY",
  grok: "XAI_API_KEY",
  "x-ai": "XAI_API_KEY",
};

export async function invokeOpenAI(
  system: string,
  user: string,
  config: ProviderConfig,
): Promise<string> {
  const engine = config.engine.toLowerCase();
  const base = ENGINE_BASES[engine] ?? DEFAULT_BASE;
  const envKey = ENGINE_ENV_KEYS[engine] ?? "OPENAI_API_KEY";
  const apiKey = process.env[envKey];

  if (!apiKey) {
    throw new OAError(
      `Missing environment variable ${envKey} for engine '${engine}'`,
      "PROVIDER_ERROR",
      "invoke",
    );
  }

  const messages: ChatMessage[] = [];
  if (system) messages.push({ role: "system", content: system });
  messages.push({ role: "user", content: user });

  const extraConfig = (config.config ?? {}) as Record<string, unknown>;
  const body: Record<string, unknown> = {
    model: config.model,
    messages,
    temperature: extraConfig["temperature"] ?? 0.2,
  };
  if (extraConfig["max_tokens"]) body["max_tokens"] = extraConfig["max_tokens"];

  const res = await fetch(`${base}/chat/completions`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${apiKey}`,
    },
    body: JSON.stringify(body),
  });

  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new OAError(
      `OpenAI-compatible API error ${res.status}: ${text}`,
      "PROVIDER_ERROR",
      "invoke",
    );
  }

  const json = (await res.json()) as {
    choices?: Array<{ message?: { content?: string } }>;
  };
  const content = json.choices?.[0]?.message?.content;
  if (!content) {
    throw new OAError(
      "OpenAI-compatible API returned an empty response",
      "PROVIDER_ERROR",
      "invoke",
    );
  }
  return content;
}

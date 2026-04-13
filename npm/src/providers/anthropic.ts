import type { ProviderConfig, ChatMessage } from "../types.js";
import { OAError } from "../loader.js";

const ANTHROPIC_BASE = "https://api.anthropic.com/v1";
const ANTHROPIC_VERSION = "2023-06-01";

export async function invokeAnthropic(
  system: string,
  user: string,
  config: ProviderConfig,
  history?: ChatMessage[],
): Promise<string> {
  const apiKey = process.env["ANTHROPIC_API_KEY"];
  if (!apiKey) {
    throw new OAError(
      "Missing environment variable ANTHROPIC_API_KEY",
      "PROVIDER_ERROR",
      "invoke",
    );
  }

  const messages: ChatMessage[] = [];
  if (history?.length) messages.push(...history);
  messages.push({ role: "user", content: user });

  const extraConfig = (config.config ?? {}) as Record<string, unknown>;
  const body: Record<string, unknown> = {
    model: config.model,
    max_tokens: extraConfig["max_tokens"] ?? 4096,
    messages,
  };
  if (system) body["system"] = system;
  if (extraConfig["temperature"] !== undefined)
    body["temperature"] = extraConfig["temperature"];

  const res = await fetch(`${ANTHROPIC_BASE}/messages`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "x-api-key": apiKey,
      "anthropic-version": ANTHROPIC_VERSION,
    },
    body: JSON.stringify(body),
  });

  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new OAError(
      `Anthropic API error ${res.status}: ${text}`,
      "PROVIDER_ERROR",
      "invoke",
    );
  }

  const json = (await res.json()) as {
    content?: Array<{ type: string; text?: string }>;
  };
  const content = json.content?.find((b) => b.type === "text")?.text;
  if (!content) {
    throw new OAError(
      "Anthropic API returned an empty text response",
      "PROVIDER_ERROR",
      "invoke",
    );
  }
  return content;
}

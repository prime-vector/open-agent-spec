import type { ProviderConfig, ChatMessage } from "../types.js";
import { OAError } from "../loader.js";
import { invokeOpenAI } from "./openai.js";
import { invokeAnthropic } from "./anthropic.js";

const OPENAI_COMPATIBLE = new Set(["openai", "grok", "x-ai"]);
const ANTHROPIC_COMPATIBLE = new Set(["anthropic"]);

export async function invokeProvider(
  system: string,
  user: string,
  config: ProviderConfig,
  history?: ChatMessage[],
): Promise<string> {
  const engine = config.engine.toLowerCase();

  if (OPENAI_COMPATIBLE.has(engine)) {
    return invokeOpenAI(system, user, config, history);
  }
  if (ANTHROPIC_COMPATIBLE.has(engine)) {
    return invokeAnthropic(system, user, config, history);
  }

  throw new OAError(
    `Unsupported engine '${config.engine}'. Supported: openai, anthropic, grok, x-ai`,
    "PROVIDER_ERROR",
    "invoke",
  );
}

export { invokeOpenAI, invokeAnthropic };

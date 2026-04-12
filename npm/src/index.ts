// Public API — import these when using open-agent-spec as a library.
export { runTask, runTaskFromSpec } from "./runner.js";
export { loadSpecFromFile, parseSpec, chooseTask, OAError } from "./loader.js";
export { resolveSpecUrl, isRemoteRef, fetchRemoteSpec } from "./registry.js";
export type {
  OASpec,
  TaskDef,
  InlineTask,
  DelegatedTask,
  TaskResult,
  RunInput,
  ProviderConfig,
} from "./types.js";

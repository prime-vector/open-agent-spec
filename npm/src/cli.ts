import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { Command } from "commander";
import { runTask } from "./runner.js";
import { loadSpecFromFile, OAError } from "./loader.js";
import type { RunInput } from "./types.js";

function readPackageVersion(): string {
  const here = dirname(fileURLToPath(import.meta.url));
  for (const candidate of [
    resolve(here, "..", "package.json"),
    resolve(here, "..", "..", "package.json"),
  ]) {
    try {
      const pkg = JSON.parse(readFileSync(candidate, "utf8")) as { version?: string };
      if (pkg.version) return pkg.version;
    } catch {
      // try next candidate
    }
  }
  return "unknown";
}

function parseInput(raw: string): RunInput {
  const trimmed = raw.trim();

  // Inline JSON string
  if (trimmed.startsWith("{")) {
    try {
      return JSON.parse(trimmed) as RunInput;
    } catch (e) {
      console.error(`Error: invalid JSON for --input: ${String(e)}`);
      process.exit(1);
    }
  }

  // File path
  const filePath = resolve(trimmed);
  let contents: string;
  try {
    contents = readFileSync(filePath, "utf8");
  } catch {
    console.error(`Error: input file not found: ${filePath}`);
    process.exit(1);
  }

  if (trimmed.toLowerCase().endsWith(".json")) {
    try {
      const parsed = JSON.parse(contents) as unknown;
      if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
        console.error("Error: input .json file must contain a JSON object, not an array or scalar.");
        process.exit(1);
      }
      return parsed as RunInput;
    } catch (e) {
      console.error(`Error: invalid JSON in file '${filePath}': ${String(e)}`);
      process.exit(1);
    }
  }

  // Plain text file — passed as the single "text" field for convenience.
  return { text: contents };
}

// Reconcile the bare positional spec path with --spec (#94/#100 parity with
// the Python CLI): either form alone is fine, both is an explicit error, and a
// bare argument must look like a YAML path so mistakes get a clear message
// instead of a confusing load failure.
export function reconcileSpecArgs(
  cmd: string,
  specArg: string | undefined,
  specOpt: string | undefined,
): { spec: string } | { error: string } {
  const forms = `Valid forms: 'oa ${cmd} <spec.yaml>' or 'oa ${cmd} --spec <path>'.`;
  if (specArg !== undefined && specOpt !== undefined) {
    return { error: "Pass the spec path either as a bare argument or with --spec, not both." };
  }
  if (specArg !== undefined && !/\.ya?ml$/i.test(specArg)) {
    return { error: `'${specArg}' does not look like a spec YAML path. ${forms}` };
  }
  const spec = specArg ?? specOpt;
  if (spec === undefined) {
    return { error: `Missing spec path. ${forms}` };
  }
  return { spec };
}

function resolveSpecOrExit(
  cmd: string,
  specArg: string | undefined,
  specOpt: string | undefined,
): string {
  const result = reconcileSpecArgs(cmd, specArg, specOpt);
  if ("error" in result) {
    console.error(`Error: ${result.error}`);
    process.exit(1);
  }
  return result.spec;
}

export function createCli(): Command {
  const program = new Command();

  program
    .name("oa")
    .description("Open Agent Spec runner — execute YAML agent specs from the command line.")
    .version(readPackageVersion());

  program
    .command("validate")
    .description("Validate a spec file against the Open Agent Spec schema (no model calls).")
    .argument("[spec]", "Path to the spec YAML file (shorthand for --spec)")
    .option("--spec <path>", "Path to the spec YAML file")
    .action((specArg: string | undefined, opts: { spec?: string }) => {
      const specPath = resolve(resolveSpecOrExit("validate", specArg, opts.spec));
      try {
        loadSpecFromFile(specPath);
      } catch (err) {
        if (err instanceof OAError) {
          console.error(`Error [${err.code}]: ${err.message}`);
        } else {
          console.error(`Error: ${String(err)}`);
        }
        process.exit(1);
      }
      console.log(`Spec is valid: ${specPath}`);
    });

  program
    .command("run")
    .description("Run a task from an OA spec file.")
    .argument("[spec]", "Path to the spec YAML file (shorthand for --spec)")
    .option("--spec <path>", "Path to the spec YAML file")
    .option("--task <name>", "Task name to run (defaults to the only task if there is one)")
    .option("--input <json-or-file>", "Input as a JSON string, a .json file path, or a plain text file path")
    .option("--quiet", "Output only JSON (no decorative logging)")
    .action(async (specArg: string | undefined, opts: {
      spec?: string;
      task?: string;
      input?: string;
      quiet?: boolean;
    }) => {
      const specPath = resolve(resolveSpecOrExit("run", specArg, opts.spec));
      const input: RunInput = opts.input ? parseInput(opts.input) : {};

      try {
        const result = await runTask({
          specPath,
          taskName: opts.task,
          input,
        });

        console.log(JSON.stringify(result, null, 2));
      } catch (err) {
        if (err instanceof OAError) {
          if (!opts.quiet) {
            console.error(`Error [${err.code}]: ${err.message}`);
          } else {
            console.error(JSON.stringify({ error: err.message, code: err.code, stage: err.stage }));
          }
        } else {
          console.error(`Error: ${String(err)}`);
        }
        process.exit(1);
      }
    });

  return program;
}

export async function main(): Promise<void> {
  const program = createCli();
  await program.parseAsync(process.argv);
}

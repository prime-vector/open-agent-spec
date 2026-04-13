import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { Command } from "commander";
import { runTask } from "./runner.js";
import { OAError } from "./loader.js";
import type { RunInput } from "./types.js";

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

export function createCli(): Command {
  const program = new Command();

  program
    .name("oa")
    .description("Open Agent Spec runner — execute YAML agent specs from the command line.")
    .version("1.5.0");

  program
    .command("run")
    .description("Run a task from an OA spec file.")
    .requiredOption("--spec <path>", "Path to the spec YAML file")
    .option("--task <name>", "Task name to run (defaults to the only task if there is one)")
    .option("--input <json-or-file>", "Input as a JSON string, a .json file path, or a plain text file path")
    .option("--quiet", "Output only JSON (no decorative logging)")
    .action(async (opts: {
      spec: string;
      task?: string;
      input?: string;
      quiet?: boolean;
    }) => {
      const input: RunInput = opts.input ? parseInput(opts.input) : {};

      try {
        const result = await runTask({
          specPath: resolve(opts.spec),
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

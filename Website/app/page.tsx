"use client";

import React, { useCallback, useState, useEffect } from "react";
import { SplitScreen } from "../components/layout/SplitScreen";
import { ResultTabs, type ResultTabId } from "../components/layout/ResultTabs";
import { YamlEditor } from "../components/editor/YamlEditor";
import { GenerateButton } from "../components/playground/GenerateButton";
import { RunButton } from "../components/playground/RunButton";
import { GeneratedCodeTab } from "../components/output/GeneratedCodeTab";
import { LogsTab } from "../components/output/LogsTab";
import { OutputTab } from "../components/output/OutputTab";
import { parseAndValidateSpec } from "../lib/spec/parse";
import { generateScaffold, type TargetLanguage } from "../lib/codegen/generate";
import { runFirstTaskWithSampleInput } from "../lib/runtime/mockRuntime";
import type { OpenAgentSpec } from "../lib/spec/types";
import type { ExecutionResult } from "../lib/runtime/types";
import { EXAMPLES, DEFAULT_EXAMPLE_ID } from "../content/examples";

export default function PlaygroundPage() {
  const [selectedExampleId, setSelectedExampleId] = useState(DEFAULT_EXAMPLE_ID);
  const [yaml, setYaml] = useState(
    () => EXAMPLES.find((e) => e.id === DEFAULT_EXAMPLE_ID)?.yaml ?? ""
  );
  const [parseError, setParseError] = useState<string | null>(null);
  const [validationErrors, setValidationErrors] = useState<{ path: string; message: string }[]>([]);
  const [spec, setSpec] = useState<OpenAgentSpec | null>(null);
  const [generatedCode, setGeneratedCode] = useState<string | null>(null);
  const [generatedFiles, setGeneratedFiles] = useState<Record<string, string> | null>(null);
  const [targetLang, setTargetLang] = useState<TargetLanguage>("python");
  const [activeTab, setActiveTab] = useState<ResultTabId>("code");
  const [executionResult, setExecutionResult] = useState<ExecutionResult | null>(null);
  const [generateLoading, setGenerateLoading] = useState(false);
  const [generateError, setGenerateError] = useState<string | null>(null);
  const [runLoading, setRunLoading] = useState(false);
  const [runError, setRunError] = useState<string | null>(null);
  const [useOpenAI, setUseOpenAI] = useState(false);
  const [apiKey, setApiKey] = useState("");

  const handleGenerate = useCallback(async () => {
    if (!spec) return;
    setGenerateError(null);
    if (targetLang === "python") {
      setGenerateLoading(true);
      try {
        const res = await fetch("/api/generate", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ yaml }),
        });
        const data = await res.json();
        if (res.ok && data.agentPy) {
          setGeneratedCode(data.agentPy);
          setGeneratedFiles({
            readme: data.readme,
            requirementsTxt: data.requirementsTxt,
            envExample: data.envExample,
          });
          setActiveTab("code");
          return;
        }
        if (data.fallback && data.error) {
          setGenerateError(data.error);
        }
      } catch (e) {
        setGenerateError(e instanceof Error ? e.message : "Generate request failed");
      } finally {
        setGenerateLoading(false);
      }
    }
    setGeneratedFiles(null);
    setGeneratedCode(generateScaffold(spec, targetLang));
    setActiveTab("code");
  }, [spec, targetLang, yaml]);

  const handleRun = useCallback(() => {
    if (!spec) return;
    const result = runFirstTaskWithSampleInput(spec);
    setExecutionResult(result ?? null);
    setActiveTab(result ? "output" : "logs");
  }, [spec]);

  const handleRunFromSpec = useCallback(async () => {
    if (!spec) return;
    setRunError(null);
    setRunLoading(true);
    try {
      const res = await fetch("/api/run-demo", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          yaml,
          apiKey: useOpenAI ? apiKey.trim() || undefined : undefined,
        }),
      });
      const data = await res.json();
      if (!res.ok) {
        if (data.rateLimited) {
          setRunError(data.error ?? "Rate limit reached.");
        } else {
          setRunError(data.error ?? `Request failed (${res.status})`);
        }
        return;
      }
      setExecutionResult(data as ExecutionResult);
      setActiveTab("output");
    } catch (e) {
      setRunError(e instanceof Error ? e.message : "Request failed");
    } finally {
      setRunLoading(false);
    }
  }, [apiKey, useOpenAI, spec, yaml]);

  useEffect(() => {
    const result = parseAndValidateSpec(yaml);
    setParseError(result.parseError ?? null);
    setValidationErrors(result.errors);
    setSpec(result.spec ?? null);
  }, [yaml]);

  const isValid = validationErrors.length === 0 && !parseError && spec !== null;

  const handleExampleChange = useCallback(
    (e: React.ChangeEvent<HTMLSelectElement>) => {
      const id = e.target.value;
      const example = EXAMPLES.find((ex) => ex.id === id);
      if (example) {
        setSelectedExampleId(example.id);
        setYaml(example.yaml);
      }
    },
    []
  );

  return (
    <div className="flex h-screen flex-col">
      <header className="border-border flex shrink-0 items-center justify-between border-b bg-surface-muted px-4 py-3">
        <div>
          <h1 className="text-lg font-semibold tracking-tight">
            Open Agent Spec
          </h1>
          <p className="text-xs text-[var(--text-muted)]">
            Declarative standard for defining AI agents
          </p>
        </div>
        <div className="flex flex-col items-end gap-1 sm:flex-row sm:items-center sm:gap-3">
          <select
            value={selectedExampleId}
            onChange={handleExampleChange}
            className="rounded border border-[var(--border)] bg-surface px-3 py-2 text-sm text-[var(--text)] focus:border-[var(--accent)] focus:outline-none"
            aria-label="Example spec"
          >
            {EXAMPLES.map((ex) => (
              <option key={ex.id} value={ex.id}>
                {ex.label}
              </option>
            ))}
          </select>
          <select
            value={targetLang}
            onChange={(e) => setTargetLang(e.target.value as TargetLanguage)}
            className="rounded border border-[var(--border)] bg-surface px-3 py-2 text-sm text-[var(--text)] focus:border-[var(--accent)] focus:outline-none"
            aria-label="Code generation language"
          >
            <option value="python">Python</option>
            <option value="typescript">TypeScript</option>
          </select>
          <GenerateButton
            onClick={handleGenerate}
            isValid={isValid}
            disabled={(!spec && validationErrors.length > 0) || generateLoading}
            loading={generateLoading}
          />
          <button
            type="button"
            onClick={handleRunFromSpec}
            disabled={!spec || runLoading}
            className="rounded bg-[var(--accent)] px-4 py-2 text-sm font-medium text-white transition hover:opacity-90 disabled:opacity-50"
            aria-label="Run first task directly from spec"
          >
            {runLoading ? "Running…" : "Run From Spec"}
          </button>
          <div className="flex flex-col items-end gap-1 text-right text-[10px] text-[var(--text-muted)] sm:ml-2">
            <div className="flex items-center justify-end gap-2">
              <button
                type="button"
                onClick={() => setUseOpenAI((v) => !v)}
                className={`rounded-full border px-3 py-1 text-[10px] font-semibold transition ${
                  useOpenAI
                    ? "border-emerald-600 bg-emerald-600 text-white shadow-sm hover:bg-emerald-500"
                    : "border-emerald-600 bg-transparent text-emerald-600 hover:bg-emerald-600 hover:text-white"
                }`}
                aria-pressed={useOpenAI}
                aria-label="Toggle real OpenAI vs mock demo"
              >
                {useOpenAI ? "Real OpenAI (API key required)" : "Demo mode (no API key)"}
              </button>
            </div>
            {useOpenAI && (
              <input
                type="password"
                placeholder="sk-..."
                value={apiKey}
                onChange={(e) => setApiKey(e.target.value)}
                className="mt-1 w-full max-w-xs rounded border px-2 py-1 text-[11px] focus:border-[var(--accent)] focus:outline-none"
                style={{
                  borderColor: "var(--border)",
                  background: "var(--surface)",
                  color: "var(--text)",
                }}
                aria-label="OpenAI API key"
              />
            )}
            {runError && (
              <div className="mt-1 max-w-xs text-[10px] text-red-500" role="alert">
                {runError}
              </div>
            )}
          </div>
        </div>
      </header>

      {generateError && (
        <div className="shrink-0 border-b border-amber-500/30 bg-amber-500/5 px-4 py-2 text-sm text-amber-600 dark:text-amber-400">
          {generateError} — showing in-browser scaffold.
        </div>
      )}

      {(parseError || validationErrors.length > 0) && (
        <div className="border-border shrink-0 border-b bg-red-500/5 px-4 py-2 font-mono text-sm text-red-400">
          {parseError && <div>Parse: {parseError}</div>}
          {validationErrors.map((e, i) => (
            <div key={i}>
              {e.path}: {e.message}
            </div>
          ))}
        </div>
      )}

      <main className="min-h-0 flex-1">
        <SplitScreen
          left={
            <div className="flex h-full flex-col">
              <div className="border-b border-[var(--border)] bg-surface-muted px-3 py-2 text-xs text-[var(--text-muted)]">
                <span className="font-semibold">Spec as source of truth.</span>{" "}
                Edit the Open Agent Spec YAML here, then either{" "}
                <span className="font-semibold">generate code</span> or{" "}
                <span className="font-semibold">run the first task directly from this spec</span>.
              </div>
              <div className="min-h-0 flex-1">
                <YamlEditor value={yaml} onChange={setYaml} height="100%" />
              </div>
            </div>
          }
          right={
            <ResultTabs active={activeTab} onSelect={setActiveTab}>
              {activeTab === "code" && (
                <GeneratedCodeTab
                  code={generatedCode}
                  language={targetLang}
                  generatedFiles={generatedFiles ?? undefined}
                  emptyMessage={generateLoading ? "Generating…" : "Generate agent to see scaffold."
                  }
                />
              )}
              {activeTab === "logs" && (
                <LogsTab logs={executionResult?.logs ?? []} />
              )}
              {activeTab === "output" && (
                <OutputTab result={executionResult} />
              )}
            </ResultTabs>
          }
          leftLabel="Spec (YAML)"
          rightLabel="Results"
        />
      </main>
    </div>
  );
}

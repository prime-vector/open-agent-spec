"use client";

import React, { useCallback, useState, useEffect } from "react";
import { SplitScreen } from "../components/layout/SplitScreen";
import { ResultTabs, type ResultTabId } from "../components/layout/ResultTabs";
import { YamlEditor } from "../components/editor/YamlEditor";
import { GenerateButton } from "../components/playground/GenerateButton";
import { RunButton } from "../components/playground/RunButton";
import { RunWithOpenAIModal } from "../components/playground/RunWithOpenAIModal";
import { GeneratedCodeTab } from "../components/output/GeneratedCodeTab";
import { LogsTab } from "../components/output/LogsTab";
import { OutputTab } from "../components/output/OutputTab";
import { parseAndValidateSpec } from "../lib/spec/parse";
import { generateScaffold, type TargetLanguage } from "../lib/codegen/generate";
import { runFirstTaskWithSampleInput } from "../lib/runtime/mockRuntime";
import type { OpenAgentSpec } from "../lib/spec/types";
import type { ExecutionResult } from "../lib/runtime/types";
import { DEFAULT_SPEC_YAML } from "../content/defaultSpec";

export default function PlaygroundPage() {
  const [yaml, setYaml] = useState(DEFAULT_SPEC_YAML);
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
  const [openAIModalOpen, setOpenAIModalOpen] = useState(false);

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

  const handleOpenAIRunResult = useCallback((result: unknown) => {
    setExecutionResult(result as ExecutionResult);
    setActiveTab("output");
  }, []);

  useEffect(() => {
    const result = parseAndValidateSpec(yaml);
    setParseError(result.parseError ?? null);
    setValidationErrors(result.errors);
    setSpec(result.spec ?? null);
  }, [yaml]);

  const isValid = validationErrors.length === 0 && !parseError && spec !== null;

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
        <div className="flex flex-wrap items-center justify-end gap-2 sm:gap-3">
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
          <RunButton onClick={handleRun} disabled={!spec} />
          <button
            type="button"
            onClick={() => setOpenAIModalOpen(true)}
            disabled={!spec}
            className="hidden rounded border border-[var(--accent)] bg-transparent px-4 py-2 text-sm font-medium text-[var(--accent)] transition hover:bg-[var(--accent)] hover:text-white disabled:opacity-50 sm:inline-flex"
            aria-label="Try with OpenAI (rate limited)"
          >
            Try with OpenAI
          </button>
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
            <YamlEditor
              value={yaml}
              onChange={setYaml}
              height="100%"
            />
          }
          right={
            <ResultTabs active={activeTab} onSelect={setActiveTab}>
              {activeTab === "code" && (
                <GeneratedCodeTab
                  code={generatedCode}
                  language={targetLang}
                  generatedFiles={generatedFiles ?? undefined}
                  emptyMessage={generateLoading ? "Generating…" : "Generate agent to see scaffold."}
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

      <RunWithOpenAIModal
        isOpen={openAIModalOpen}
        onClose={() => setOpenAIModalOpen(false)}
        onResult={handleOpenAIRunResult}
        yaml={yaml}
        disabled={!spec}
      />
    </div>
  );
}

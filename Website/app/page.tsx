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
import { DEFAULT_SPEC_YAML } from "../content/defaultSpec";

export default function PlaygroundPage() {
  const [yaml, setYaml] = useState(DEFAULT_SPEC_YAML);
  const [parseError, setParseError] = useState<string | null>(null);
  const [validationErrors, setValidationErrors] = useState<{ path: string; message: string }[]>([]);
  const [spec, setSpec] = useState<OpenAgentSpec | null>(null);
  const [generatedCode, setGeneratedCode] = useState<string | null>(null);
  const [targetLang, setTargetLang] = useState<TargetLanguage>("python");
  const [activeTab, setActiveTab] = useState<ResultTabId>("code");
  const [executionResult, setExecutionResult] = useState<ExecutionResult | null>(null);

  const handleGenerate = useCallback(() => {
    if (!spec) return;
    setGeneratedCode(generateScaffold(spec, targetLang));
    setActiveTab("code");
  }, [spec, targetLang]);

  const handleRun = useCallback(() => {
    if (!spec) return;
    const result = runFirstTaskWithSampleInput(spec);
    setExecutionResult(result ?? null);
    setActiveTab(result ? "output" : "logs");
  }, [spec]);

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
        <div className="flex items-center gap-3">
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
            disabled={!spec && validationErrors.length > 0}
          />
          <RunButton onClick={handleRun} disabled={!spec} />
        </div>
      </header>

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

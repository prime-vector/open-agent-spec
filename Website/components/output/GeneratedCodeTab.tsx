"use client";

import React, { useState } from "react";

interface GeneratedCodeTabProps {
  code: string | null;
  language: "python" | "typescript";
  emptyMessage?: string;
  /** When using real generator: optional other files to show (e.g. README, requirements) */
  generatedFiles?: Record<string, string>;
}

export function GeneratedCodeTab({
  code,
  language,
  emptyMessage = "Generate agent to see scaffold.",
  generatedFiles,
}: GeneratedCodeTabProps) {
  const fileMap: Record<string, string> = {};
  if (code?.trim()) fileMap["agent.py"] = code;
  if (generatedFiles) {
    if (generatedFiles.readme) fileMap["README.md"] = generatedFiles.readme;
    if (generatedFiles.requirementsTxt) fileMap["requirements.txt"] = generatedFiles.requirementsTxt;
    if (generatedFiles.envExample) fileMap[".env.example"] = generatedFiles.envExample;
  }
  const fileKeys = Object.keys(fileMap);
  const [selectedFile, setSelectedFile] = useState(fileKeys[0] ?? "agent.py");

  if (fileKeys.length === 0) {
    return (
      <div className="flex h-full items-center justify-center p-6 text-sm text-[var(--pane-text-muted)]">
        {emptyMessage}
      </div>
    );
  }

  const displayCode = fileMap[selectedFile] ?? code ?? "";

  return (
    <div className="flex h-full flex-col">
      {fileKeys.length > 1 && (
        <div className="flex shrink-0 gap-2 border-b p-2" style={{ borderColor: "var(--pane-border)" }}>
          <label className="text-xs text-[var(--pane-text-muted)]" htmlFor="file-select">
            File:
          </label>
          <select
            id="file-select"
            value={selectedFile}
            onChange={(e) => setSelectedFile(e.target.value)}
            className="rounded border bg-[var(--pane-surface)] px-2 py-1 text-sm"
            style={{ borderColor: "var(--pane-border)", color: "var(--pane-text)" }}
          >
            {fileKeys.map((k) => (
              <option key={k} value={k}>
                {k}
              </option>
            ))}
          </select>
        </div>
      )}
      <pre className="min-h-0 flex-1 overflow-auto whitespace-pre-wrap break-words p-4 font-mono text-sm text-[var(--pane-text)]">
        <code>{displayCode}</code>
      </pre>
    </div>
  );
}

"use client";

import React from "react";

interface GeneratedCodeTabProps {
  code: string | null;
  language: "python" | "typescript";
  emptyMessage?: string;
}

export function GeneratedCodeTab({
  code,
  language,
  emptyMessage = "Generate agent to see scaffold.",
}: GeneratedCodeTabProps) {
  if (!code?.trim()) {
    return (
      <div className="flex h-full items-center justify-center p-6 text-sm text-[var(--pane-text-muted)]">
        {emptyMessage}
      </div>
    );
  }

  return (
    <pre className="h-full overflow-auto whitespace-pre-wrap break-words p-4 font-mono text-sm text-[var(--pane-text)]">
      <code>{code}</code>
    </pre>
  );
}

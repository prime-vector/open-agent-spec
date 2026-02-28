"use client";

import React from "react";
import type { ExecutionResult } from "@/lib/runtime/types";

interface OutputTabProps {
  result: ExecutionResult | null;
  emptyMessage?: string;
}

export function OutputTab({
  result,
  emptyMessage = "Run agent to see output.",
}: OutputTabProps) {
  if (!result) {
    return (
      <div className="flex h-full items-center justify-center p-6 text-sm text-[var(--pane-text-muted)]">
        {emptyMessage}
      </div>
    );
  }

  const indicators = [
    result.memoryUsed && { label: "Memory", value: "injected" },
    result.toolCalls.length > 0 && {
      label: "Tools",
      value: result.toolCalls.map((t) => t.toolId).join(", "),
    },
    { label: "Engine", value: result.engine },
    { label: "Task", value: result.taskName },
    { label: "Duration", value: `${result.durationMs}ms` },
  ].filter(Boolean) as { label: string; value: string }[];

  return (
    <div className="h-full overflow-auto p-4">
      <div className="mb-4 flex flex-wrap gap-2">
        {indicators.map(({ label, value }) => (
          <span
            key={label}
            className="rounded border px-2 py-1 text-xs"
            style={{
              borderColor: "var(--pane-border)",
              background: "var(--pane-surface-muted)",
              color: "var(--pane-text-muted)",
            }}
          >
            <span className="font-medium">{label}:</span> {value}
          </span>
        ))}
      </div>
      {result.error && (
        <pre className="mb-4 rounded border border-red-500/50 bg-red-500/10 p-3 text-sm text-red-400">
          {result.error}
        </pre>
      )}
      {result.output && Object.keys(result.output).length > 0 && (
        <div>
          <div className="mb-2 text-xs font-medium uppercase text-[var(--pane-text-muted)]">
            Structured output
          </div>
          <pre
            className="rounded border p-4 font-mono text-sm"
            style={{
              borderColor: "var(--pane-border)",
              background: "var(--pane-surface-muted)",
              color: "var(--pane-text)",
            }}
          >
            {JSON.stringify(result.output, null, 2)}
          </pre>
        </div>
      )}
    </div>
  );
}

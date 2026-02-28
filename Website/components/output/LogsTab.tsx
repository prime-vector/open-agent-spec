"use client";

import React from "react";
import type { LogEntry, LogKind } from "@/lib/runtime/types";

const kindStyles: Record<LogKind, string> = {
  prompt: "text-amber-400",
  memory: "text-violet-400",
  tool: "text-cyan-400",
  engine: "text-emerald-400",
  response: "text-green-400",
  step: "text-blue-400",
  info: "text-[var(--pane-text-muted)]",
  error: "text-red-400",
};

interface LogsTabProps {
  logs: LogEntry[];
  emptyMessage?: string;
}

export function LogsTab({ logs, emptyMessage = "Run agent to see logs." }: LogsTabProps) {
  if (!logs.length) {
    return (
      <div className="flex h-full items-center justify-center p-6 text-sm text-[var(--pane-text-muted)]">
        {emptyMessage}
      </div>
    );
  }

  return (
    <div className="h-full overflow-auto p-4 font-mono text-xs">
      {logs.map((entry, i) => (
        <div
          key={i}
          className="border-b border-dashed py-2 last:border-0"
          style={{ borderColor: "var(--pane-border)" }}
        >
          <span className="text-[var(--pane-text-muted)]">{entry.timestamp}</span>{" "}
          <span className={kindStyles[entry.kind] ?? ""}>[{entry.kind}]</span>{" "}
          {entry.message}
          {entry.detail && (
            <pre className="mt-1 overflow-x-auto text-[var(--pane-text-muted)]">
              {JSON.stringify(entry.detail, null, 2)}
            </pre>
          )}
        </div>
      ))}
    </div>
  );
}

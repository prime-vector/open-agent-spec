"use client";

import React from "react";

interface SplitScreenProps {
  left: React.ReactNode;
  right: React.ReactNode;
  leftLabel?: string;
  rightLabel?: string;
}

export function SplitScreen({ left, right, leftLabel, rightLabel }: SplitScreenProps) {
  return (
    <div className="flex h-full w-full flex-1 flex-col md:flex-row">
      <section
        className="flex min-h-0 flex-1 flex-col border-r"
        style={{
          background: "var(--pane-surface)",
          color: "var(--pane-text)",
          borderColor: "var(--pane-border)",
        }}
        aria-label={leftLabel ?? "Editor"}
      >
        {leftLabel && (
          <div
            className="border-b px-3 py-2 text-xs font-medium uppercase tracking-wider"
            style={{
              background: "var(--pane-surface-muted)",
              borderColor: "var(--pane-border)",
              color: "var(--pane-text-muted)",
            }}
          >
            {leftLabel}
          </div>
        )}
        <div className="min-h-0 flex-1 overflow-hidden">{left}</div>
      </section>
      <section
        className="flex min-h-0 flex-1 flex-col"
        style={{
          background: "var(--pane-surface)",
          color: "var(--pane-text)",
        }}
        aria-label={rightLabel ?? "Output"}
      >
        {rightLabel && (
          <div
            className="border-b px-3 py-2 text-xs font-medium uppercase tracking-wider"
            style={{
              background: "var(--pane-surface-muted)",
              borderColor: "var(--pane-border)",
              color: "var(--pane-text-muted)",
            }}
          >
            {rightLabel}
          </div>
        )}
        <div className="min-h-0 flex-1 overflow-hidden">{right}</div>
      </section>
    </div>
  );
}

"use client";

import React from "react";

export type ResultTabId = "code" | "logs" | "output";

interface ResultTabsProps {
  active: ResultTabId;
  onSelect: (id: ResultTabId) => void;
  children: React.ReactNode;
}

const TABS: { id: ResultTabId; label: string }[] = [
  { id: "code", label: "Generated Code" },
  { id: "logs", label: "Logs" },
  { id: "output", label: "Output" },
];

export function ResultTabs({ active, onSelect, children }: ResultTabsProps) {
  return (
    <div className="flex h-full flex-col">
      <div className="flex border-b" style={{ borderColor: "var(--pane-border)" }}>
        {TABS.map(({ id, label }) => (
          <button
            key={id}
            type="button"
            onClick={() => onSelect(id)}
            className="-mb-px border-b-2 px-4 py-2 text-sm font-medium transition focus:outline-none focus:ring-1 focus:ring-[var(--accent)]"
            style={{
              borderBottomColor: active === id ? "var(--accent)" : "transparent",
              color: active === id ? "var(--accent)" : "var(--pane-text-muted)",
            }}
            aria-selected={active === id}
            role="tab"
          >
            {label}
          </button>
        ))}
      </div>
      <div className="min-h-0 flex-1 overflow-hidden">{children}</div>
    </div>
  );
}

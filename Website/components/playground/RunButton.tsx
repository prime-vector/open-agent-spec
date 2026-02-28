"use client";

import React from "react";

interface RunButtonProps {
  onClick: () => void;
  disabled?: boolean;
}

export function RunButton({ onClick, disabled }: RunButtonProps) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      className="rounded bg-[var(--accent)] px-4 py-2 text-sm font-medium text-white transition hover:opacity-90 disabled:opacity-50"
      aria-label="Run agent with mock execution"
    >
      Run Agent
    </button>
  );
}

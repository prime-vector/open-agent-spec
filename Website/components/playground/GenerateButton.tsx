"use client";

import React from "react";

interface GenerateButtonProps {
  onClick: () => void;
  disabled?: boolean;
  isValid?: boolean;
  loading?: boolean;
}

export function GenerateButton({ onClick, disabled, isValid, loading }: GenerateButtonProps) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      className="rounded border border-[var(--accent)] bg-transparent px-4 py-2 text-sm font-medium text-[var(--accent)] transition hover:bg-[var(--accent)] hover:text-white disabled:opacity-50 disabled:hover:bg-transparent disabled:hover:text-[var(--accent)]"
      aria-label="Generate agent code from spec"
    >
      {loading ? "Generating…" : isValid === false ? "Fix errors first" : "Generate Code"}
    </button>
  );
}

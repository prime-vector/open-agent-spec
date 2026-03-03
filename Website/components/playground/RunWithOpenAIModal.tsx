"use client";

import React, { useState } from "react";

interface RunWithOpenAIModalProps {
  isOpen: boolean;
  onClose: () => void;
  onResult: (result: unknown) => void;
  yaml: string;
  disabled?: boolean;
}

export function RunWithOpenAIModal({
  isOpen,
  onClose,
  onResult,
  yaml,
  disabled,
}: RunWithOpenAIModalProps) {
  const [apiKey, setApiKey] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleRun = async () => {
    setError(null);
    setLoading(true);
    try {
      const res = await fetch("/api/run-demo", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          yaml,
          apiKey: apiKey.trim() || undefined,
        }),
      });
      const data = await res.json();
      if (!res.ok) {
        if (data.rateLimited) {
          setError(data.error ?? "Rate limit reached.");
        } else {
          setError(data.error ?? `Request failed (${res.status})`);
        }
        return;
      }
      onResult(data);
      onClose();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Request failed");
    } finally {
      setLoading(false);
    }
  };

  if (!isOpen) return null;

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4"
      role="dialog"
      aria-modal="true"
      aria-labelledby="run-openai-title"
    >
      <div
        className="w-full max-w-md rounded-lg border bg-white p-6 shadow-xl dark:bg-[var(--pane-surface)]"
        style={{ borderColor: "var(--pane-border)" }}
      >
        <h2 id="run-openai-title" className="mb-2 text-lg font-semibold">
          Try with OpenAI
        </h2>
        <p className="mb-4 text-sm text-[var(--pane-text-muted)]">
          Run the first task once with a real model. Rate limit: 1 run per IP per day. Leave the key
          blank to see a mock result.
        </p>
        <label className="mb-2 block text-sm font-medium" htmlFor="run-openai-key">
          OpenAI API key (optional)
        </label>
        <input
          id="run-openai-key"
          type="password"
          placeholder="sk-..."
          value={apiKey}
          onChange={(e) => setApiKey(e.target.value)}
          className="mb-4 w-full rounded border px-3 py-2 text-sm focus:border-[var(--accent)] focus:outline-none"
          style={{
            borderColor: "var(--pane-border)",
            background: "var(--pane-surface-muted)",
            color: "var(--pane-text)",
          }}
        />
        {error && (
          <p className="mb-4 text-sm text-red-500" role="alert">
            {error}
          </p>
        )}
        <div className="flex justify-end gap-2">
          <button
            type="button"
            onClick={onClose}
            className="rounded border px-4 py-2 text-sm font-medium"
            style={{ borderColor: "var(--pane-border)" }}
          >
            Cancel
          </button>
          <button
            type="button"
            onClick={handleRun}
            disabled={disabled || loading}
            className="rounded bg-[var(--accent)] px-4 py-2 text-sm font-medium text-white hover:opacity-90 disabled:opacity-50"
          >
            {loading ? "Running…" : "Run"}
          </button>
        </div>
      </div>
    </div>
  );
}

"use client";

import { useCallback, useState } from "react";

const PIP = "pip install open-agent-spec";
const BREW = `brew tap prime-vector/homebrew-prime-vector
brew install open-agent-spec`;

function CopyButton({ text, label }: { text: string; label: string }) {
  const [copied, setCopied] = useState(false);

  const copy = useCallback(async () => {
    try {
      await navigator.clipboard.writeText(text.trim());
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      setCopied(false);
    }
  }, [text]);

  return (
    <button
      type="button"
      onClick={copy}
      className="shrink-0 rounded-md border border-stone-400/40 bg-stone-100/80 px-3 py-1.5 text-[11px] font-medium text-stone-700 transition hover:bg-stone-200/90 active:scale-[0.98] sm:text-xs"
      aria-label={`Copy ${label}`}
    >
      {copied ? "Copied" : "Copy"}
    </button>
  );
}

export default function InstallCard() {
  return (
    <div className="w-full max-w-2xl rounded-2xl border border-stone-300/60 bg-stone-900 px-4 py-4 shadow-xl sm:px-5 sm:py-5">
      <div className="mb-3 flex flex-col gap-1 sm:flex-row sm:items-baseline sm:justify-between">
        <span className="text-[10px] font-semibold uppercase tracking-[0.2em] text-stone-500">
          Install
        </span>
        <span className="font-serif text-lg text-stone-100 sm:text-xl">
          Available in the terminal
        </span>
      </div>
      <div className="space-y-3">
        <div>
          <div className="mb-1 text-[10px] font-medium uppercase tracking-wide text-stone-500">
            pip
          </div>
          <div className="flex flex-col gap-2 rounded-lg bg-stone-950/80 p-3 sm:flex-row sm:items-center sm:justify-between">
            <code className="break-all font-mono text-[12px] text-stone-200 sm:text-sm">
              {PIP}
            </code>
            <CopyButton text={PIP} label="pip command" />
          </div>
        </div>
        <div>
          <div className="mb-1 text-[10px] font-medium uppercase tracking-wide text-stone-500">
            Homebrew
          </div>
          <div className="flex flex-col gap-2 rounded-lg bg-stone-950/80 p-3 sm:flex-row sm:items-start sm:justify-between">
            <code className="whitespace-pre-wrap break-all font-mono text-[12px] text-stone-200 sm:text-sm">
              {BREW}
            </code>
            <CopyButton text={BREW} label="brew commands" />
          </div>
        </div>
      </div>
      <p className="mt-3 text-[11px] text-stone-500">
        Then run <code className="rounded bg-stone-800 px-1 py-0.5 text-stone-300">oa --version</code>
      </p>
    </div>
  );
}

"use client";

import { useCallback, useState } from "react";

const PIP = "pip install open-agent-spec";
const BREW = `brew tap prime-vector/homebrew-prime-vector
brew install open-agent-spec`;
const NPM = "npm install -g @prime-vector/open-agent-spec";
const NPX = "npx @prime-vector/open-agent-spec run --spec agent.yaml";

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

type Tab = "pip" | "brew" | "npm";

export default function InstallCard() {
  const [tab, setTab] = useState<Tab>("pip");

  return (
    <div className="w-full max-w-2xl rounded-2xl border border-stone-300/60 bg-stone-900 px-4 py-4 shadow-xl sm:px-5 sm:py-5">
      <div className="mb-3 flex flex-col gap-1 sm:flex-row sm:items-baseline sm:justify-between">
        <span className="text-[10px] font-semibold uppercase tracking-[0.2em] text-stone-500">
          Install
        </span>
        <span className="font-serif text-lg text-stone-100 sm:text-xl">
          Python · Node.js · Homebrew
        </span>
      </div>

      {/* Tab switcher */}
      <div className="mb-3 flex gap-1">
        {(["pip", "brew", "npm"] as Tab[]).map((t) => (
          <button
            key={t}
            type="button"
            onClick={() => setTab(t)}
            className={`rounded-md px-3 py-1 text-[11px] font-medium transition ${
              tab === t
                ? "bg-stone-700 text-stone-100"
                : "text-stone-500 hover:text-stone-300"
            }`}
          >
            {t === "pip" ? "pip" : t === "brew" ? "Homebrew" : "npm"}
          </button>
        ))}
      </div>

      {/* pip */}
      {tab === "pip" && (
        <div className="space-y-2">
          <div className="flex flex-col gap-2 rounded-lg bg-stone-950/80 p-3 sm:flex-row sm:items-center sm:justify-between">
            <code className="break-all font-mono text-[12px] text-stone-200 sm:text-sm">
              {PIP}
            </code>
            <CopyButton text={PIP} label="pip command" />
          </div>
          <p className="text-[11px] text-stone-500">
            Then run{" "}
            <code className="rounded bg-stone-800 px-1 py-0.5 text-stone-300">
              oa --version
            </code>
          </p>
        </div>
      )}

      {/* Homebrew */}
      {tab === "brew" && (
        <div className="space-y-2">
          <div className="flex flex-col gap-2 rounded-lg bg-stone-950/80 p-3 sm:flex-row sm:items-start sm:justify-between">
            <code className="whitespace-pre-wrap break-all font-mono text-[12px] text-stone-200 sm:text-sm">
              {BREW}
            </code>
            <CopyButton text={BREW} label="brew commands" />
          </div>
          <p className="text-[11px] text-stone-500">
            Then run{" "}
            <code className="rounded bg-stone-800 px-1 py-0.5 text-stone-300">
              oa --version
            </code>
          </p>
        </div>
      )}

      {/* npm */}
      {tab === "npm" && (
        <div className="space-y-2">
          <div className="flex flex-col gap-2 rounded-lg bg-stone-950/80 p-3 sm:flex-row sm:items-center sm:justify-between">
            <code className="break-all font-mono text-[12px] text-stone-200 sm:text-sm">
              {NPM}
            </code>
            <CopyButton text={NPM} label="npm install command" />
          </div>
          <p className="text-[11px] text-stone-500">
            Or run without installing:
          </p>
          <div className="flex flex-col gap-2 rounded-lg bg-stone-950/80 p-3 sm:flex-row sm:items-center sm:justify-between">
            <code className="break-all font-mono text-[12px] text-stone-200 sm:text-sm">
              {NPX}
            </code>
            <CopyButton text={NPX} label="npx command" />
          </div>
          <p className="text-[11px] text-stone-500">
            Node.js 18+ · no Python required ·{" "}
            <a
              href="https://www.npmjs.com/package/@prime-vector/open-agent-spec"
              target="_blank"
              rel="noopener noreferrer"
              className="text-stone-400 underline hover:text-stone-200"
            >
              npmjs.com ↗
            </a>
          </p>
        </div>
      )}
    </div>
  );
}

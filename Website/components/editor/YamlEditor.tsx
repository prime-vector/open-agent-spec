"use client";

import React, { useCallback } from "react";
import dynamic from "next/dynamic";

const MonacoEditor = dynamic(() => import("@monaco-editor/react"), { ssr: false });

interface YamlEditorProps {
  value: string;
  onChange: (value: string) => void;
  height?: string;
  readOnly?: boolean;
}

export function YamlEditor({
  value,
  onChange,
  height = "100%",
  readOnly = false,
}: YamlEditorProps) {
  const handleChange = useCallback(
    (val: string | undefined) => {
      onChange(val ?? "");
    },
    [onChange]
  );

  return (
    <div className="h-full w-full" style={{ minHeight: 300 }}>
      <MonacoEditor
        height={height}
        language="yaml"
        value={value}
        onChange={handleChange}
        theme="vs"
        options={{
          readOnly,
          minimap: { enabled: false },
          fontSize: 13,
          lineNumbers: "on",
          scrollBeyondLastLine: false,
          padding: { top: 12 },
          wordWrap: "on",
          automaticLayout: true,
        }}
        loading={
          <div
            className="flex h-full items-center justify-center"
            style={{
              background: "var(--pane-surface)",
              color: "var(--pane-text-muted)",
            }}
          >
            Loading editor…
          </div>
        }
      />
    </div>
  );
}

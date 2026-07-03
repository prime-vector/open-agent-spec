---
name: document-summariser
description: >
  Produce a structured summary (one paragraph + key points) of any document
  or text. Use this whenever the user asks to summarise a file, article,
  changelog, meeting notes, or any other body of text and expects a
  consistent, structured result.
---

# Document Summariser

This skill delegates to a typed, validated **Open Agent Spec** task instead
of improvising summarisation behaviour. Do not summarise the text yourself —
execute the spec so the output shape is guaranteed.

## Prerequisites

The `oa` CLI must be available (`pipx install open-agent-spec` or
`npm i -g @prime-vector/open-agent-spec`) with an `OPENAI_API_KEY` set.

## How to use

1. Obtain the text to summarise (read the file yourself if the user gave a
   path).
2. Run the spec from this skill's directory, passing the text as input:

```bash
oa run --spec spec.yaml --task summarise \
  --input '{"text": "<the document text>"}' --quiet
```

For long documents, write the input to a JSON file first and pass the path:

```bash
oa run --spec spec.yaml --task summarise --input input.json --quiet
```

3. The result is guaranteed by the spec's output schema to be a JSON object
   with exactly:
   - `summary` — one-paragraph summary (string)
   - `key_points` — key takeaways (array of strings)

Relay `summary` and `key_points` to the user. If the command exits non-zero,
report the structured error (`code`, `stage`) — do not retry with your own
improvised summary.

## Why this is a spec and not instructions

The behaviour lives in `spec.yaml`: typed input/output schemas, a pinned
model, explicit prompts, and structured errors. That makes this skill's
behaviour identical across every agent and machine that invokes it —
something prose instructions cannot guarantee. See
https://openagentspec.dev for the standard.

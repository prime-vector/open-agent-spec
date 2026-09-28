# OpenTelemetry execution evidence

OA can emit standard OpenTelemetry traces for a run without changing the
portable agent contract or requiring a vendor-specific observability product.
Install the optional extra when you need it:

```bash
pip install 'open-agent-spec[otel]'
```

Then opt in for a run:

```bash
export OTEL_SERVICE_NAME=oa-example
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318
oa run --spec examples/telemetry/agent.yaml --input examples/telemetry/input.json \
  --telemetry
```

OA uses the official Python OpenTelemetry SDK and the OTLP/HTTP exporter. The
destination, headers, TLS, batching and resource configuration remain standard
`OTEL_*` deployment configuration; they are not fields in an OA spec.

## Trace shape

An execution produces an `invoke_agent <agent>` root span. Task, model and tool
observations are children of that span:

```text
invoke_agent reviewer
└── execute_task review
    ├── chat gpt-4o-mini
    ├── execute_tool github.read
    ├── oa.sandbox.decision
    ├── oa.validation
    └── oa.contract.evaluate
```

Model spans use the OpenTelemetry GenAI semantic convention attributes where
available, including:

| Attribute | Meaning |
| --- | --- |
| `gen_ai.operation.name` | `invoke_agent`, `chat`, or `execute_tool` |
| `gen_ai.provider.name` | Provider/engine selected by the spec |
| `gen_ai.request.model` | Requested model identifier |
| `gen_ai.usage.input_tokens` | Input tokens reported by the provider |
| `gen_ai.usage.output_tokens` | Output tokens reported by the provider |
| `gen_ai.tool.name` | Observed tool identity |
| `gen_ai.tool.call.id` | Provider call identifier, when available |

OA adds only facts that OA can know and evaluate itself:

| Attribute | Meaning |
| --- | --- |
| `oa.spec.name` | Agent name from the declaration |
| `oa.spec.version` | `open_agent_spec` declaration value |
| `oa.spec.hash` | SHA-256 of canonical JSON for the parsed effective spec |
| `oa.task.name` | Task being executed |
| `oa.task.declared_tools` | Tools declared on that task |
| `oa.contract.enabled` / `oa.contract.result` | BCE presence and deterministic result |
| `oa.sandbox.enabled` / `oa.sandbox.result` | IIS presence and allow/block decision |
| `oa.validation.kind` / `oa.validation.result` | Input/output schema observation |
| `oa.conformance.status` / `oa.conformance.reason` | OA's declaration-versus-observation result |

The hash is over the parsed YAML object after YAML resolution, so formatting and
map ordering do not change identity. Delegated specs get their own span and
identity. Runtime input, CLI prompt overrides, prompts, outputs, tool arguments,
tool results, secrets and environment values are not included in telemetry.

## Conformance outcomes

OA does not call every error a policy violation:

- `conformant` means the deterministic observations completed within the
  declaration.
- `violation_prevented` means an undeclared or sandbox-forbidden action was
  attempted and stopped before I/O.
- `nonconformant` means a declared output boundary failed, such as output JSON
  schema validation or a behavioural contract.
- `indeterminate` means execution failed for a provider/runtime reason and OA
  cannot make a contract claim from that failure.

This keeps a trace useful to an external backend without requiring that backend
to understand OA's internal execution model.

## Privacy and configuration

Telemetry is metadata-only by design. There is no content-capture flag in this
initial implementation: prompts, user input, model responses, tool arguments,
tool results, secrets and environment values are never emitted by the adapter.
The `oa.telemetry.content_capture=false` attribute makes that default explicit.

When `--telemetry` is absent, OA uses a no-op adapter and does not import the
OpenTelemetry SDK. Library callers can pass a custom `TelemetryAdapter` to
`run_task_from_spec` if they already own SDK setup; this keeps OA from replacing
an application's tracer provider.

## Library use

```python
from oas_cli.runner import run_task_from_file
from oas_cli.telemetry import create_telemetry

telemetry = create_telemetry(enabled=True)
try:
    result = run_task_from_file("agent.yaml", telemetry=telemetry)
finally:
    telemetry.close()
```

For a production process, configure the SDK/exporter once and pass an
`OpenTelemetryAdapter(tracer_provider=your_provider,
configure_otlp=False)` instead of creating one per request.

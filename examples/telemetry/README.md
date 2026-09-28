# OpenTelemetry execution evidence example

This is a minimal OA run with metadata-only OTLP tracing enabled. It does not
capture prompts or responses.

## Run against an OTLP collector

Start any OpenTelemetry Collector with an OTLP/HTTP receiver. For a local
collector, the following configuration is enough to print received spans:

```yaml
receivers:
  otlp:
    protocols:
      http:
        endpoint: 0.0.0.0:4318

exporters:
  debug:
    verbosity: basic

service:
  pipelines:
    traces:
      receivers: [otlp]
      exporters: [debug]
```

With the collector listening on `localhost:4318`:

```bash
pip install 'open-agent-spec[otel]'
export OPENAI_API_KEY=...
export OTEL_SERVICE_NAME=oa-telemetry-example
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318
oa run --spec examples/telemetry/agent.yaml \
  --input examples/telemetry/input.json --telemetry
```

The resulting trace includes the model span, OA spec identity, declared tools,
schema result, and conformance status. The example uses no tools so it can be
run without any external service besides the configured model provider.

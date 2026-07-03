"""Standalone OpenTelemetry smoke test — backend reachability check.

Emits one hand-made span to the OTLP endpoint and exits. Its only job is
to prove the local Grafana/Tempo backend is reachable and ingesting
spans BEFORE we instrument the real app — so if a trace fails to show up
later, we already know the backend is not the culprit.

Run:  uv run python scripts/otel_smoke_test.py
Then: Grafana (http://localhost:3000) -> Explore -> Tempo datasource ->
      Search -> Service Name = otel-smoke-test -> Run query.

Reads OTEL_EXPORTER_OTLP_ENDPOINT from .env (e.g. http://localhost:4317).
"""
import time

from dotenv import load_dotenv
from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

# Load .env so OTEL_EXPORTER_OTLP_ENDPOINT is available to the exporter,
# which reads it from the environment. Without this, the exporter would
# fall back to its localhost:4317 default — fine locally, but explicit is
# better and mirrors how the real app will resolve the endpoint.
load_dotenv()

# service.name is the identity Tempo groups traces under and the field you
# search on in Grafana. A distinct name keeps this test isolated from the
# real app's traces.
resource = Resource.create({"service.name": "otel-smoke-test"})

provider = TracerProvider(resource=resource)
# BatchSpanProcessor buffers spans and exports on a background thread. For a
# one-shot script we rely on shutdown() below to force-flush before exit.
provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
trace.set_tracer_provider(provider)

tracer = trace.get_tracer("smoke-test")

# One parent span with a nested child, so the Tempo waterfall shows real
# structure (a span with a child) rather than a lone bar.
with tracer.start_as_current_span("hello-tempo") as parent:
    parent.set_attribute("smoke_test.note", "backend reachability check")
    with tracer.start_as_current_span("nested-work"):
        time.sleep(0.05)  # give the span a visible, non-zero duration

# Force-flush and drain the exporter. Without this the process could exit
# before the background batch thread ships the span.
provider.shutdown()
print("span sent -> service.name=otel-smoke-test (check Grafana Explore -> Tempo)")

"""OpenTelemetry tracing bootstrap for the search API.

Configures the global TracerProvider + OTLP exporter and enables
auto-instrumentation for the four network clients this service uses:
FastAPI (inbound HTTP), httpx (outbound LLM / TMDB / proxy calls),
psycopg v3 (Postgres), and redis.asyncio (cache).

Design goals:
- One-time setup, called once from api/main.py after the app is built.
- Endpoint + service identity come from env vars (OTEL_EXPORTER_OTLP_*,
  OTEL_SERVICE_NAME), so switching local -> Grafana Cloud on EC2 is a
  config change, never a code change.
- Idempotent: guarded so uvicorn --reload / repeated imports don't
  double-instrument (which raises for FastAPI or double-wraps the clients).

Not covered here: the Qdrant vector search. Its async client talks gRPC,
which auto-instrumentation doesn't wrap cleanly; that timing is picked up
by the manual vector-search span added in the next phase.
"""
import os

from dotenv import load_dotenv
from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.instrumentation.psycopg import PsycopgInstrumentor
from opentelemetry.instrumentation.redis import RedisInstrumentor

# Module-level guard against double initialization. uvicorn --reload and
# test clients can import api.main more than once; instrumenting a client
# twice raises (FastAPI) or silently double-wraps (httpx/psycopg/redis).
_configured = False


def setup_tracing(app) -> None:
    """Initialize tracing and turn on auto-instrumentation.

    Call once, after the FastAPI app is constructed and before it serves
    traffic. Passing the app instance (vs. global FastAPI patching) makes
    each inbound request the root span that every downstream DB / cache /
    HTTP span hangs off of, yielding one trace per request.

    Args:
        app: the FastAPI application instance to instrument.
    """
    global _configured
    if _configured:
        return

    # Load .env so OTEL_EXPORTER_OTLP_ENDPOINT (and OTEL_SERVICE_NAME, if set)
    # resolve on a bare `uvicorn api.main:app` run. load_dotenv does NOT
    # override already-set vars, so Docker's env_file/environment still wins
    # in the container. If neither source sets the endpoint, the exporter
    # falls back to its localhost:4317 default — correct for local dev.
    load_dotenv()

    resource = Resource.create({
        "service.name": os.getenv("OTEL_SERVICE_NAME"),
        # deployment.environment lets Tempo/Grafana separate local vs. prod
        # traces once both ship to the same backend.
        "deployment.environment": os.getenv("DEPLOY_ENV"),
    })

    provider = TracerProvider(resource=resource)
    # BatchSpanProcessor buffers spans and exports them on a background
    # thread, off the request hot path — the production-correct choice vs.
    # SimpleSpanProcessor, which exports inline and adds latency per span.
    # OTLPSpanExporter() with no args reads OTEL_EXPORTER_OTLP_ENDPOINT /
    # _PROTOCOL from the environment, so the endpoint stays pure config
    # (and an https:// endpoint auto-selects TLS for Grafana Cloud).
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    trace.set_tracer_provider(provider)

    # --- Auto-instrumentation -------------------------------------------
    # instrument_app targets THIS app instance so every request is a root
    # span. The other three patch their libraries process-wide, so one call
    # each covers all httpx clients (the shared TMDB client + LLM provider
    # SDKs that ride on httpx), the psycopg pool, and the redis pool.
    FastAPIInstrumentor.instrument_app(app)
    HTTPXClientInstrumentor().instrument()
    # capture_parameters=True adds `db.statement.parameters` (str(bound params))
    # to every psycopg query span, so otherwise-identical parameterized statements
    # (e.g. `SELECT ... FROM public.movie_card WHERE movie_id = ANY($1)` fired for
    # anchors vs candidates vs hydration) are distinguishable by their bound IDs.
    # Global switch (all queries, all endpoints); our params are movie/trait IDs
    # and filter values (no secrets/PII). High-cardinality → span-attr only, never a
    # metric label. Non-standard attribute (not OTel semconv), intended for debugging.
    PsycopgInstrumentor().instrument(capture_parameters=True)
    RedisInstrumentor().instrument()

    _configured = True

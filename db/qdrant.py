import os
from qdrant_client import AsyncQdrantClient

# gRPC transport (vs HTTP) cuts ~15-30% off per-call latency for the
# request shapes this codebase issues (single-vector and batched
# named-vector searches). Default is ON; set QDRANT_PREFER_GRPC=0 to
# fall back to HTTP for debugging. Docker-compose exposes both 6333
# (HTTP) and 6334 (gRPC); the client picks gRPC when prefer_grpc=True
# and silently falls back to HTTP if the gRPC port is unreachable.
_prefer_grpc = os.getenv("QDRANT_PREFER_GRPC", "1").lower() in {"1", "true", "yes"}
qdrant_client = AsyncQdrantClient(
    host=os.getenv("QDRANT_HOST", "localhost"),
    port=int(os.getenv("QDRANT_PORT", 6333)),
    grpc_port=int(os.getenv("QDRANT_GRPC_PORT", 6334)),
    prefer_grpc=_prefer_grpc,
    timeout=10,
)

async def check_qdrant() -> str:
    try:
        await qdrant_client.get_collections()
        return "ok"
    except Exception as e:
        return str(e)

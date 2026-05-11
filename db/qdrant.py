import os
from qdrant_client import AsyncQdrantClient

# `query_batch_points` (used by search_v2.similar_movies for the
# 8-named-vector shape search) works over HTTP and already collapses
# N round trips into one — the biggest single Qdrant latency cut.
#
# gRPC transport would add another ~15-30% per-call speedup but
# requires exposing port 6334 in docker-compose (the current
# compose only maps 6333). To enable: add `"6334:6334"` to the
# qdrant service in docker-compose.yml, restart the container,
# then set the env var QDRANT_PREFER_GRPC=1.
_prefer_grpc = os.getenv("QDRANT_PREFER_GRPC", "0").lower() in {"1", "true", "yes"}
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

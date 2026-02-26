import os                                                                                                                                                           
from qdrant_client import AsyncQdrantClient                                                                                                                         
                  
qdrant_client = AsyncQdrantClient(
    host=os.getenv("QDRANT_HOST", None),
    port=int(os.getenv("QDRANT_PORT", None)),
    timeout=10,
)

async def check_qdrant() -> str:
    try:
        await qdrant_client.get_collections()
        return "ok"
    except Exception as e:
        return str(e)
"""
OpenAI Files API and Batch API interactions.

Thin wrapper around the OpenAI Python SDK for batch operations.
Nothing in this module knows about movies, generations, or schemas —
it only knows about files and batches.

Functions:

upload_jsonl(file_path: Path) -> str:
    Uploads a JSONL file via client.files.create(purpose="batch").
    Returns the OpenAI file ID.

create_batch(input_file_id: str) -> str:
    Creates a batch via client.batches.create() with:
        endpoint="/v1/chat/completions"
        completion_window="24h"
    Returns the OpenAI batch ID.

check_status(batch_id: str) -> BatchStatus:
    Retrieves batch status via client.batches.retrieve().
    Returns a BatchStatus dataclass with:
        status (str), request_counts (total/completed/failed),
        output_file_id (str|None), error_file_id (str|None)

download_file(file_id: str, dest_path: Path) -> Path:
    Downloads file content via client.files.content().
    Writes raw bytes to dest_path. Used for both output and
    error result files.

Dependencies:
    - OpenAI Python SDK (openai)
    - OPENAI_API_KEY from environment / .env

Output files are downloaded to ingestion_data/metadata_batches/.
"""

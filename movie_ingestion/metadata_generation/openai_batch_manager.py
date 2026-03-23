"""
OpenAI Files API and Batch API wrapper.

Thin wrapper around the OpenAI Python SDK for batch operations.
Nothing in this module knows about movies, metadata types, or schemas —
it only knows about files and batches. Named openai_batch_manager (not
batch_manager) because other metadata types may use non-OpenAI providers.

Functions:

upload_and_create_batch(requests) -> str:
    Serializes request dicts to JSONL in memory, uploads via Files API,
    creates a batch, returns the batch_id.

check_batch_status(batch_id) -> BatchStatus:
    Retrieves batch status via batches.retrieve().

download_results(file_id) -> list[dict]:
    Downloads file content and returns parsed JSONL lines in memory.
"""

import io
import os
import json
from dataclasses import dataclass

from openai import OpenAI

from dotenv import load_dotenv

load_dotenv()

# Synchronous client — batch operations are infrequent (one call per
# batch, not per movie) so async adds complexity for no benefit.
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError(
        "OPENAI_API_KEY environment variable not set. "
        "Please set it before importing this module."
    )

# Create OpenAI client instance for use throughout the module
_client = OpenAI(api_key=api_key)


@dataclass(slots=True)
class BatchStatus:
    """Status of an OpenAI batch."""
    batch_id: str
    # validating, in_progress, finalizing, completed, failed, expired, cancelled
    status: str
    total: int
    completed: int
    failed: int
    output_file_id: str | None
    error_file_id: str | None
    # Batch-level errors (e.g. invalid JSONL, schema validation).
    # Per-request errors go in the error file; these are why the
    # entire batch failed. List of dicts with code/message keys.
    errors: list[dict] | None = None


def upload_and_create_batch(requests: list[dict]) -> str:
    """Serialize request dicts to JSONL, upload, and create a batch.

    1. Writes each request dict as a JSON line to an in-memory buffer.
    2. Uploads the buffer via client.files.create(purpose="batch").
    3. Creates a batch via client.batches.create() with 24h window.

    Args:
        requests: List of Batch API request dicts, each containing
            custom_id, method, url, and body keys.

    Returns:
        The OpenAI batch ID string.
    """
    # Build JSONL in memory — 10K requests × ~2KB each ≈ 20MB, fits easily.
    buffer = io.BytesIO()
    for req in requests:
        line = json.dumps(req, ensure_ascii=False) + "\n"
        buffer.write(line.encode("utf-8"))
    buffer.seek(0)

    # Upload the JSONL file
    file_obj = _client.files.create(
        file=("batch_requests.jsonl", buffer),
        purpose="batch",
    )

    # Create the batch
    batch = _client.batches.create(
        input_file_id=file_obj.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )

    return batch.id


def check_batch_status(batch_id: str) -> BatchStatus:
    """Retrieve the current status of an OpenAI batch.

    Args:
        batch_id: The OpenAI batch ID to check.

    Returns:
        BatchStatus with current counts and file IDs.
    """
    batch = _client.batches.retrieve(batch_id)
    counts = batch.request_counts

    # Extract batch-level errors (distinct from per-request error file).
    # batch.errors is an Errors object with a .data list of ErrorData objects.
    batch_errors = None
    if batch.errors and batch.errors.data:
        batch_errors = [
            {"code": e.code, "message": e.message}
            for e in batch.errors.data
        ]

    return BatchStatus(
        batch_id=batch.id,
        status=batch.status,
        total=counts.total if counts else 0,
        completed=counts.completed if counts else 0,
        failed=counts.failed if counts else 0,
        output_file_id=batch.output_file_id,
        error_file_id=batch.error_file_id,
        errors=batch_errors,
    )


def download_results(file_id: str) -> list[dict]:
    """Download a batch result/error file and return parsed JSONL lines.

    Downloads the file content via client.files.content(), decodes as
    UTF-8, and parses each line as JSON. The entire result is kept in
    memory — 10K results at ~500 bytes each is ~5MB.

    Args:
        file_id: The OpenAI file ID (output_file_id or error_file_id).

    Returns:
        List of parsed result dicts from the JSONL file.
    """
    response = _client.files.content(file_id)
    content = response.text

    results = []
    for line in content.strip().splitlines():
        if line:
            results.append(json.loads(line))

    return results

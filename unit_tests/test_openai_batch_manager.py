"""
Unit tests for movie_ingestion.metadata_generation.openai_batch_manager.

Since this module creates an OpenAI client at import time (requires
OPENAI_API_KEY), all tests mock at the module level via monkeypatch.

Covers:
  - BatchStatus dataclass fields and defaults
  - upload_and_create_batch: files.create, batches.create, JSONL serialization
  - check_batch_status: BatchStatus mapping, error handling
  - download_results: JSONL parsing
"""

import io
import json
import os
from dataclasses import fields
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Module import with mocked OPENAI_API_KEY
# ---------------------------------------------------------------------------

# The openai_batch_manager module raises ValueError at import time if
# OPENAI_API_KEY is not set. We need to patch it before importing.

@pytest.fixture(autouse=True)
def _mock_openai_env(monkeypatch):
    """Ensure OPENAI_API_KEY is set for import and mock the OpenAI client."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-for-unit-tests")


# We import the module inside each test class to ensure the env var is set.
# Using a fixture to provide the mocked module.

@pytest.fixture()
def batch_manager(monkeypatch):
    """Import openai_batch_manager with a mocked OpenAI client."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-for-unit-tests")

    # Mock the OpenAI class before importing the module
    mock_client = MagicMock()
    with patch("movie_ingestion.metadata_generation.openai_batch_manager.OpenAI") as MockOpenAI:
        MockOpenAI.return_value = mock_client
        # Force re-import with mocked client
        import importlib
        import movie_ingestion.metadata_generation.openai_batch_manager as mod
        importlib.reload(mod)
        # Replace the module-level _client with our mock
        mod._client = mock_client
        yield mod, mock_client


# ---------------------------------------------------------------------------
# Tests: BatchStatus
# ---------------------------------------------------------------------------


class TestBatchStatus:
    """Tests for the BatchStatus dataclass."""

    def test_batch_status_fields(self, batch_manager) -> None:
        """BatchStatus has expected fields."""
        mod, _ = batch_manager
        bs = mod.BatchStatus(
            batch_id="batch_123",
            status="completed",
            total=100,
            completed=95,
            failed=5,
            output_file_id="file-output-123",
            error_file_id="file-error-123",
        )
        assert bs.batch_id == "batch_123"
        assert bs.status == "completed"
        assert bs.total == 100
        assert bs.completed == 95
        assert bs.failed == 5
        assert bs.output_file_id == "file-output-123"
        assert bs.error_file_id == "file-error-123"

    def test_batch_status_errors_defaults_to_none(self, batch_manager) -> None:
        """errors field defaults to None."""
        mod, _ = batch_manager
        bs = mod.BatchStatus(
            batch_id="b", status="in_progress",
            total=10, completed=0, failed=0,
            output_file_id=None, error_file_id=None,
        )
        assert bs.errors is None


# ---------------------------------------------------------------------------
# Tests: upload_and_create_batch
# ---------------------------------------------------------------------------


class TestUploadAndCreateBatch:
    """Tests for upload_and_create_batch."""

    def test_calls_files_create(self, batch_manager) -> None:
        """files.create is called with JSONL content and purpose='batch'."""
        mod, mock_client = batch_manager

        mock_file = MagicMock()
        mock_file.id = "file-123"
        mock_client.files.create.return_value = mock_file

        mock_batch = MagicMock()
        mock_batch.id = "batch-abc"
        mock_client.batches.create.return_value = mock_batch

        requests = [{"custom_id": "test_1", "method": "POST", "url": "/v1/chat/completions", "body": {}}]
        mod.upload_and_create_batch(requests)

        mock_client.files.create.assert_called_once()
        call_kwargs = mock_client.files.create.call_args[1]
        assert call_kwargs["purpose"] == "batch"

    def test_calls_batches_create(self, batch_manager) -> None:
        """batches.create is called with correct input_file_id and endpoint."""
        mod, mock_client = batch_manager

        mock_file = MagicMock()
        mock_file.id = "file-123"
        mock_client.files.create.return_value = mock_file

        mock_batch = MagicMock()
        mock_batch.id = "batch-abc"
        mock_client.batches.create.return_value = mock_batch

        requests = [{"custom_id": "test_1", "method": "POST", "url": "/v1/chat/completions", "body": {}}]
        mod.upload_and_create_batch(requests)

        mock_client.batches.create.assert_called_once()
        call_kwargs = mock_client.batches.create.call_args[1]
        assert call_kwargs["input_file_id"] == "file-123"
        assert call_kwargs["endpoint"] == "/v1/chat/completions"

    def test_returns_batch_id(self, batch_manager) -> None:
        """Returned string matches batch.id from mock."""
        mod, mock_client = batch_manager

        mock_file = MagicMock()
        mock_file.id = "file-123"
        mock_client.files.create.return_value = mock_file

        mock_batch = MagicMock()
        mock_batch.id = "batch-xyz"
        mock_client.batches.create.return_value = mock_batch

        requests = [{"custom_id": "test_1", "method": "POST", "url": "/v1/chat/completions", "body": {}}]
        result = mod.upload_and_create_batch(requests)

        assert result == "batch-xyz"

    def test_serializes_jsonl(self, batch_manager) -> None:
        """BytesIO content contains valid JSONL (one JSON line per request)."""
        mod, mock_client = batch_manager

        mock_file = MagicMock()
        mock_file.id = "file-123"
        mock_client.files.create.return_value = mock_file

        mock_batch = MagicMock()
        mock_batch.id = "batch-abc"
        mock_client.batches.create.return_value = mock_batch

        requests = [
            {"custom_id": "a_1", "method": "POST", "url": "/v1/chat/completions", "body": {"key": "val1"}},
            {"custom_id": "b_2", "method": "POST", "url": "/v1/chat/completions", "body": {"key": "val2"}},
        ]
        mod.upload_and_create_batch(requests)

        # Extract the file argument from files.create call
        call_kwargs = mock_client.files.create.call_args[1]
        file_arg = call_kwargs["file"]
        # file_arg is a tuple (filename, BytesIO)
        _, buffer = file_arg
        content = buffer.read().decode("utf-8")
        lines = [line for line in content.strip().split("\n") if line]
        assert len(lines) == 2
        # Each line should be valid JSON
        for line in lines:
            parsed = json.loads(line)
            assert "custom_id" in parsed


# ---------------------------------------------------------------------------
# Tests: check_batch_status
# ---------------------------------------------------------------------------


class TestCheckBatchStatus:
    """Tests for check_batch_status."""

    def test_returns_batch_status(self, batch_manager) -> None:
        """Returned BatchStatus fields match mock batch response."""
        mod, mock_client = batch_manager

        mock_batch = MagicMock()
        mock_batch.id = "batch-123"
        mock_batch.status = "completed"
        mock_batch.request_counts.total = 100
        mock_batch.request_counts.completed = 95
        mock_batch.request_counts.failed = 5
        mock_batch.output_file_id = "file-out"
        mock_batch.error_file_id = "file-err"
        mock_batch.errors = None
        mock_client.batches.retrieve.return_value = mock_batch

        result = mod.check_batch_status("batch-123")

        assert result.batch_id == "batch-123"
        assert result.status == "completed"
        assert result.total == 100
        assert result.completed == 95
        assert result.failed == 5
        assert result.output_file_id == "file-out"

    def test_handles_errors(self, batch_manager) -> None:
        """Batch with errors.data populates BatchStatus.errors."""
        mod, mock_client = batch_manager

        mock_error = MagicMock()
        mock_error.code = "token_limit_exceeded"
        mock_error.message = "Too many tokens"

        mock_errors_obj = MagicMock()
        mock_errors_obj.data = [mock_error]

        mock_batch = MagicMock()
        mock_batch.id = "batch-456"
        mock_batch.status = "failed"
        mock_batch.request_counts.total = 10
        mock_batch.request_counts.completed = 0
        mock_batch.request_counts.failed = 10
        mock_batch.output_file_id = None
        mock_batch.error_file_id = "file-err"
        mock_batch.errors = mock_errors_obj
        mock_client.batches.retrieve.return_value = mock_batch

        result = mod.check_batch_status("batch-456")

        assert result.errors is not None
        assert len(result.errors) == 1
        assert result.errors[0]["code"] == "token_limit_exceeded"

    def test_no_errors(self, batch_manager) -> None:
        """Batch without errors has BatchStatus.errors as None."""
        mod, mock_client = batch_manager

        mock_batch = MagicMock()
        mock_batch.id = "batch-789"
        mock_batch.status = "in_progress"
        mock_batch.request_counts.total = 50
        mock_batch.request_counts.completed = 25
        mock_batch.request_counts.failed = 0
        mock_batch.output_file_id = None
        mock_batch.error_file_id = None
        mock_batch.errors = None
        mock_client.batches.retrieve.return_value = mock_batch

        result = mod.check_batch_status("batch-789")
        assert result.errors is None


# ---------------------------------------------------------------------------
# Tests: download_results
# ---------------------------------------------------------------------------


class TestDownloadResults:
    """Tests for download_results."""

    def test_parses_jsonl(self, batch_manager) -> None:
        """Multi-line JSONL content is parsed into list of dicts."""
        mod, mock_client = batch_manager

        jsonl_content = (
            '{"custom_id": "plot_events_1", "response": {"status_code": 200}}\n'
            '{"custom_id": "plot_events_2", "response": {"status_code": 200}}\n'
        )
        mock_response = MagicMock()
        mock_response.text = jsonl_content
        mock_client.files.content.return_value = mock_response

        result = mod.download_results("file-123")

        assert len(result) == 2
        assert result[0]["custom_id"] == "plot_events_1"
        assert result[1]["custom_id"] == "plot_events_2"

    def test_empty_file(self, batch_manager) -> None:
        """Empty content returns empty list."""
        mod, mock_client = batch_manager

        mock_response = MagicMock()
        mock_response.text = ""
        mock_client.files.content.return_value = mock_response

        result = mod.download_results("file-empty")
        assert result == []

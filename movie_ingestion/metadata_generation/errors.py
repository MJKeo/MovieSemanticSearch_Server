"""
Shared exceptions for LLM metadata generation.

Two distinct error types cover the failure modes that can occur
during any generation call (plot_events, reception, etc.):

- MetadataGenerationError: The LLM API call itself raised an exception
  (network error, timeout, invalid response, validation failure, etc.)

- MetadataGenerationEmptyResponseError: The API call succeeded but
  returned a None parsed response, meaning the model produced no
  usable output.
"""


class MetadataGenerationError(Exception):
    """Raised when an LLM metadata generation call fails with an exception."""

    def __init__(self, generation_type: str, title: str, cause: Exception):
        self.generation_type = generation_type
        self.title = title
        self.cause = cause
        super().__init__(
            f"{generation_type} generation failed for '{title}': {cause}"
        )


class MetadataGenerationEmptyResponseError(Exception):
    """Raised when an LLM metadata generation call returns None."""

    def __init__(self, generation_type: str, title: str):
        self.generation_type = generation_type
        self.title = title
        super().__init__(
            f"{generation_type} generation returned None for '{title}'"
        )

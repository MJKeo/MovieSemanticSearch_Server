"""
Per-generation request body builders.

Each module exposes a single function:

    build_request(inputs: ConsolidatedInputs, wave1_outputs: Wave1Outputs | None = None) -> dict

The returned dict is the 'body' field for a Batch API JSONL request line:
    {
        "model": "gpt-5-mini",
        "messages": [
            {"role": "system", "content": "<system prompt>"},
            {"role": "user", "content": "<assembled user prompt>"}
        ],
        "response_format": <json_schema from Pydantic model>,
        "reasoning_effort": "low"
    }

The JSONL wrapping (custom_id, method, url) is handled by request_builder.py,
not here. Generators only know about their inputs, prompt, and schema.

This separation means generators are transport-agnostic -- the same
build_request() output could be used for real-time API calls during
testing or debugging without any changes.

Generation types and their waves:
    Wave 1: plot_events, reception
    Wave 2: plot_analysis, viewer_experience, watch_context,
            narrative_techniques, production (2 sub-calls)
"""

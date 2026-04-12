"""
Per-generation async LLM callers.

Each module exposes two public functions:
    build_<type>_user_prompt(movie, ...) -> str   # prompt construction
    generate_<type>(movie, ...) -> Tuple[Output, TokenUsage]  # async LLM call

Generators are real-time async callers (ADR-027): they take MovieInputData,
build a user prompt, call generate_llm_response_async, and return the parsed
output with token usage. Callers can override provider/model/kwargs.

Generation types and their waves:
    Wave 1: plot_events, reception
    Wave 2: plot_analysis, viewer_experience, watch_context,
            narrative_techniques, production_keywords,
            production_techniques, source_of_inspiration
"""

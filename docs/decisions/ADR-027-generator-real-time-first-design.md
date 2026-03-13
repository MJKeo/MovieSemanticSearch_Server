# [027] — Generator Contract: Real-Time Callers Before Batch API Wrapping

## Status
Active

## Context

ADR-024 designed Stage 6 generators as transport-agnostic request builders:
each `generators/` file would return a `body` dict that `request_builder.py`
wraps into Batch API JSONL format. This kept generators decoupled from the
transport layer and allowed the same generator to be called in both real-time
and batch contexts.

When implementing the first generator (`plot_events.py`), a parallel
requirement emerged: evaluate output quality across 5 LLM providers and
multiple models before committing to a provider for the batch run (~$150
total cost at current estimates). This evaluation requires running generators
interactively in a Jupyter notebook against a curated set of movies.

The body-dict contract is incompatible with interactive evaluation: a function
that builds a request body but doesn't call the LLM cannot be used for
side-by-side comparison in a notebook.

## Decision

Implement generators as **async real-time callers**, not body-dict builders.

Each generator takes `MovieInputData` + explicit `provider` / `model` + `**kwargs`,
calls `generate_llm_response_async`, and returns `Tuple[Output, TokenUsage]`.
The generator is responsible for prompt assembly, the LLM call, and typed
error wrapping (`MetadataGenerationError`, `MetadataGenerationEmptyResponseError`).

The Batch API scaffolding in `request_builder.py` and `run.py` (designed for
body-dict generators) is currently misaligned with the implemented generator
interface. Alignment is deferred until model selection is finalized.

**Provider and model are required params with no defaults** on every generator
function. The caller (notebook, future orchestrator) must always specify these.
This ensures the choice is explicit rather than implicit.

## Alternatives Considered

1. **Keep body-dict contract, add a separate evaluation wrapper**: Would
   preserve the Batch API design while enabling interactive evaluation.
   Rejected — adds a second abstraction per generator and duplicates
   prompt-building logic (the body-dict builder would need to build the
   same prompt as the wrapper).

2. **Implement body-dict generators and run them via the Batch API for
   model comparison**: Would produce quality data but at batch latency
   (up to 24h per wave). Interactive notebooks need synchronous feedback.
   Rejected.

3. **Design generators to support both contracts via a flag**: Adds
   conditional branching to every generator. Rejected — the two use cases
   have different return types (`body` dict vs. `Tuple[Output, TokenUsage]`).

## Consequences

- The Batch API scaffolding in `request_builder.py` and `run.py` requires
  rework before Stage 6 can run at full scale. Current generators do not
  produce the `body` dicts those modules expect.
- Interactive model evaluation is straightforward: notebooks call
  `generate_plot_events(movie, provider=..., model=..., **kwargs)` directly.
- Each generator can raise `MetadataGenerationError` or
  `MetadataGenerationEmptyResponseError` — callers must handle both.
- `TokenUsage` is imported from `implementation.llms.vector_metadata_generation_methods`
  (reused NamedTuple) rather than defined per generator.

## References

- ADR-024 (Batch API architecture) — original body-dict generator contract
- ADR-026 (multi-provider routing) — `generate_llm_response_async` used by generators
- ADR-012 (LLM cost optimization) — model evaluation rationale
- movie_ingestion/metadata_generation/generators/plot_events.py — reference implementation
- movie_ingestion/metadata_generation/metadata_generation_playground.ipynb — evaluation notebook

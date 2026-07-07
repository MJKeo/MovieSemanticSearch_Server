"""Canonical LLM model pricing + cost computation.

Single source of truth for the USD cost of a structured-generation call,
used by the LLM router (`generate_llm_response_async`) to emit the
`llm.cost_usd` telemetry attribute. Prices are USD per 1,000,000 tokens as
`(input, cached_input, output)`, standard (non-batch) API pricing. The
`cached_input` rate is the discounted price providers charge for input tokens
served from their prompt cache; see `compute_llm_cost_usd` for how the three
rates combine.

Design notes:
- **Dependency-free** (stdlib only) so it can be imported from the low-level
  LLM router without dragging in the OTel SDK or provider clients.
- **Honest on gaps.** An unpriced model returns `None` rather than a fabricated
  cost — the caller then omits the cost attribute and logs a warning, so a
  missing pricing entry surfaces instead of silently reporting `$0` or a wrong
  number. Add models here as they enter the routed set.
- Keyed by the model string passed to the router (the same identifier the
  provider SDKs receive), so `gen_ai.request.model` and the price lookup agree.

NOTE: `movie_ingestion/metadata_generation/helper_scripts/estimate_generation_cost.py`
carries a parallel `MODEL_PRICING` table for offline batch-cost estimation.
That helper should eventually import from here to de-duplicate (see docs/TODO.md);
it is intentionally left untouched in this changeset to keep scope contained.
"""

from __future__ import annotations


# (input_price, cached_input_price, output_price) per 1,000,000 tokens, USD,
# standard (non-batch) API pricing. `cached_input_price` is the discounted rate
# for input tokens served from the provider's prompt cache (a subset of input
# tokens — see compute_llm_cost_usd). Scoped to the models the *live serving
# path* actually routes to (the search endpoints) plus the embedding model —
# not the offline ingestion / metadata-generation models.
#
# `0.0` entries are placeholders to fill in manually. Two traps to know:
#  (1) a listed model with a zero rate reports `llm.cost_usd = 0.0` (not
#      `None`), so it does NOT trip the "no pricing" warning an absent model
#      does — these won't self-announce as unpriced and must be remembered.
#  (2) a zero `cached_input_price` prices cached tokens as FREE, not merely
#      discounted, so cache-heavy calls under-report until the real rate is set.
# Models absent here still yield `None` from compute_llm_cost_usd.
_MODEL_PRICING_PER_MILLION: dict[str, tuple[float, float, float]] = {
    # --- Structured-generation models (input, cached_input, output) — prices TBD ---
    "gemini-3.5-flash":       (1.5, 0.15, 9.0),   # steps 0 / 1 / 2 + implicit expectations
    "gpt-5.4-mini":           (0.75, 0.075, 4.5),  # step 3 + all query generation + entity flows
    # --- Embedding model (input only; no output, no prompt caching) ---
    "text-embedding-3-large": (0.13, 0.13, 0.0),  # OpenAI standard price ($0.13 / 1M)
}

_TOKENS_PER_MILLION = 1_000_000


def compute_llm_cost_usd(
    model: str,
    input_tokens: int,
    output_tokens: int,
    cached_input_tokens: int = 0,
) -> float | None:
    """Return the USD cost of a call, or `None` if the model is unpriced.

    `cached_input_tokens` is the portion of `input_tokens` the provider served
    from its prompt cache — a SUBSET of input, never additional to it. It is
    billed at the discounted `cached_input_price`; the remaining (uncached)
    input is billed at the full `input_price`:

        cached_input_tokens            * cached_input_price
      + (input_tokens - cached_input)  * input_price
      + output_tokens                  * output_price

    `None` is a deliberate "unknown" signal — the router omits the cost
    attribute and warns rather than recording a wrong or zero cost.
    """
    pricing = _MODEL_PRICING_PER_MILLION.get(model)
    if pricing is None:
        return None
    input_price, cached_input_price, output_price = pricing
    # Clamp so a provider quirk where cached > input can never produce a
    # negative uncached-token count (and thus a nonsensical negative cost).
    uncached_input_tokens = max(0, input_tokens - cached_input_tokens)
    return (
        cached_input_tokens * cached_input_price
        + uncached_input_tokens * input_price
        + output_tokens * output_price
    ) / _TOKENS_PER_MILLION

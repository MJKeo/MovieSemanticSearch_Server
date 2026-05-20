# Search V2 — Standalone character-franchise fanout LLM call.
#
# The CHARACTER_FRANCHISE_FANOUT handler bucket fires a single LLM call
# that walks one character referent across both its cast-list credits
# (character_forms) and its film series titles (franchise_forms). The
# prompt is purpose-built for the cross-credit aliasing problem (e.g.
# "Tony Stark" is the credited form in MCU films, not "Iron Man") and
# gives strictly better recall than two independent generator calls.
#
# This module exposes the same LLM call as a standalone helper so the
# Step-0 character-franchise entity-flow runner can invoke it directly,
# without routing through the Step-1/2/3 query-understanding stack.
# Step 0 already pinned the canonical character name; the standalone
# call is a one-LLM-roundtrip alias-expansion step on top of that.
#
# Two design notes:
#
#   1. We deliberately do NOT call handler._run_handler_llm. That
#      function takes a CategoryCall + Trait, which the entity-flow
#      path does not have. Fabricating those would couple this module
#      to the Step-3 schema. The user message is small enough to
#      serialize inline.
#
#   2. We deliberately do NOT call build_user_message. It also requires
#      a CategoryCall and renders a sibling-categories block we don't
#      use. Inlining the ~5-line XML shape is simpler and keeps the
#      entity-flow path off the Step-3 schema dependency.
#
# Reuses HANDLER_LLM_PROVIDER / HANDLER_LLM_MODEL / HANDLER_LLM_KWARGS
# from the category handler so the two callers stay in lockstep on
# model + reasoning settings — if the handler upgrades models, this
# caller follows automatically.

from __future__ import annotations

import logging
from xml.sax.saxutils import escape as xml_escape

from implementation.llms.generic_methods import generate_llm_response_async
from schemas.trait_category import CategoryName
from search_v2.endpoint_fetching.category_handlers.endpoint_registry import (
    CharacterFranchiseFanoutSchema,
)
from search_v2.endpoint_fetching.category_handlers.handler import (
    HANDLER_LLM_KWARGS,
    HANDLER_LLM_MODEL,
    HANDLER_LLM_PROVIDER,
)
from search_v2.endpoint_fetching.category_handlers.prompt_builder import (
    build_system_prompt,
)

logger = logging.getLogger(__name__)


# Empty sibling-categories wrapper. The fanout prompt's "Reading sibling
# context" section sees combine_mode="single" and falls back to standalone
# behavior, which is exactly the right framing for an entity-flow caller
# with no parallel sibling categories. Mirrors the no-sibling branch in
# prompt_builder._render_sibling_block.
_EMPTY_SIBLING_BLOCK = '<sibling_categories combine_mode="single"/>'


def _build_user_message(*, retrieval_intent: str, expressions: list[str]) -> str:
    """Render the minimal handler-LLM user message XML.

    Same three sections as prompt_builder.build_user_message
    (retrieval_intent + expressions + sibling_categories), but the
    sibling block is always the empty single-mode wrapper because
    entity-flow callers have no sibling categories.
    """
    intent_xml = xml_escape(retrieval_intent)
    expression_lines = "\n".join(
        f"  <expression>{xml_escape(expr)}</expression>" for expr in expressions
    )
    return (
        f"<retrieval_intent>{intent_xml}</retrieval_intent>\n"
        f"<expressions>\n{expression_lines}\n</expressions>\n"
        f"{_EMPTY_SIBLING_BLOCK}"
    )


async def run_character_franchise_fanout_call(
    *,
    retrieval_intent: str,
    expressions: list[str],
) -> CharacterFranchiseFanoutSchema | None:
    """Invoke the character-franchise fanout prompt as a standalone call.

    Builds the same system prompt the standard handler builds for
    CategoryName.CHARACTER_FRANCHISE (which selects the fanout bucket
    in its category metadata), assembles the minimal user-message XML
    inline, and calls generate_llm_response_async with the shared
    HANDLER_LLM_* settings.

    Args:
        retrieval_intent: One-sentence positive-presence framing of the
            lookup, e.g. "Movies featuring the character Spider-Man."
        expressions: Surface forms from the original query. For the
            entity-flow caller this is typically a single-element list
            containing the canonical character name.

    Returns:
        The structured fanout output, or None on any LLM failure
        (timeout, parse error, exhausted retries). Callers soft-fail
        on None — the runner returns an empty result rather than
        propagating the exception.
    """
    system_prompt = build_system_prompt(CategoryName.CHARACTER_FRANCHISE)
    user_message = _build_user_message(
        retrieval_intent=retrieval_intent,
        expressions=expressions,
    )

    try:
        # Timeout, retry, and jittered backoff live inside
        # generate_llm_response_async — same configuration the
        # standard handler uses.
        response, _, _ = await generate_llm_response_async(
            provider=HANDLER_LLM_PROVIDER,
            user_prompt=user_message,
            system_prompt=system_prompt,
            response_format=CharacterFranchiseFanoutSchema,
            model=HANDLER_LLM_MODEL,
            **HANDLER_LLM_KWARGS,
        )
        return response
    except Exception as exc:  # noqa: BLE001 — soft-fail by design
        # Include the first expression (typically the canonical
        # character name) so production log greps can tie failures
        # back to the originating query without stack-hunting.
        referent = expressions[0] if expressions else "<no expressions>"
        logger.warning(
            "character-franchise fanout LLM call failed for "
            "referent=%r; caller will soft-fail to an empty result "
            "(error=%r)",
            referent,
            exc,
        )
        return None

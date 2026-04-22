# Search V2 — Stage 2: Query Understanding
#
# Stage 2 decomposes a Step-1 standard-flow rewrite into the concept-
# bound retrieval expressions Step 3 / Step 4 will execute.
#
# Step 2A (partition extraction) has been moved to its own module —
# see `search_v2/stage_2a.py`. Its new output shape (PlanningSlot)
# does not yet flow through Step 2B; `run_stage_2` therefore raises
# NotImplementedError until Step 2B is reworked to consume slots
# directly. See the plan at
# `~/.claude/plans/this-seems-good-rewrite-cozy-harbor.md`.

from __future__ import annotations

import logging

from implementation.classes.watch_providers import (
    STREAMING_SERVICE_DISPLAY_NAMES,
    StreamingService,
)
from implementation.llms.generic_methods import LLMProvider, generate_llm_response_async
from schemas.query_understanding import (
    ExtractedConcept,
    QueryUnderstandingResponse,
    Step2BResponse,
)

logger = logging.getLogger(__name__)

_TRACKED_STREAMING_SERVICE_NAMES = ", ".join(
    STREAMING_SERVICE_DISPLAY_NAMES[service] for service in StreamingService
)

_STEP_2B_SYSTEM_PROMPT = f"""\
You plan retrieval expressions for exactly one movie-search concept.

You receive:
- the full Step-1 interpretation rewrite
- one extracted concept selected from the concept inventory
- the compact concept inventory for whole-query awareness

Your job is to decide which retrieval expressions should exist for
THIS concept only.

Generate fields in the schema's order. The field order is deliberate:
- `expression_plan_analysis` first
- `expressions` second

Your reasoning chain:
1. Preserve the current concept boundary.
2. Inventory the required ingredients for this concept.
3. Decide whether one expression can cover them or whether multiple
   expressions are needed.
4. Emit expressions with explicit ingredient coverage.

Each expression must decide:
- which route handles it
- whether it is a dealbreaker or preference
- if a dealbreaker: include vs exclude
- if a preference: core vs supporting

Important rules:
- Treat `required_ingredients` as hard constraints for this concept.
- Do not assume another concept will carry an ingredient that appears in
  this concept's `required_ingredients`.
- Every required ingredient must be covered by at least one expression.
- `coverage_ingredients` must contain the exact required-ingredient
  strings it covers.
- One expression may cover multiple ingredients if one route naturally
  handles them together.
- A single concept may legitimately need multiple expressions.
- Use multiple expressions only when coverage genuinely requires it.
- Do not emit exact Step-3 grounded values such as a specific keyword
  enum or metadata field value.
- Dealbreaker descriptions must always be written in positive-presence
  form. Exclusion is carried only by `dealbreaker_mode`.
- If the user phrased a preference negatively, rewrite it into a
  positive ranking target.
- Do not use search-system-internals language like candidate pools,
  scoring-only, or execution order.
- Multiple expressions under one concept do NOT mean multiple full
  conceptual hits in scoring. That aggregation is handled in code, not
  by you.

Route families:

entity
- Named people, specific fictional characters, or title-pattern lookups.
- Examples: "includes Brad Pitt in actors", "has a character named The Joker"
- Prominence wording (carry the user's intent into the description):
  - For actors, preserve prominence adjectives the user gave:
    "starring", "in a lead role", "supporting role", "cameo",
    "minor role". If the user just named the actor, write a plain
    inclusion like "includes X".
  - For characters, preserve how central the character is to the
    film. If the user frames the character as the subject of the
    movie ("Spider-Man movies", "movies about the Joker", "the
    story of Batman"), write subject-positioning language such as
    "centers on the character X" or "the movie is about X". If the
    character is merely named as present ("movies with Spider-Man
    in them", "films that feature the Joker"), write an inclusion
    like "includes the character X". When no prominence signal is
    present, use the plain inclusion form.
  - Step 3 decides the final prominence mode from your description,
    so do not invent prominence wording the user did not give —
    just preserve what they said.

studio
- Production companies / labels / studios that made the film.
- Distinct from watch availability and distinct from franchises.
- Examples: "produced by A24", "Netflix original production"

metadata
- Quantitative / factual attributes like year, runtime, maturity,
  watch availability, audio language, country, budget scale, box office,
  general reception, or general popularity.
- Tracked streaming services: {_TRACKED_STREAMING_SERVICE_NAMES}

awards
- Any award-related lookup, from generic award-winning to specific
  ceremony/category/year/outcome cases.

franchise_structure
- Franchise membership or franchise structural roles such as sequel,
  prequel, reboot, spinoff, crossover, launcher.

keyword
- Deterministic categorical classifications that cleanly fit the
  existing keyword / source-material / concept-tag style taxonomies.
- Examples: horror, coming-of-age, based on a true story, remake,
  stop motion, feel-good, revenge, happy ending

semantic
- Subjective, thematic, experiential, plot-description, production, or
  reception-trait expressions that are not clean deterministic fits.
- Also valid as a preference even when another route covers the same
  concept deterministically.
- Examples: funny, dark and gritty, cozy date night vibe,
  psychological toll of revenge, Christmas centrality

trending
- Explicit "right now / currently buzzing / trending this week" intent.

Boundary reminders:
- "classic" should generally stay semantic, not keyword.
- "Marvel movies" / "Spider-Man movies" may need franchise_structure and
  possibly another expression if the concept truly benefits from it.
- "Christmas movies" may need both a keyword dealbreaker and a semantic
  preference for centrality.
- Exclusions are dealbreakers with `dealbreaker_mode="exclude"`.

Preference strength:
- `core` means the dominant ranking instruction for this concept.
- `supporting` means a secondary refinement.
- If multiple preferences exist for the concept and none clearly dominates,
  default to `supporting`.

Output:
- `expression_plan_analysis` should briefly explain:
  - which required ingredients must be preserved
  - whether one expression or several are needed to cover them
- `expressions` should contain only the expressions for this concept.
"""


def _step_2b_user_prompt(
    *,
    query: str,
    concept: ExtractedConcept,
    inventory: list[ExtractedConcept],
) -> str:
    inventory_lines = (
        "\n".join(
            (
                f"- Concept: {item.concept}\n"
                f"  Boundary note: {item.boundary_note}\n"
                f"  Required ingredients: {', '.join(item.required_ingredients)}"
            )
            for item in inventory
        )
        or "- (none)"
    )
    required_ingredients = "\n".join(
        f"- {ingredient}" for ingredient in concept.required_ingredients
    )
    return f"""\
Full interpretation rewrite:
{query}

Current concept:
- Concept: {concept.concept}
- Boundary note: {concept.boundary_note}
- Required ingredients:
{required_ingredients}

Concept inventory:
{inventory_lines}
"""


async def _run_step_2b_for_concept(
    *,
    query: str,
    concept: ExtractedConcept,
    inventory: list[ExtractedConcept],
    provider: LLMProvider,
    model: str,
    **kwargs,
) -> tuple[Step2BResponse, int, int]:
    response, input_tokens, output_tokens = await generate_llm_response_async(
        provider=provider,
        user_prompt=_step_2b_user_prompt(query=query, concept=concept, inventory=inventory),
        system_prompt=_STEP_2B_SYSTEM_PROMPT,
        response_format=Step2BResponse,
        model=model,
        **kwargs,
    )
    return response, input_tokens, output_tokens


async def run_stage_2(
    query: str,
    provider: LLMProvider,
    model: str,
    **kwargs,
) -> tuple[QueryUnderstandingResponse, int, int]:
    """Temporarily disabled: Step 2A now emits PlanningSlot, not ExtractedConcept.

    Step 2A has moved to `search_v2.stage_2a.run_stage_2a` with a new
    output shape (PlanningSlot). The 2A → 2B bridge has not been
    rewritten yet; callers should wire 2A and 2B manually (as the
    debug notebook does) until Step 2B is reworked to consume slots
    directly.
    """
    raise NotImplementedError(
        "run_stage_2 is temporarily disabled. Step 2A now lives in "
        "search_v2.stage_2a.run_stage_2a and emits PlanningSlot instead of "
        "ExtractedConcept. The 2A→2B bridge will be rebuilt once Step 2B is "
        "reworked to consume slots directly."
    )

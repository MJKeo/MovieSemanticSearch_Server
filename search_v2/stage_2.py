# Search V2 — Stage 2: Query Understanding
#
# Revamped concept-based Stage 2:
#   Step 2A — extract the concept inventory from the Step-1 rewrite
#   Step 2B — plan one or more retrieval expressions per concept
#
# The public entrypoint remains `run_stage_2()`.

from __future__ import annotations

import asyncio
import logging

from pydantic import ValidationError

from implementation.classes.watch_providers import (
    STREAMING_SERVICE_DISPLAY_NAMES,
    StreamingService,
)
from implementation.llms.generic_methods import LLMProvider, generate_llm_response_async
from schemas.query_understanding import (
    ExtractedConcept,
    QueryConcept,
    QueryUnderstandingResponse,
    Step2AResponse,
    Step2BResponse,
)

logger = logging.getLogger(__name__)

_TRACKED_STREAMING_SERVICE_NAMES = ", ".join(
    STREAMING_SERVICE_DISPLAY_NAMES[service] for service in StreamingService
)

_STEP_2A_SYSTEM_PROMPT = """\
You extract the stable concept inventory for a movie-search query.

You receive a query that has already been rewritten by Step 1 into a
complete, concrete statement of what the user wants. Your job here is
to identify the minimal query ingredients and decide concept boundaries.

Generate fields in the schema's order. The field order is deliberate:
- `ingredient_inventory` first
- `concept_inventory_analysis` second
- `concepts` last

A concept is one user intent such as:
- "Spider-Man movie"
- "Christmas movie"
- "dark and gritty tone"
- "animated movie" in an exclusion context

Important rules:
- Think in user-intent concepts, not route-bound search items.
- `ingredient_inventory` is a short evidence inventory of the meaningful
  ingredients in the rewrite.
- One concept may later require multiple retrieval expressions.
- Keep one concept unified when multiple expressions are just alternate
  retrieval views of the same user intent.
- Keep genuinely separate user intents separate, even if they may end
  up routed to the same endpoint later.
- Split aggressively when one ingredient defines the candidate domain
  and another ingredient ranks or characterizes items within that domain.
- Keep concepts unified only when the phrase names one fused phenomenon
  or one entity whose multiple retrieval routes are internal to the same idea.
- Do not narrow a broad scope anchor to a more specific sub-brand or
  sub-unit unless the query explicitly says so.
- If unsure whether to keep "scope anchor + trait" together, split them.
- Do not decide routes.
- Do not decide dealbreaker vs preference.
- Do not decide include vs exclude.
- Do not decide exact Step-3 grounded values.

Examples:
- "Disney classics" should split into:
  - "Disney movies"
  - "historically significant / classic"
- "A24 horror movies" should split into:
  - "A24 movies"
  - "horror"
- "Spider-Man movies" should stay one concept even if later it needs
  both franchise and character expressions.
- "Christmas movies" should stay one concept even if later it needs a
  keyword dealbreaker plus a semantic preference for Christmas centrality.
- "dark and gritty" often stays one unified tonal concept.
- "award-winning comedy" is two concepts because the user is asking for
  award status and comedy classification separately.

Output:
- `ingredient_inventory` should list the minimal meaningful ingredients in the rewrite.
- `concept_inventory_analysis` should briefly explain the boundary choices.
- `concepts` should contain one object per concept with:
  - `boundary_note`: a brief note explaining why this is one concept boundary
  - `concept`: the concept label
  - `required_ingredients`: the exact ingredient strings from
    `ingredient_inventory` that this concept must preserve
"""

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
    """Decompose a standard-flow query into concept-bound expressions."""
    query = query.strip()
    if not query:
        raise ValueError("query must be a non-empty string.")

    step_2a_user_prompt = f"Interpretation rewrite:\n{query}"
    step_2a_response, input_tokens, output_tokens = await generate_llm_response_async(
        provider=provider,
        user_prompt=step_2a_user_prompt,
        system_prompt=_STEP_2A_SYSTEM_PROMPT,
        response_format=Step2AResponse,
        model=model,
        **kwargs,
    )

    step_2b_tasks = [
        _run_step_2b_for_concept(
            query=query,
            concept=concept,
            inventory=step_2a_response.concepts,
            provider=provider,
            model=model,
            **kwargs,
        )
        for concept in step_2a_response.concepts
    ]

    concepts: list[QueryConcept] = []
    if step_2b_tasks:
        raw_results = await asyncio.gather(*step_2b_tasks, return_exceptions=True)
        for extracted_concept, raw_result in zip(step_2a_response.concepts, raw_results):
            if isinstance(raw_result, Exception):
                logger.warning(
                    "Dropping Step 2 concept %r after Step 2B failure: %s",
                    extracted_concept.concept,
                    raw_result,
                )
                continue

            step_2b_response, concept_input_tokens, concept_output_tokens = raw_result
            try:
                concepts.append(
                    QueryConcept(
                        boundary_note=extracted_concept.boundary_note,
                        concept=extracted_concept.concept,
                        required_ingredients=extracted_concept.required_ingredients,
                        expression_plan_analysis=step_2b_response.expression_plan_analysis,
                        expressions=step_2b_response.expressions,
                    )
                )
            except ValidationError as exc:
                logger.warning(
                    "Dropping Step 2 concept %r after invalid Step 2B coverage: %s",
                    extracted_concept.concept,
                    exc,
                )
                continue
            input_tokens += concept_input_tokens
            output_tokens += concept_output_tokens

    if step_2a_response.concepts and not concepts:
        raise RuntimeError(
            "Step 2A extracted concepts, but every Step 2B concept was dropped."
        )

    # TODO: Priors were intentionally removed from the revamped Step 2.
    return (
        QueryUnderstandingResponse(
            ingredient_inventory=step_2a_response.ingredient_inventory,
            concept_inventory_analysis=step_2a_response.concept_inventory_analysis,
            concepts=concepts,
        ),
        input_tokens,
        output_tokens,
    )

# Search V2 — Stage 3 Studio Endpoint: Query Translation
#
# Translates one studio dealbreaker or preference from step 2 into a
# StudioQuerySpec that execution code runs against the brand-posting
# table and the freeform token-intersection index.
#
# The studio endpoint is a split-off of the old entity-type=STUDIO
# path. The v1 path did exact string match against `lex.lexical_dictionary`
# + `lex.inv_studio_postings` — that hit every one of the 11 failure
# classes documented in
# search_improvement_planning/v2_search_data_improvements.md (no
# brand/subsidiary bridging, no time-bounded ownership, no rename
# handling, no ordinal normalization, no long-tail sub-label matching).
#
# This translator emits two complementary axes:
#   1. brand (closed enum, 31 curated ProductionBrand values) for
#      umbrella queries. Execution reads the ingest-time-stamped
#      `lex.inv_production_brand_postings` keyed by the enum's int
#      brand_id attribute, so time-bounded membership (Lucasfilm under
#      Disney only from 2012-) and rename chains (Twentieth Century
#      Fox → 20th Century Studios) are handled automatically.
#   2. freeform_names (≤3 IMDB surface forms) for specific sub-labels
#      and long-tail studios. Execution normalizes + tokenizes + does
#      DF-filtered intersection against `lex.studio_token`, then joins
#      the resulting production_company_ids against
#      `movie_card.production_company_ids`.
#
# Upstream routing guarantees: by the time an item reaches this
# endpoint, step-2 has already decided the user is asking about a
# production company (not a streaming platform). Streamer
# disambiguation (NETFLIX / AMAZON_MGM / APPLE_STUDIOS as producer
# vs. watch-provider) lives entirely in stage_2.py.

from implementation.llms.generic_methods import LLMProvider, generate_llm_response_async
from schemas.production_brand_surface_forms import render_brand_registry_for_prompt
from schemas.studio_translation import StudioQuerySpec

# ---------------------------------------------------------------------------
# System prompt — modular sections concatenated at module level.
#
# Structure: task → direction-agnostic framing → brand-vs-freeform
# decision → brand registry table → freeform canonicalization →
# output field guidance.
#
# Prompt authoring conventions applied:
# - Evidence-inventory reasoning (thinking precedes brand /
#   freeform_names and forces umbrella-vs-specific scope reasoning)
# - Principle-based constraints, not failure catalogs
# - Evaluation guidance over outcome shortcuts (tell the LLM what a
#   "good surface form" is, not a keyword list)
# - No schema/implementation details leaked to the LLM (normalization,
#   DF filtering, intersection logic all happen in execution)
# ---------------------------------------------------------------------------

_TASK = """\
You translate one studio or production-company lookup into a concrete \
studio query specification. You receive a single studio requirement \
that has already been interpreted, routed, and framed as a positive- \
presence lookup. Your job is not to decide what the user meant — that \
is already done. Your job is to produce the exact parameters the \
retrieval layer needs.

Inputs you receive:
- intent_rewrite — the full concrete statement of what the user is \
looking for. Use it to disambiguate the studio if the description \
alone is ambiguous (for example, "Fox" could mean Twentieth Century \
Fox or Fox Searchlight depending on surrounding context).
- description — the single studio requirement you are translating, \
always written in positive-presence form ("includes A24 as a \
production company", "produced by Marvel Studios", "made by Ghibli").
- routing_rationale — a concept-type label explaining why this item \
was routed to this endpoint.

Trust the upstream routing. By the time an item reaches this endpoint, \
the query has already been classified as a *production-company* \
lookup, not a streaming-availability lookup. For entities that are \
both producers and streaming platforms (Netflix, Amazon, Apple), \
assume the user means the producer. You do not need to second-guess \
this — if they meant "movies on Netflix" the item would have gone to \
a different endpoint.

You choose between two paths. Set exactly one:
- brand for umbrella / parent-brand queries that name one of the \
curated brands in the registry below.
- freeform_names for specific sub-labels or long-tail studios that \
the registry doesn't cover at umbrella level.

---

"""

# ---------------------------------------------------------------------------
# Direction-agnostic framing: same invariant as entity / award. The
# description is always positive-presence even when the user's intent
# was to exclude; a separate execution layer handles exclusion logic.
# ---------------------------------------------------------------------------

_DIRECTION_AGNOSTIC = """\
POSITIVE-PRESENCE INVARIANT

Every description you receive describes what to search FOR, never \
what to search AGAINST. If the user's original intent was to exclude \
a studio, the description has already been rewritten in positive- \
presence form and a separate execution layer handles the exclusion \
logic on the result set. Produce a spec that identifies movies \
containing this studio, period. Do not invert, do not search for \
"everyone except X".

---

"""

# ---------------------------------------------------------------------------
# Brand vs freeform decision framework. Principle-based: teach the
# boundary, don't give a keyword list.
# ---------------------------------------------------------------------------

_BRAND_VS_FREEFORM = """\
BRAND vs FREEFORM_NAMES

Set brand when the user is asking at the umbrella / parent-brand \
level and the registry below covers that brand. Umbrella intent \
means "the whole catalog of this studio's productions, across \
subsidiaries and historical sub-labels." Examples: "Disney movies", \
"Warner Bros. films", "A24 indies", "MGM catalog", "Ghibli" (the \
registry covers it), "Marvel Studios films" (its own registry entry \
AND a member of the DISNEY umbrella — pick marvel-studios for the \
narrow reading, disney for the broad reading).

The brand path respects time-bounded ownership at ingest — a movie \
is only stamped with a brand if the movie's release year falls \
inside that member-company's active window. "Disney" does NOT match \
Star Wars (1977) even though Lucasfilm is now Disney-owned, because \
Lucasfilm joined the Disney brand in 2012. You do not need to \
reason about this; just pick the brand. The data handles the rest.

Set freeform_names when the query names:
- A specific sub-label not in the registry ("Walt Disney Animation \
Studios" specifically — the disney brand would return the whole \
Disney catalog, which is too broad; "HBO Documentary Films", "Fox \
Searchlight" before its 2020 rename).
- A long-tail or niche studio absent from the registry ("Villealfa \
Filmproductions", "Cannon Films", "Carolco", "Shochiku").
- A foreign studio the user named in its native surface form.

Emit up to 3 surface forms. The right forms to pick are the variants \
most likely to appear verbatim in IMDB's production_companies \
credits for films associated with that studio. Good covering set:
- one condensed / acronym form ("MGM", "HBO", "BBC")
- one expanded / full form ("Metro-Goldwyn-Mayer", "Home Box Office")
- one alternate well-known variant when a distinct form exists \
("20th Century Studios" vs "Twentieth Century Fox" — but remember \
these live under brand=twentieth-century if an umbrella query)

Emit fewer than 3 when fewer distinct forms exist. Do not pad with \
variations that are just spelling or capitalization changes — \
normalization at execution already handles those. Do not emit \
semantic translations of a name ("Japan Broadcasting Corporation" \
for NHK) unless that translation is a form IMDB actually uses in \
credits.

Leave brand null when you use freeform_names. Leave freeform_names \
null (or empty) when you use brand. Both set at once is allowed but \
only has effect when the brand path returns empty — execution will \
then try freeform_names as a fallback.

---

"""

# ---------------------------------------------------------------------------
# Brand registry — rendered from ProductionBrand + its member companies.
# Prompt text wraps the rendered block so the model sees it as an
# authoritative table.
# ---------------------------------------------------------------------------

_BRAND_REGISTRY = f"""\
REGISTRY BRANDS

The closed set of brand values you may emit, each line followed by \
a sample of IMDB `production_companies` surface forms that count as \
that brand. The form list is a sample, not exhaustive — umbrella \
brands cover more strings than shown. Use this table to decide (a) \
which brand to pick for umbrella queries, and (b) whether a query \
is actually umbrella-level at all (if no registry brand covers the \
user's phrasing, use freeform_names instead).

{render_brand_registry_for_prompt()}

---

"""

# ---------------------------------------------------------------------------
# Freeform canonicalization — what a "good surface form" looks like.
# Focus on *what IMDB actually stores*, not what the user typed.
# ---------------------------------------------------------------------------

_NAME_CANONICALIZATION = """\
FREEFORM NAME CANONICALIZATION

Each freeform_name you emit is treated as a phrase that should match \
an IMDB production_companies string. Execution normalizes (lowercase, \
diacritic fold, punctuation strip, collapse whitespace, rewrite \
numeric ordinals to word form — `20th` → `twentieth`), splits on \
whitespace AND hyphens, and intersects the resulting discriminative \
tokens against the token index. Both sides run the same normalizer, \
so you do NOT need to pre-normalize. Emit the natural surface form \
the studio is known by.

Emit each form the way IMDB would store it, not the way the user \
typed it. "WB" rarely appears standalone in IMDB credits — emit \
"Warner Bros." instead (and you'd use brand=warner-bros for that \
anyway). "Mouse House" never appears — don't emit it; use \
brand=disney. Hyphenated and spaced variants (`Tri-Star` vs \
`TriStar`) are handled by tokenization; emit the form you believe \
is most common.

Do not include suffixes the studio doesn't use in credits. "A24" is \
how A24 is credited — don't expand it to "A24 Films LLC". \
"Ghibli" expands to "Studio Ghibli" because that is the credited \
form. Use your knowledge of IMDB's conventions for the specific \
studio.

---

"""

# ---------------------------------------------------------------------------
# Output field guidance: schema order, cognitive scaffolding in front.
# ---------------------------------------------------------------------------

_OUTPUT = """\
OUTPUT FIELD GUIDANCE

Generate fields in the schema's order. The thinking field scaffolds \
the decisions that follow it — reason first, commit after.

thinking — One or two sentences. FIRST: state whether the query is \
asking at umbrella / parent-brand level (a registry brand covers \
it) or at a specific sub-label / long-tail level (no registry brand \
cleanly covers it). SECOND: either name the registry brand whose \
slug you will emit, or list the 1-3 IMDB surface forms you will \
emit. This field is where the brand-vs-freeform decision is made; \
the structured fields just encode that decision.

brand — The registry brand slug when the query is umbrella-level. \
Leave null when you are using freeform_names.

freeform_names — Up to 3 surface forms when the query is specific or \
long-tail. Leave null or empty when you are using brand. Follow \
the canonicalization guidance: each entry should be a form that \
plausibly appears verbatim in IMDB's production_companies credits \
for films associated with this studio.
"""

SYSTEM_PROMPT = (
    _TASK
    + _DIRECTION_AGNOSTIC
    + _BRAND_VS_FREEFORM
    + _BRAND_REGISTRY
    + _NAME_CANONICALIZATION
    + _OUTPUT
)


async def generate_studio_query(
    intent_rewrite: str,
    description: str,
    routing_rationale: str,
    provider: LLMProvider,
    model: str,
    **kwargs,
) -> tuple[StudioQuerySpec, int, int]:
    """Translate one studio dealbreaker or preference into a StudioQuerySpec.

    The LLM receives the step 1 intent_rewrite (for disambiguation
    context) and one step 2 item's description plus routing_rationale.
    It produces either a brand enum value (for umbrella queries) or
    up to 3 freeform_names (for sub-labels and long-tail studios).

    Args:
        intent_rewrite: The full concrete statement of what the user is
            looking for, from step 1.
        description: The positive-presence statement of the studio
            requirement to translate (from a Dealbreaker or Preference).
        routing_rationale: The concept-type label from step 2 explaining
            why this item was routed to the studio endpoint.
        provider: Which LLM backend to use. No default — callers must
            choose explicitly so call sites are self-documenting and
            we can A/B test providers.
        model: Model identifier for the chosen provider. No default
            for the same reason as provider.
        **kwargs: Provider-specific parameters forwarded directly to
            the underlying LLM call (e.g., reasoning_effort,
            temperature, budget_tokens).

    Returns:
        A tuple of (StudioQuerySpec, input_tokens, output_tokens).
    """
    # TODO: When the stage-3 orchestrator introduces a shared request
    # model (one Pydantic class per endpoint call batching all step-2
    # item fields), move these strip + non-empty checks into that
    # model via `constr(strip_whitespace=True, min_length=1)` and
    # delete the manual validation here. Matches the pattern in
    # entity_query_generation.py / award_query_generation.py.
    intent_rewrite = intent_rewrite.strip()
    description = description.strip()
    routing_rationale = routing_rationale.strip()
    if not intent_rewrite:
        raise ValueError("intent_rewrite must be a non-empty string.")
    if not description:
        raise ValueError("description must be a non-empty string.")
    if not routing_rationale:
        raise ValueError("routing_rationale must be a non-empty string.")

    # Labelled sections so the model can keep the three inputs distinct.
    user_prompt = (
        f"intent_rewrite: {intent_rewrite}\n"
        f"description: {description}\n"
        f"routing_rationale: {routing_rationale}"
    )

    response, input_tokens, output_tokens = await generate_llm_response_async(
        provider=provider,
        user_prompt=user_prompt,
        system_prompt=SYSTEM_PROMPT,
        response_format=StudioQuerySpec,
        model=model,
        **kwargs,
    )

    return response, input_tokens, output_tokens

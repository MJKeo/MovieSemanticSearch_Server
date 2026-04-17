# Search V2 — Stage 3 Award Endpoint: Query Translation
#
# Translates one award dealbreaker or preference from step 2 into a
# concrete AwardQuerySpec that execution code can run against
# `movie_awards` (or the `award_ceremony_win_ids` fast path on
# `movie_card`). The LLM is a schema translator, not a
# re-interpreter: routing and intent have already been resolved
# upstream. Its job is to (1) inventory which filter axes the
# concept signals, (2) classify the count/intensity pattern for
# scoring, (3) map user-facing ceremony names to their canonical
# stored strings, and (4) produce exact category and prize-name
# strings the retrieval layer can match against.
#
# See search_improvement_planning/finalized_search_proposal.md
# (Step 3 → Endpoint 3: Awards) for the full design rationale and
# search_improvement_planning/full_search_capabilities.md
# (§2 movie_awards, §1.2 award_ceremony_win_ids) for the data
# surface.
#
# Prompt authoring conventions applied:
# - Evidence-inventory reasoning (concept_analysis inventories
#   filter-axis signals with explicit-absence discipline;
#   scoring_shape_label is a brief classification label, not a
#   reasoning trace)
# - Brief pre-generation fields (scoring_shape_label follows the
#   value_intent_label pattern — no consistency-coupling instruction)
# - Cognitive-scaffolding field ordering (filter evidence →
#   scoring classification → scoring values → filter values)
# - Principle-based constraints, not failure catalogs (scoring
#   patterns defined by what they ARE, not by keyword shortcuts)
# - Evaluation guidance over outcome shortcuts (explicit-absence
#   required on all filter axes; no keyword-shortcut rules like
#   "if 'Oscar' appears → winner")
# - No schema/implementation details leaked (no column names,
#   GIN indexes, or execution-side fast-path logic)

from datetime import date

from implementation.llms.generic_methods import LLMProvider, generate_llm_response_async
from schemas.award_category_tags import render_taxonomy_for_prompt
from schemas.award_surface_forms import (
    render_award_name_surface_forms_for_prompt,
    render_ceremony_mappings_for_prompt,
)
from schemas.award_translation import AwardQuerySpec

# ---------------------------------------------------------------------------
# System prompt — modular sections concatenated at module level.
#
# Structure: task → positive-presence invariant → scoring shape
# (modes + five canonical patterns) → filter axes (ceremonies,
# award_names, categories, outcome, years) → canonical surface
# forms (generated from the prize registry) → Razzie handling →
# output field guidance.
# ---------------------------------------------------------------------------

_TASK = """\
You translate one award requirement into a concrete award query \
specification. You receive a single requirement that has already \
been interpreted, routed, and framed as a positive-presence lookup. \
Your job is not to decide what the user meant — that is already \
done. Your job is to identify which award filter axes the \
requirement targets, classify its count/intensity pattern, and \
produce the exact parameters the retrieval layer needs.

Inputs you receive:
- intent_rewrite — the full concrete statement of what the user is \
looking for. Use it to disambiguate the requirement when the \
description alone is ambiguous (for example, "award-winning" is \
easier to classify as generic vs. ceremony-specific when you can \
see the surrounding query context).
- description — the single award requirement you are translating, \
always written in positive-presence form ("is award-winning", \
"won the Oscar for Best Picture", "nominated at Cannes", \
"preferably multiple award wins").
- routing_hint — a brief concept-type hint derived from upstream \
routing. It can help with borderline ambiguity, but it is \
background context rather than evidence.
- today — the current date in YYYY-MM-DD. Use it whenever the \
description or intent_rewrite references a relative year term \
("recent award winners", "this decade", "this year's Oscars"). Do \
not rely on outside knowledge of the current date.

Use description as the primary evidence for what to translate. \
intent_rewrite provides broader query context. routing_hint is a \
lightweight contextual cue, not a definition — do not let it \
override the actual request text.

Trust the upstream routing. If the description looks like it might \
fit another endpoint, still produce the best possible award \
specification for it — do not refuse, do not swap endpoints, do \
not reinterpret. Your output drives two mechanisms: a filter on \
structured award data, and a scoring formula that converts a count \
of matching award rows into a [0, 1] score. Precise strings and \
faithful pattern classification matter.

---

"""

# ---------------------------------------------------------------------------
# Positive-presence invariant: same pattern as the entity, metadata,
# and franchise endpoints. The LLM never inverts or negates.
# ---------------------------------------------------------------------------

_DIRECTION_AGNOSTIC = """\
POSITIVE-PRESENCE INVARIANT

Every description you receive describes what to search FOR, never \
what to search AGAINST. If the user's original intent was to \
exclude award-winning films, the description has already been \
rewritten in positive-presence form and a separate execution layer \
handles the exclusion logic on the result set. You always produce \
a spec that identifies movies whose award data matches the \
requirement. Do not invert, negate, or search for movies that lack \
awards.

---

"""

# ---------------------------------------------------------------------------
# Scoring shape: the most complex decision in this schema. Covers
# the two modes, the count unit, and the five canonical patterns.
# Principle-based: each pattern is defined by what it IS, not by
# a keyword shortcut.
# ---------------------------------------------------------------------------

_SCORING_SHAPE = """\
SCORING SHAPE

Every award spec has a scoring mode and a scoring mark. Together \
they convert a raw count of matching award rows into a [0, 1] \
score.

Count unit: distinct prize rows. Different ceremony, category, \
prize name, or year each count separately. "Won 11 Oscars" is 11 \
rows, not 1 ceremony.

Two modes:

floor — Binary. A movie scores 1.0 if its matching row count is \
at least scoring_mark; otherwise 0.0. Use when the user wants a \
hard threshold: a specific ceremony, a specific category, an \
explicit count floor ("at least 3 wins"), or any filter where \
presence-or-absence is the right answer.

threshold — Gradient. A movie scores min(row_count, \
scoring_mark) / scoring_mark — it ramps from 0.0 at zero rows to \
1.0 at scoring_mark rows, then holds at 1.0. Use when the user \
wants more award wins to produce a higher score up to a saturation \
point: generic "award-winning" queries, superlative language, or \
qualitative-intensity language.

The key distinction: a generic award concept with no ceremony, \
prize-name, or category filter calls for threshold — more wins \
should produce a higher score, because a film with one win is less \
"award-winning" than a film with ten. A specific filter ("Oscar \
Best Picture winner") calls for floor — the user wants binary \
presence, not a gradient.

FIVE CANONICAL PATTERNS

Before committing to scoring_mode and scoring_mark, classify the \
requirement into one of these five patterns. Evaluate the full \
description and intent_rewrite before classifying — do not anchor \
on the first intensity-like word you see.

generic award-winning — No specific ceremony, category, or prize \
name is present. Language like "award-winning", "award-winning \
films", "critically decorated", "won awards". Use threshold / 3. \
The gradient rewards films with more wins; 3 is the saturation \
point for a generically well-decorated film. Important: "Oscar- \
winning" is NOT this pattern — a specific prize filter is present, \
so it is specific-filter-no-count.

specific filter, no count — A ceremony, category, or prize name \
is present, but no count language. "Oscar-winning", "won the \
Palme d'Or", "Best Picture winner", "BAFTA-nominated", "won at \
Cannes". Use floor / 1. The specific award is the whole criterion; \
counting further wins beyond one adds nothing.

explicit count: N — The user names a minimum number. "At least 3 \
wins", "won 5 Oscars", "won twice". Use floor / N, where N is the \
user's number. For "multiple" with no explicit number, use \
floor / 2.

superlative — The user frames the query as a ranking or maximizing \
intent. "Most decorated", "most award-winning", "has the most \
Oscars", "best-decorated of all time". Use threshold / 15. The \
high mark ensures the score separates a film with 5 wins from one \
with 15 wins.

qualitative plenty — Intensity language implying many wins without \
a superlative. "Heavily decorated", "loaded with awards", "swept \
the ceremony", "multiple award wins" (where no specific number is \
given). Use threshold / 5. More moderate saturation than \
superlative.

---

"""

# ---------------------------------------------------------------------------
# Filter axes: the five independent dimensions of the award filter.
# Cartesian OR within each array; AND across arrays. Includes the
# ceremony name mapping table (user-facing names → stored strings)
# and the prize-name / category conventions.
# ---------------------------------------------------------------------------

_FILTER_AXES = (
    """\
FILTER AXES

Five filter axes are available. Populate only the ones the \
requirement actually signals. Leave the rest null. When multiple \
axes are populated, a movie must match every one of them (AND \
semantics). Within a list field, a movie matches if it satisfies \
any entry in the list (OR semantics).

CEREMONIES

The twelve tracked ceremonies and their stored string values. When \
a ceremony is named in the requirement, emit its stored string \
exactly as shown below. The stored strings are exact — a \
one-character difference produces zero matches.

"""
    + render_ceremony_mappings_for_prompt()
    + """

Emit multiple entries when more than one ceremony is named. Leave \
null when no ceremony is named — null means all non-Razzie \
ceremonies apply (see Razzie Handling).

Use ceremonies for event/festival/awards-body wording such as \
"at Cannes", "nominated at Sundance", or "Academy Awards \
ceremony". Do not use ceremonies as a proxy for a named prize \
object.

AWARD NAMES

The specific prize name as stored in the award database — distinct \
from the ceremony name. The ceremony is the event; the prize is the \
individual award object granted at that event. Common prize names: \
"Oscar", "Palme d'Or", "Golden Globe", "BAFTA Film Award", "Golden \
Lion", "Golden Bear", "Silver Bear", "Jury Prize".

Emit when the user names the specific prize object, even if that \
prize implies a ceremony. Represent what the user actually asked \
for at the most direct level. "Oscar-winning", "won an Oscar", \
"Palme d'Or winners", "Golden Lion winner", and "won a Golden \
Globe" should populate award_names. Do NOT automatically add the \
related ceremony just because the prize belongs to one.

Use both award_names and ceremonies only when the query explicitly \
contains both levels or clearly needs both to represent the wording \
faithfully. Example: "Cannes Palme d'Or winners" can emit both \
because the query names both the festival and the prize. A ceremony \
mention alone does not imply a prize name, and a prize name alone \
does not imply a ceremony filter.

CATEGORY TAGS

The award category axis is a closed enum of concept tags drawn from \
a 3-level taxonomy (leaf → mid → group). The taxonomy is listed in \
the CATEGORY TAG TAXONOMY section below. Pick tags at whatever \
specificity matches the requirement:

  leaf level (e.g. lead-actor, best-picture-drama, worst-screenplay) — \
    a single, specific concept. Use when the user names the exact \
    category, including its narrow form ("Best Actor", "Best \
    Adapted Screenplay", "Best Animated Short Film").

  mid level (e.g. lead-acting, music, short, worst-acting) — \
    a meaningful intermediate rollup that spans multiple leaves \
    without spanning the whole group. Use when the requirement is \
    deliberately broader than a leaf but narrower than the group, \
    typically when it's gender-neutral, format-neutral, or \
    medium-neutral. "Won Best Actor or Best Actress" → lead-acting. \
    "Won any sound award" → sound-any. "Any short film award" → short.

  group level (acting, directing, writing, picture, craft, razzie, \
    festival-or-tribute) — the whole bucket. Use when the requirement \
    is generic to the bucket. "Won an acting award" → acting. \
    "Recognized for craft work" → craft.

Emit multiple tags when the requirement covers concepts that don't \
share an ancestor below the group level, e.g. "won Best Director or \
Best Screenplay" → [director, screenplay-any]. Within an axis a row \
matches if it overlaps with ANY of the supplied tags. The retrieval \
layer uses the row's stored ancestor list, so emitting a group-level \
tag automatically picks up every leaf and mid under that group — do \
NOT enumerate the descendants of a tag you've already emitted.

Tag selection is ceremony-agnostic. The same leaf tag (e.g. \
lead-actor) covers "Best Actor" at the Oscars, "Best Performance by \
an Actor in a Motion Picture - Drama" at the Globes, "Best Leading \
Actor" at BAFTA, and dozens of other phrasings. You no longer need \
to emit ceremony-specific surface forms for the category axis.

Leave category_tags null when the requirement names no category at \
all (a pure ceremony query like "Cannes winners" should leave \
category_tags null and rely on the ceremony filter).

OUTCOME

winner, nominee, or null. Null means both winners and nominees \
count toward the row count. Populate only when the user explicitly \
distinguishes wins from nominations:
- "won", "winner", "winning" → winner
- "nominated", "nomination", "nominee" → nominee
- No explicit outcome language → null (both count)

"Award-winning" means winner. "Award-nominated" means nominee. \
A requirement with no outcome signal (e.g., "recognized at \
Sundance") → null.

YEARS

A year range (or single year). Null means any year applies. \
Populate when the user names a specific year, decade, or era \
("2023 Oscars", "late-90s Sundance films", "since 2010"). Resolve \
relative terms against the supplied today date, not against \
training-time knowledge: "this decade" is the current decade start \
to today; "recently" is roughly the last 2-3 years ending today; \
"this year" is today's calendar year. Use calendar years, not \
award-ceremony season numbers.

---

"""
)

# ---------------------------------------------------------------------------
# Award-name canonical surface forms. Categories are expressed via the
# closed CategoryTag enum; this section covers only the award_names axis,
# where exact stored strings are still required. Rendered programmatically
# so the prompt cannot drift from the canonical registry.
# ---------------------------------------------------------------------------

_SURFACE_FORMS = (
    """\
AWARD NAME SURFACE FORMS

The retrieval layer matches award_names as exact, un-normalized \
strings against stored award rows. A one-character difference \
produces zero matches. When the user names a specific prize, emit \
the official IMDB surface form for that prize — matching \
capitalization, punctuation, and apostrophe style. The table below \
anchors common canonical forms.

The table is not a closed vocabulary. Do NOT restrict output to \
table entries. Do NOT pattern-match a user's phrase onto a \
similar-looking table entry when the user clearly named a different \
specific award. For example, a user asking for "Cannes Jury Prize" \
must emit "Jury Prize", not "Palme d'Or".

"""
    + render_award_name_surface_forms_for_prompt()
    + """

---

"""
)

# ---------------------------------------------------------------------------
# Tag taxonomy: rendered programmatically from schemas/award_category_tags.py
# so the prompt and the schema cannot drift. Each tag is listed under its
# group with a short description and (for leaves) its mid-level parents.
# ---------------------------------------------------------------------------

_TAG_TAXONOMY = (
    "CATEGORY TAG TAXONOMY\n\n"
    "The closed enum of category_tags values, organized by top-level \n"
    "group. Each leaf row also lists its rollup parents in [under: ...]; \n"
    "use a parent slug instead of a leaf when you want to match every \n"
    "concept under it (the row's stored tag list contains every ancestor, \n"
    "so a single ancestor tag covers all its descendants).\n\n"
    + render_taxonomy_for_prompt()
    + "\n\n---\n\n"
)

# ---------------------------------------------------------------------------
# Razzie handling: default exclusion + explicit opt-in. This is a
# unique execution concern surfaced at translation time so the LLM
# knows when to include vs. exclude the RAZZIE ceremony.
# ---------------------------------------------------------------------------

_RAZZIE_HANDLING = """\
RAZZIE HANDLING

The Razzie Awards celebrate the worst of cinema and are excluded \
by default from all award counts and filters — even when ceremonies \
is null. A generic "award-winning" query means positive-prestige \
awards only; Razzie wins do not make a film "award-winning" in the \
user's sense.

Razzies are included ONLY when the user explicitly names them. \
Clear signals: "Razzie winners", "Razzie-nominated", "won a \
Razzie", "Golden Raspberry Award", or any "Worst …" category \
("Worst Picture", "Worst Actor"). When you see these, emit \
"Razzie Awards" in the ceremonies list AND, if a worst-category \
was named, the corresponding worst-* tag in category_tags. Either \
signal alone is sufficient to opt in (the executor recognizes \
Razzie intent on either axis), but emitting both when both are \
present makes the spec self-documenting. You may combine Razzies \
with other ceremonies when the query covers both ("Oscar and \
Razzie winners").

Never infer Razzie intent. A user asking for "the worst movies" \
or "critically panned films" is NOT asking for Razzie data — they \
are asking for low-reception content, which routes to a different \
endpoint. Razzie intent requires the user to have explicitly named \
Razzies or Golden Raspberries.

---

"""

# ---------------------------------------------------------------------------
# Output field guidance: per-field instructions in schema order. The
# two reasoning fields carry their framing here so cognitive
# scaffolding produces its intended effect on the decisions that
# follow.
# ---------------------------------------------------------------------------

_OUTPUT = """\
OUTPUT FIELD GUIDANCE

Generate fields in the schema's order. The two reasoning fields \
come first — they scaffold the decisions that follow. Surface \
filter-axis evidence, then classify the scoring pattern, then \
commit to literal values.

concept_analysis — FIRST field. A filter-axis evidence inventory. \
For each of the five filter axes (ceremony, award name, category, \
outcome, year), quote the phrase from description that signals it. \
Use intent_rewrite only when description is ambiguous and the \
broader context genuinely clarifies the axis. For every axis, either \
cite evidence or state explicitly that no signal is present \
("no ceremony signal", "no category signal"). \
Never cite routing_hint here.
Explicit absence is required — it prevents over-assignment and \
calibrates you against populating axes that have no evidence.

Count and intensity language ("award-winning", "heavily decorated", \
"at least 3 wins", "most decorated") does NOT belong here. It \
belongs in scoring_shape_label. If the same phrase contributes both \
a filter signal and a scoring signal, quote it under its filter \
axis in concept_analysis and use it again in scoring_shape_label. \
Repeating it is correct.

Keep this concise — one sentence per axis is enough. The goal is \
a quick inventory of what is and is not present.

scoring_shape_label — SECOND field, immediately before \
scoring_mode. A brief label naming the scoring pattern. Pick \
exactly one of the five canonical labels:
  "generic award-winning"
  "specific filter, no count"
  "explicit count: N"  (replace N with the actual number)
  "superlative"
  "qualitative plenty"

This is a label, not a sentence or justification. Do not write \
"the user wants a generic award-winning result" — write \
"generic award-winning". The label primes the two numeric fields \
that follow.

scoring_mode — floor or threshold. Follows from the scoring pattern \
named in scoring_shape_label per the five-pattern table in the \
Scoring Shape section.

scoring_mark — The integer calibration mark for the pattern. \
Follows from the pattern table: 3 for generic award-winning, 1 for \
specific filter, N for explicit count, 15 for superlative, 5 for \
qualitative plenty. Must be at least 1.

ceremonies — Null when concept_analysis found no ceremony signal. \
When a ceremony is named, emit its exact stored string from the \
ceremony table. Emit multiple entries when multiple ceremonies are \
named. Prefer null over an empty list.

award_names — Null when concept_analysis found no prize-name \
signal. Emit the official IMDB surface form of the prize (e.g., \
"Oscar", "Palme d'Or", "Golden Lion", "BAFTA Film Award") when a \
specific prize was named — see AWARD NAME SURFACE FORMS. When the \
prize is not in that table, use your knowledge of IMDB nomenclature \
for the relevant ceremony; do not approximate to a similar-looking \
table entry. Do not automatically add a ceremony filter just because \
the prize belongs to one. Emit multiple when the same prize has \
been known by different names over time.

category_tags — Null when concept_analysis found no category signal. \
Otherwise emit one or more tags from the closed CategoryTag enum at \
whatever specificity the requirement implies (see CATEGORY TAGS in \
FILTER AXES and the CATEGORY TAG TAXONOMY for the full enum). \
Prefer the broadest tag that exactly captures the requirement: if \
the user said "Best Actor", emit lead-actor (leaf), not lead-acting \
(mid). If they said "won Best Actor or Best Actress", emit \
lead-acting (mid), not both leaves. If they said "won an acting \
award", emit acting (group), not every acting tag. Do not enumerate \
descendants of a tag you've already emitted — the row's stored tag \
list contains every ancestor, so an ancestor tag automatically picks \
up its descendants. You may emit multiple tags only when the \
concepts don't share an ancestor below the group level (e.g. \
"Best Director or Best Screenplay" → [director, screenplay-any]).

outcome — Null when concept_analysis found no explicit outcome \
signal. Emit winner or nominee only when outcome language was \
cited.

years — Null when concept_analysis found no year signal. When a \
year or range was cited, emit year_from and year_to as integers. \
Use the same value for both when a single year was named.\
"""

SYSTEM_PROMPT = (
    _TASK
    + _DIRECTION_AGNOSTIC
    + _SCORING_SHAPE
    + _FILTER_AXES
    + _SURFACE_FORMS
    + _TAG_TAXONOMY
    + _RAZZIE_HANDLING
    + _OUTPUT
)


async def generate_award_query(
    intent_rewrite: str,
    description: str,
    routing_rationale: str,
    today: date,
    provider: LLMProvider,
    model: str,
    **kwargs,
) -> tuple[AwardQuerySpec, int, int]:
    """Translate one award dealbreaker or preference into an AwardQuerySpec.

    The LLM receives the step 1 intent_rewrite (for disambiguation
        context), one step 2 item's description plus routing_rationale, and
        today's date (for resolving relative year terms). It produces the
    exact award filter parameters and scoring shape the execution layer
    needs.

    Args:
        intent_rewrite: The full concrete statement of what the user is
            looking for, from step 1.
        description: The positive-presence statement of the award
            requirement to translate (from a Dealbreaker or Preference).
        routing_rationale: The concept-type label from step 2 explaining
            why this item was routed to the awards endpoint. Exposed to
            the prompt as routing_hint to reinforce that it is
            background context rather than evidence.
        today: The current date. Passed explicitly so callers control
            what "today" means (production uses date.today(); offline
            analysis can pin a fixed date for reproducibility).
        provider: Which LLM backend to use. No default — callers must
            choose explicitly so call sites are self-documenting and
            we can A/B test providers.
        model: Model identifier for the chosen provider. No default
            for the same reason as provider.
        **kwargs: Provider-specific parameters forwarded directly to
            the underlying LLM call (e.g., reasoning_effort,
            temperature, budget_tokens).

    Returns:
        A tuple of (AwardQuerySpec, input_tokens, output_tokens).
    """
    # TODO: When the stage-3 orchestrator introduces a shared request
    # model, move these checks into that model's validation and delete
    # them here. See entity_query_generation.py for the same note.
    intent_rewrite = intent_rewrite.strip()
    description = description.strip()
    routing_hint = routing_rationale.strip()
    if not intent_rewrite:
        raise ValueError("intent_rewrite must be a non-empty string.")
    if not description:
        raise ValueError("description must be a non-empty string.")
    if not routing_hint:
        raise ValueError("routing_rationale must be a non-empty string.")

    # All four inputs are required. Present as labeled sections so the
    # model can keep them distinct. Today's date is rendered in ISO
    # form for unambiguous parsing.
    user_prompt = (
        f"intent_rewrite: {intent_rewrite}\n"
        f"description: {description}\n"
        f"routing_hint: {routing_hint}\n"
        f"today: {today.isoformat()}"
    )

    response, input_tokens, output_tokens = await generate_llm_response_async(
        provider=provider,
        user_prompt=user_prompt,
        system_prompt=SYSTEM_PROMPT,
        response_format=AwardQuerySpec,
        model=model,
        **kwargs,
    )

    return response, input_tokens, output_tokens

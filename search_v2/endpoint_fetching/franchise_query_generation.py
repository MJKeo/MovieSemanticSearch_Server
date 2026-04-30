# Search V2 — Stage 3 Franchise Endpoint: Query Translation
#
# Translates one franchise dealbreaker or preference from step 2 into
# a concrete FranchiseQuerySpec. The LLM is a schema translator, not a
# re-interpreter: routing and intent have already been resolved upstream.
# Its job is to (1) inventory which franchise axes the concept signals,
# (2) produce canonical strings that can match the exact stored forms
# after shared normalization, and (3) set structural flags and launch
# scope consistently with the ingest-side definitions.
#
# See search_improvement_planning/finalized_search_proposal.md
# (Step 3 → Endpoint 4: Franchise Structure) for the full design
# rationale. Canonical-naming, subgroup, spinoff, crossover, and
# launched_franchise definitions are inherited from the ingest-side
# generator (movie_ingestion/metadata_generation/prompts/franchise.py)
# so the two LLMs agree on what lives in each slot.

from implementation.llms.generic_methods import LLMProvider, generate_llm_response_async
from schemas.franchise_translation import FranchiseQuerySpec

# ---------------------------------------------------------------------------
# System prompt — modular sections concatenated at module level.
#
# Structure: task → positive-presence invariant → input authority →
# searchable axes → canonical naming → scope discipline → output
# field guidance.
#
# Prompt authoring conventions applied:
# - Evidence-inventory reasoning (concept_analysis quotes signal
#   phrases and explicit absences)
# - Cognitive-scaffolding field ordering (analysis → names →
#   subgroup → lineage position → structural flags → launch scope)
# - Principle-based constraints, not failure catalogs
# - Evaluation guidance over outcome shortcuts
# - Ingest/search alignment for subgroup and naming rules
# ---------------------------------------------------------------------------

_TASK = """\
You translate one franchise requirement into a concrete franchise \
query specification. You receive a single requirement that has \
already been interpreted, routed, and framed as a positive-presence \
lookup. Your job is not to decide what the user meant — that is \
already done. Your job is to pick exactly the franchise axes the \
requirement targets and produce the literal parameters those axes \
need.

Inputs you receive:
- intent_rewrite — the full concrete statement of what the user is \
looking for. Use it only to contextualize the description when the \
description is vague or underspecified.
- description — the single franchise requirement you are \
translating, always written in positive-presence form ("is a Marvel \
movie", "is a sequel", "spinoff movies", "movies that launched a \
subgroup"). This is the authoritative statement of what to translate.
- route_rationale — a concept-type label explaining why this \
requirement was routed to this endpoint. Treat it as a hint, not as \
evidence.

Trust the upstream routing. If the description looks like it might \
fit another endpoint, still produce the best possible franchise \
lookup for it — do not refuse, do not swap endpoints, do not \
reinterpret. Precise strings and faithful axis selection are what \
matter.

---

"""

_DIRECTION_AGNOSTIC = """\
POSITIVE-PRESENCE INVARIANT

Every description you receive describes what to search FOR, never \
what to search AGAINST. If the user's original intent was to \
exclude a franchise, role, or structural trait, the description has \
already been rewritten in positive-presence form and a separate \
execution layer handles the inclusion vs. exclusion logic on the \
result set. You always produce a spec that identifies movies whose \
franchise data matches the requirement.

---

"""

_INPUT_AUTHORITY = """\
INPUT AUTHORITY

Use the inputs in this order:

1. description — authoritative for axis selection. Choose which \
   axes to populate from the requirement stated here.
2. intent_rewrite — contextualizes the description when the \
   description is vague. Use it to resolve referents, not to add \
   extra franchise constraints that the description itself did not \
   ask for.
3. route_rationale — background hint only. Do not let it override \
   the actual evidence in description. If the hint leans one way \
   but the description clearly points another way, trust the \
   description.

When writing concept_analysis, extract evidence from description \
first. Bring in intent_rewrite only when it clarifies what a vague \
phrase in description refers to. Do not treat route_rationale as \
evidence.

---

"""

_SEARCHABLE_AXES = """\
SEARCHABLE AXES

Six axes are available. The first five are retrieval axes and all \
are independent — populate only the ones the concept actually \
signals, leave the rest null. A subgroup-only spec ("trilogies"), \
a structural-only spec ("spinoffs"), and a named-franchise spec \
("Marvel movies") are all valid. Multiple axes may populate \
naturally when one concept spans them (for example a named \
franchise plus a structural role). The sixth axis (prefer_lineage) \
is a scoring bias, not a retrieval axis.

franchise_or_universe_names — The named franchise, IP, or shared \
cinematic universe the user is searching for. Signals include \
proper-noun franchise references such as "James Bond", "Star Wars", \
"Harry Potter", "Marvel", "Marvel Cinematic Universe", or \
"MonsterVerse". At ingest time a franchise name is stored as \
either the lineage (the specific-title slot) or the shared_universe \
(the umbrella slot) — retrieval covers both from this single \
field, so you do NOT need to predict which slot the stored value \
lives in. The lineage-vs-universe distinction is consumed by \
prefer_lineage below to bias scoring, not to restrict matches. \
Populate whenever a named franchise is part of the requirement. \
Leave null when the concept is purely structural or purely \
subgroup-based with no named franchise.

recognized_subgroups — A named subgroup: phase, saga, trilogy, \
timeline, director-era slice, or other widely-used sub-lineage \
label. Pull this guidance from the same standards used at ingest:
- valid examples include labels such as "phase one", "phase three", \
  "infinity saga", "multiverse saga", "kelvin timeline", \
  "snyderverse", "disney live-action remakes", "sequel trilogy", \
  "skywalker saga"
- do not restate the franchise itself as a subgroup
- do not invent labels on the spot
- populate whenever the user's requirement actually targets a \
  recognized subgroup — INCLUDING standalone subgroup concepts like \
  "trilogies" or "phase one movies" where no parent franchise is \
  named. This axis is independent; it does not require \
  franchise_or_universe_names to be populated.

lineage_position — The narrative-position enum with four values: \
sequel, prequel, remake, reboot. Sequel and prequel are the common \
cases. Remake is rare at this endpoint and belongs here only for a \
franchise-specific remake concept rather than a broad remake query.

structural_flags — Optional list of structural traits. Available \
values:
- spinoff — use when the requirement targets branch entries rather \
  than the main trunk
- crossover — use when the requirement targets films whose identity \
  is separate known entities or characters meeting or colliding

launch_scope — Distinguishes what kind of thing the movie launched:
- franchise — the movie launched a franchise
- subgroup — the movie launched a subgroup inside a broader \
  franchise

launch_scope=subgroup does NOT require the subgroup to be named. If \
the user asks for "movies that launched a subgroup", set \
launch_scope=subgroup even when recognized_subgroups stays null. \
Populate recognized_subgroups only when the user names the subgroup \
itself.

prefer_lineage — A scoring-bias bool. When true, movies whose \
lineage side matches the query name are scored higher than movies \
that match only via the shared_universe side. The match set is \
unchanged — spinoffs and universe-adjacent films still appear in \
the results, just ranked below main-line entries. Default false. \
Set true ONLY when all of the following hold:
- exactly one specific franchise is named (not an umbrella like \
  "Marvel" or "DC"), and that franchise has a main line plus known \
  spinoffs or universe-adjacent entries
- the requirement does not explicitly invite spinoffs or \
  universe-adjacent content
- franchise_or_universe_names is not a multi-name umbrella sweep
- recognized_subgroups is null — a named subgroup already \
  disambiguates intent, so layering prefer_lineage on top adds \
  noise

Positive examples (set prefer_lineage=true):
- "shrek movies" — main Shrek films upranked over Puss in Boots
- "john wick movies" — main John Wick films upranked over The \
  Continental
- "toy story movies" — main Toy Story films upranked over Forky \
  shorts or Lightyear
- "harry potter movies" — main Harry Potter films upranked over \
  Fantastic Beasts
- "the conjuring movies" — main Conjuring films upranked over \
  Annabelle / The Nun

Negative examples (leave prefer_lineage=false):
- "MCU movies" / "marvel movies" — the umbrella IS the request
- "DCU movies" / "dc movies" — same
- "star wars movies" — Star Wars is the umbrella, not a single \
  lineage
- "middle-earth movies" — umbrella shared universe
- "shrek spinoffs" — the user explicitly asked for the \
  non-lineage side
- "harry potter and fantastic beasts" — the user explicitly \
  invites the shared-universe entries
- "marvel phase one movies" — subgroup disambiguates; no \
  lineage preference needed
- any entry with two or more franchise_or_universe_names (umbrella \
  sweep by definition)

When in doubt, leave prefer_lineage=false. False is the safer \
default — it preserves the current behavior of scoring every match \
equally, and misclassifying true as false only loses a small \
ranking nudge. Misclassifying false as true when the user \
actually wanted the umbrella demotes content they wanted.

---

"""

_NAME_CANONICALIZATION = """\
CANONICAL NAMING

The names you emit are resolved through a shared tokenizer and a \
token inverted index. The tokenizer applies: lowercase, diacritic \
fold, punctuation strip, whitespace collapse, ordinal digit-to-word \
("20th" → "twentieth"), cardinal 0-99 digit-to-word ("phase 1" → \
"phase one"), whitespace + hyphen split, and stopword drop \
("the of and a in to on my i for at by with"). The same tokenizer \
ran at ingest time, so orthographic variants do not need to be \
enumerated by you — they collide automatically. Focus your \
alternates on genuinely different canonical forms.

Follow the same canonical naming rules used by the ingest-side \
franchise generator:

- use the most common, well-known canonical form
- lowercase
- spell digits as words
- expand "&" to "and"
- expand abbreviations only when the expanded form is also in common \
  use
  - "MCU" → "marvel cinematic universe"
  - "DCEU" → "dc extended universe"
  - "LOTR" → "the lord of the rings"
  - "monsterverse" stays "monsterverse"
  - "x-men" stays "x-men"
- for director-era subgroup labels, drop first names when the \
  surname alone is the common form
  - "Peter Jackson's Lord of the Rings Trilogy" → \
    "jackson lotr trilogy"
  - "John Carpenter Halloween films" → \
    "carpenter halloween films"

Specificity — pick the broadest or narrowest form based on the \
user's apparent intent:
- UMBRELLA query ("Marvel movies", "Lord of the Rings films") → \
  emit the broad form ("marvel cinematic universe", \
  "the lord of the rings")
- NARROW lineage inside an umbrella ("Doctor Strange", \
  "Captain America") → emit the narrow form alone. Every Doctor \
  Strange film is already MCU; adding "marvel cinematic universe" \
  as a second entry would OR-union the MCU sweep back in and \
  over-broaden the query.

For franchise_or_universe_names:
- emit 1 entry in the common case
- emit 2-3 entries ONLY when genuinely different canonical names are \
  in common use that the ingest side might plausibly have stored — \
  the extra entries perform an umbrella sweep via across-name union \
  (e.g. "marvel cinematic universe" plus "marvel" sweeps the MCU \
  proper PLUS every other "marvel"-tagged franchise entry; \
  "the lord of the rings" plus "middle-earth")
- do NOT pad the list with spelling, punctuation, casing, \
  hyphenation, diacritic, or digit-vs-word variants — the shared \
  tokenizer handles those symmetrically

For recognized_subgroups:
- apply the same rules
- only emit labels that studios, mainstream film criticism, or \
  widely-used fan terminology actually use
- do not invent a subgroup label just because one would be useful

Adding fake or speculative alternatives widens the query \
incorrectly. Be deliberate.

---

"""

_SCOPE_DISCIPLINE = """\
SCOPE DISCIPLINE

Every populated axis narrows the result set. A movie returned by \
this endpoint must match EVERY populated axis.

Populate only the axes the concept actually signals. Do not add \
extra axes just to describe the franchise more fully. If the user \
asks for "sequels", that is a lineage_position query. It is not an \
invitation to guess a franchise name.

If the item seems mildly misrouted, recover only the narrowest \
franchise interpretation that is still literally supported by the \
description plus minimal context. Do not guess franchise names or \
subgroup labels that are not actually supported.

---

"""

_OUTPUT = """\
OUTPUT FIELD GUIDANCE

Generate fields in the schema's order. concept_analysis comes \
first and scaffolds all later fields.

concept_analysis — FIRST field. An evidence inventory that grounds \
axis presence and absence, AND commits to query scope before any \
field values are emitted.

Begin by stating the scope: umbrella (broad franchise sweep) vs \
specific lineage inside an umbrella vs sub-phase of an umbrella vs \
pure subgroup with no named franchise vs pure structural. Scope \
drives whether you emit 1 vs 2-3 entries in \
franchise_or_universe_names, and whether you pair it with a \
subgroup, and whether it stays null entirely. Committing to scope \
up front avoids OR-broadening mistakes like emitting both \
"doctor strange" and "marvel cinematic universe" for a narrow \
request (which would union every MCU film back in).

After scope, quote signal phrases from description first. Use \
intent_rewrite only when it clarifies what a vague term in \
description refers to. For each relevant phrase, pair it with the \
axis it implicates:
- franchise / IP / cinematic-universe name → \
  franchise_or_universe_names
- named phase / saga / timeline / trilogy / subgroup label → \
  recognized_subgroups
- sequel / prequel / remake / reboot phrasing → lineage_position
- spinoff / branch / side-story phrasing → structural_flags=spinoff
- crossover / team-up / collision-of-separate-stories phrasing → \
  structural_flags=crossover
- launched / started / kicked off a franchise → launch_scope=franchise
- launched / started a subgroup / phase / saga / era / timeline → \
  launch_scope=subgroup
- naming exactly one specific franchise with known spinoffs or \
  universe-adjacent entries, with no phrasing inviting those \
  entries and no subgroup → prefer_lineage=true (otherwise \
  prefer_lineage=false)

If no phrase signals a given axis, say so explicitly. Do not leave \
absence implicit. This is an evidence inventory, not a justification.

franchise_or_universe_names — Up to three canonical franchise or \
shared-universe surface forms. Apply the Canonical Naming specificity \
rule: broad form for umbrella queries, narrow form alone for \
specific-lineage queries. Leave null when no named franchise is part \
of the requirement.

recognized_subgroups — Up to three canonical subgroup labels. \
Populate whenever a named subgroup is part of the requirement, even \
when franchise_or_universe_names is null (standalone subgroup \
queries like "trilogies" or "phase one movies" are valid).

lineage_position — One of sequel, prequel, remake, reboot, or null.

structural_flags — Null when no structural trait is requested. \
Otherwise emit a list containing one or both of:
- spinoff
- crossover

launch_scope — Null when launch behavior is not requested. Otherwise \
emit exactly one of:
- franchise
- subgroup

prefer_lineage — Emit true only when the concept_analysis captured \
a single specific franchise with known spinoffs or \
universe-adjacent entries AND the user did not invite those \
entries AND no subgroup is populated AND \
franchise_or_universe_names has exactly one entry. Otherwise emit \
false. When uncertain, emit false.

Remember the input hierarchy: description chooses the axes; \
intent_rewrite disambiguates vague references; route_rationale is \
just a hint.
"""

SYSTEM_PROMPT = (
    _TASK
    + _DIRECTION_AGNOSTIC
    + _INPUT_AUTHORITY
    + _SEARCHABLE_AXES
    + _NAME_CANONICALIZATION
    + _SCOPE_DISCIPLINE
    + _OUTPUT
)


async def generate_franchise_query(
    intent_rewrite: str,
    description: str,
    route_rationale: str,
    provider: LLMProvider,
    model: str,
    **kwargs,
) -> tuple[FranchiseQuerySpec, int, int]:
    """Translate one franchise dealbreaker or preference into a FranchiseQuerySpec.

    The LLM receives the step 1 intent_rewrite (for disambiguation
    context) and one step 2 item's description plus route_rationale.
    It produces the exact franchise query parameters the structured
    execution layer needs.

    Args:
        intent_rewrite: The full concrete statement of what the user is
            looking for, from step 1.
        description: The positive-presence statement of the franchise
            requirement to translate (from a Dealbreaker or Preference).
        route_rationale: The concept-type label from step 2 explaining
            why this item was routed to the franchise_structure endpoint.
        provider: Which LLM backend to use. No default — callers must
            choose explicitly so call sites are self-documenting and
            we can A/B test providers.
        model: Model identifier for the chosen provider. No default
            for the same reason as provider.
        **kwargs: Provider-specific parameters forwarded directly to
            the underlying LLM call (e.g., reasoning_effort,
            temperature, budget_tokens).

    Returns:
        A tuple of (FranchiseQuerySpec, input_tokens, output_tokens).
    """
    intent_rewrite = intent_rewrite.strip()
    description = description.strip()
    route_rationale = route_rationale.strip()
    if not intent_rewrite:
        raise ValueError("intent_rewrite must be a non-empty string.")
    if not description:
        raise ValueError("description must be a non-empty string.")
    if not route_rationale:
        raise ValueError("route_rationale must be a non-empty string.")

    user_prompt = (
        f"intent_rewrite: {intent_rewrite}\n"
        f"description: {description}\n"
        f"route_rationale: {route_rationale}"
    )

    response, input_tokens, output_tokens = await generate_llm_response_async(
        provider=provider,
        user_prompt=user_prompt,
        system_prompt=SYSTEM_PROMPT,
        response_format=FranchiseQuerySpec,
        model=model,
        **kwargs,
    )

    return response, input_tokens, output_tokens

# Search V2 — Stage 1: Flow Routing
#
# Produces one required primary intent plus up to two optional
# alternative intents. Stage 1 decides both the major search flow and
# whether additional searches would improve browsing value under
# ambiguity. This is the entry point of the V2 search pipeline — all
# queries pass through here before any decomposition or retrieval
# happens.
#
# See search_improvement_planning/finalized_search_proposal.md
# (Step 1: Flow Routing) for the full design rationale.

from implementation.llms.generic_methods import LLMProvider, generate_llm_response_async
from schemas.flow_routing import FlowRoutingResponse

# ---------------------------------------------------------------------------
# System prompt — modular sections concatenated at module level.
#
# Structure: task/outcome → core principles → vagueness vs ambiguity →
# flow definitions → reading-evidence hierarchy → branching policy →
# intent-rewrite discipline → inference policy → crude-language
# handling → boundary examples → creative spins → output field
# guidance.
#
# Prompt authoring conventions applied:
# - Evidence-inventory reasoning (cite concrete query text before
#   committing to branch shape or flow)
# - Brief pre-generation fields (ambiguity_analysis and
#   difference_rationale)
# - Explicit empty paths for optional list fields
# - Principle-based constraints, not failure catalogs
# - Field order that scaffolds the next decision for small models
# ---------------------------------------------------------------------------

_TASK_AND_OUTCOME = """\
You route movie search queries into the correct search flow and decide \
whether additional searches would improve the user's browsing \
experience. There are three flows — exact_title, similarity, and \
standard — each handled by a different downstream pipeline.

Always produce one primary_intent. Produce alternative_intents only \
when the query has true reading ambiguity — multiple competing \
interpretations of what the user is asking for. Productive sub-angles \
within a single broad intent belong to creative_alternatives, not \
alternative_intents.

Use the query text as the primary evidence. You may use movie \
knowledge to recognize typed titles, alternate titles, or well-known \
single-movie abbreviations, but not to invent exact movie titles from \
descriptions. Do not guess a title unless the query itself is already \
pointing to a literal title reference.

Your job is:
- choose the correct flow for each emitted branch
- decide whether branching is worthwhile
- rewrite each branch into a clear, faithful search statement

Your job is NOT:
- to decompose the query into ingredients, filters, or retrieval axes
- to add proxy qualities like "highly-rated", "important", "iconic", \
  or "prestigious" unless the user directly asked for that idea
- to turn a clue, example, or scene description into evaluative language
- to narrow a broad query to a more specific subtype unless the query \
  itself clearly does that

---

"""

_CORE_PRINCIPLES = """\
CORE PRINCIPLES

The primary_intent must be the single most likely interpretation of the \
query in movie-search context.

Alternatives are optional helpers. They are not coequal defaults. Do \
not put a less likely but more interesting interpretation into \
primary_intent.

Additional searches must be materially different. Never emit near-\
duplicate branches, wording variants, or paraphrases that would lead \
to essentially the same downstream search.

When branching, vary only the genuinely ambiguous part of the query. \
Keep the rest of the meaning fixed.

Do not decompose in Stage 1. Stage 1 routes and lightly rewrites. \
Later stages will decide which parts of the query are concepts, \
constraints, preferences, or routes.

---

"""

_VAGUENESS_VS_AMBIGUITY = """\
VAGUENESS VS AMBIGUITY

A query can be vague in two different ways. Only one is a reason to \
branch.

Semantic vagueness — one retrieval target with fuzzy edges. Words \
like "cozy", "feel-good", "epic", or "prestige" point at a single \
reading whose boundaries are soft. Later stages handle soft \
boundaries. Do not branch on this.

Reading ambiguity — two or more genuinely different retrieval \
targets hiding inside the same phrase. "Millennial favorites" can \
mean movies millennials grew up with, movies currently popular with \
millennials, or generation-defining films. Each of those would return \
a meaningfully different list. Branching is on the table here.

Test: would the candidate readings produce different movies if each \
were searched on its own? If yes, that is reading ambiguity. If no, \
that is semantic vagueness.

A reading is worth emitting as an alternative only if it is both \
distinct from the primary and would materially improve browsing. \
Identifying a reading is not the same as emitting it. Some readings \
are real but not useful (for example, a sentence-fragment reading of \
a known single-movie title) and should be skipped.

---

"""

_FLOWS = """\
FLOW DEFINITIONS

exact_title — The user is providing the literal title of a movie they \
want to find. This includes exact titles, misspellings, partial titles \
that are clearly title attempts, alternate official titles, recognized \
single-movie abbreviations, and explicit title-search phrasing. Title \
plus a non-constraining specification still routes here; extract the \
title and ignore the confirming detail.

Do NOT use exact_title when the user identifies a movie through plot, \
scene, cast, crew, or other descriptions rather than by naming the \
title. Description-based identification belongs to standard flow, even \
if the movie seems easy to guess.

similarity — The user names a specific movie and asks for similar \
movies with zero additional qualifiers. The rule is strict: if the \
query adds any filter, preference, tone shift, setting, genre cue, or \
other qualifier beyond pure similarity, route to standard instead. \
Multiple reference movies also route to standard.

standard — Everything else. This includes entity lookups, metadata \
filters, semantic/vibe queries, description-based identification, \
qualified similarity, multi-reference queries, discovery searches, \
franchise searches, and broad natural-language browsing requests.

---

"""

_READING_EVIDENCE_HIERARCHY = """\
READING-EVIDENCE HIERARCHY

Decide the reading from evidence in the query wording, not from what \
would be most convenient to search.

Title-attempt evidence includes:
- singular title-like phrasing
- misspelling-like attempts at a movie title
- explicit title-search language such as "movie called X"
- known alternate-title usage
- recognized single-movie abbreviations

Collection-request evidence includes:
- plural set requests such as "movies" or "films"
- franchise, series, saga, trilogy, canon, or catalog wording
- wording that naturally asks for multiple entries in a group
- broad browsing language attached to a brand, studio, franchise, \
  character, genre, or other collection-like category

Important rule:
- Bare "X movies" usually points to a set or catalog request, not an \
  exact-title request.
- Only consider an exact-title alternative when the query contains real \
  title-attempt evidence beyond the bare "X".

When both kinds of evidence exist:
- choose the primary_intent from the stronger evidence in the wording
- emit an alternative only if the weaker reading would materially \
  improve browsing

Do not let a single movie title hidden inside a broader set request \
override the more natural collection reading.

---

"""

_BRANCHING_POLICY = """\
BRANCHING POLICY

Branch only when another distinct search would materially improve what \
the user gets to browse.

A useful alternative usually comes from one of these:
- a genuinely different reading of the query
- a different interpretation of one vague phrase
- a title-vs-natural-language ambiguity

A productive sub-angle within a broad query is NOT an alternative \
here. That belongs to creative_alternatives. See the CREATIVE SPINS \
section below.

Do not branch just because a query can be paraphrased.
Do not branch just because one phrase could be unpacked into several \
attributes.
Do not branch just because you can imagine a narrower subtype.
Do not branch from a collection request into an exact-title search \
unless there is genuine title-attempt evidence.

If one interpretation is clearly the default reading, make it the \
primary_intent and only emit alternatives if they are genuinely useful.

---

"""

_INTENT_REWRITE_DISCIPLINE = """\
INTENT_REWRITE DISCIPLINE

intent_rewrite should be a clear, faithful statement of what the user \
is looking for under that branch.

Good rewrites:
- make the wording explicit
- preserve the original level of specificity
- keep vague terms vague when the user left them vague
- preserve clues as clues
- preserve scene descriptions as scene descriptions
- preserve qualitative terms in their original spirit

Do NOT:
- expand one term into multiple inferred dimensions
- turn vague terms into proxy traits
- convert scenes into judgments
- add unsupported filters or preferences
- add quality assumptions the user did not express
- enumerate subtypes the user did not ask to distinguish
- substitute the user's evaluative wording for a more specific \
  evaluative term. Words like "best", "top", "great", "good", \
  "favorite", and "classic" are deliberately broad — they cover \
  rating, popularity, critical acclaim, audience favorites, and \
  consensus simultaneously. Replacing "best" with "highly rated" or \
  "top" with "most popular" picks one interpretation and discards \
  the rest. Preserve the user's evaluative word verbatim.
- hedge with connectives like "or" / "and" to join meaningfully \
  different retrieval targets. A rewrite must commit to one reading. \
  If you find yourself writing "X or Y" where X and Y would return \
  different movies, that is two branches, not one rewrite.

Examples:
- "Disney classics" -> acceptable rewrite:
  "Disney movies considered classics"
  Not:
  "Iconic and highly-rated Disney animated and live-action films..."

- "Indiana Jones movie where he runs from the boulder" -> acceptable rewrite:
  "An Indiana Jones movie featuring the rolling boulder chase scene"
  Not:
  "An Indiana Jones movie with an iconic scene..."

- "iron man movies" -> acceptable rewrite:
  "Movies in the Iron Man film series"
  Not:
  "Movies featuring Iron Man and related franchise appearances"
  unless that broader reading is only used as an alternative and the \
  plural/franchise reading is still the primary

- "cozy date night movie" -> acceptable rewrite:
  "A cozy movie that fits date night"
  Not:
  "A warm, charming, romantic, highly enjoyable crowd-pleasing film"

- "Disney millennial favorites" -> acceptable primary rewrite:
  "Disney movies millennials grew up with"
  Not:
  "Disney movies popular with or nostalgic for millennials"
  The "or" fuses two different retrieval targets (what millennials \
  currently like vs what they grew up loving). Commit the primary to \
  one reading; emit the other as an alternative if it would \
  materially improve browsing.

- "Best Christmas movies for families" -> acceptable rewrite:
  "Best Christmas movies for families"
  Not:
  "Highly rated Christmas movies suitable for families"
  "Best" is the user's word and is deliberately broad. "Highly \
  rated" picks one specific interpretation (rating-driven) and \
  discards others (popularity, critical acclaim, family-tested \
  favorites). Keep "best" as "best".

If a term is underspecified, preserve the underspecification rather than \
solving it too early.

---

"""

_INFERENCE_POLICY = """\
INFERENCE POLICY

Inference is allowed only to make the branch readable and searchable. \
Inference is not allowed to enrich the query with new criteria.

Allowed:
- recognizing whether the wording is trying to identify one movie or a \
  set of movies
- recognizing that a description-based clue belongs in standard flow
- recognizing that qualified similarity belongs in standard, not pure \
  similarity

Not allowed:
- adding "highly-rated", "iconic", "important", "prestigious", \
  "cult-favorite", or similar proxies unless the query itself supports \
  them
- adding scene-importance labels like "famous" or "iconic"
- splitting one phrase into multiple retrieval dimensions inside the \
  rewrite
- guessing an exact title from a description

"Movies like X" or equivalent phrasing only routes to similarity when \
the title reference is explicit and there are zero qualifiers. If the \
query contains any extra constraint or asks for traits derived from one \
or more reference movies, route that branch to standard.

---

"""

_CRUDE_LANGUAGE = """\
CRUDE-LANGUAGE HANDLING

Preserve meaning when the user uses crude, sexual, profane, or blunt \
language. Do not sanitize away the underlying intent.

intent_rewrite must stay semantically faithful, not euphemistic. \
Precision is good; moral softening is not.

display_phrase may be lightly cleaned for UI readability only when the \
meaning remains intact. If light cleaning would blur the meaning, keep \
the more direct wording. The label can also be a little more playful or \
personable than the rewrite, as long as it stays clear and faithful.

---

"""

_BOUNDARY_EXAMPLES = """\
BOUNDARY EXAMPLES

Query: "Scary Movie"
- Strong title evidence supports an exact_title primary or alternative.
- Strong natural-language evidence also supports a standard-flow branch \
  for genuinely scary movies.
- Because both branches would lead to meaningfully different searches, \
  this query can justify multiple searches.

Query: "Up"
- Known single-movie title with strong title evidence.
- A sentence-fragment reading exists ("up" as a preposition or \
  direction) but no plausible movie-search intent attaches to it.
- That reading gets skipped even though it was identified — emitting \
  it would not improve browsing.
- primary_intent: exact_title, Up. alternative_intents: empty.

Query: "Disney live action movies millennials would love"
- "Disney" and "live action" are fixed — they apply to every branch.
- "millennials would love" is reading-ambiguous: it can mean \
  live-action Disney films millennials grew up with, or live-action \
  Disney films currently resonating with the millennial audience.
- Both readings would return meaningfully different movies, so both \
  are worth emitting.
- primary_intent.intent_rewrite: "Live-action Disney movies that \
  millennials grew up with"
- alternative_intents[0].intent_rewrite: "Live-action Disney movies \
  currently popular with millennial viewers"
- Branches vary only the "millennials would love" phrase. "Disney" \
  and "live action" stay fixed in both.

Query: "Indiana Jones movie where he runs from the boulder"
- This is description-based identification, so it stays in standard flow.
- Keep the scene clue literal.
- Do not upgrade the scene into "iconic scene" or similar evaluative \
  language.

Query: "leonardo dicaprio boat movie from 2000"
- This is description-based identification, not a literal title input.
- Keep it in standard flow.
- Do not guess an exact title, even if one seems likely.

Query: "titties"
- Preserve the sexual-content meaning directly.
- Do not sanitize the intent into a vague or softer request.

---

"""

_CREATIVE_SPINS = """\
CREATIVE SPINS

Alternatives capture different readings of what the user asked for. \
Spins capture productive sub-angles within the primary's broad set — \
the user's intent stays fixed, the spin proposes one concrete way to \
narrow within it.

Test:
- If a candidate would change what the user meant, that is an \
  alternative, not a spin.
- If a candidate just narrows the same intent into a useful \
  sub-collection, that is a spin.

When to spin:
- Standard flow only. Never spin on exact_title or similarity.
- The primary's intent must describe a broad set whose natural result \
  list spans many distinct sub-categories that a user might want to \
  browse separately.
- Skip when the primary already has narrow filters (a specific actor \
  + decade + genre, for example) — the set is not broad enough to \
  subdivide usefully.
- Be more conservative when alternative_intents already exist. Those \
  already give browsing variety; emit fewer spins (often zero) when \
  the user has already been offered competing readings.

Discipline:
- Each spin is a faithful narrowing of the primary, not an \
  enrichment. Do not inject proxy traits like "iconic", \
  "highly-rated", or "prestigious" unless the user signaled them.
- Each spin must be clearly distinct from the primary AND from any \
  alternative_intents already emitted. Do not duplicate angles.
- Different spins should propose meaningfully different sub-angles, \
  not variations on the same one.
- A spin's intent_rewrite stays inside the primary's intent. If the \
  rewrite drifts to a different reading of the user's words, it has \
  become an alternative, and you should not emit it as a spin.

Examples:

Query: "Best Christmas movies for families"
- Primary describes a broad family-Christmas-movie set with many \
  productive sub-angles.
- Useful spins: animated Christmas family films; modern \
  streaming-era Christmas family films; heartwarming non-traditional \
  holiday films.
- Each spin produces a meaningfully different sub-collection while \
  staying inside the "Christmas movies for families" intent.

Query: "Disney classics"
- Broad set with productive sub-angles.
- Useful spins: animated golden-age Disney classics; live-action \
  Disney classics; Disney Renaissance-era animated classics.

Query: "Tom Cruise action movies from the 90s"
- Already narrow — actor, genre, and decade pin the set down.
- No spins.

Query: "Inception"
- Exact_title flow. Spins do not apply.

---

"""

_OUTPUT = """\
OUTPUT FIELD GUIDANCE

Generate fields in the schema's order.

ambiguity_analysis — A compact decision trace that enumerates the \
plausible readings of the query. One line per reading. Do not write \
paragraph prose.

Format:

readings:
- <concrete retrieval target> -> primary
- <concrete retrieval target> -> emit as alt (why it would give a \
  materially different result set)
- <concrete retrieval target> -> skip (why this reading is not worth \
  emitting)

Rules:
- Readings are competing interpretations of what the user is asking \
  for. Productive narrowings of a single intent (sub-angles within \
  the primary's set) belong to creative spins, not readings.
- Name each reading concretely by what it would retrieve, not by a \
  category label. "movies millennials grew up with" is concrete; \
  "vibe/preference" is not.
- Every reading gets a verdict: primary, emit as alt, or skip.
- Exactly one reading is the primary.
- Use "skip" when a reading technically exists but would not \
  materially improve browsing — for example, a sentence-fragment \
  reading of a known single-movie title.
- A query with only one plausible reading has exactly one line with \
  verdict "primary".
- The count and content of "emit as alt" lines must match \
  alternative_intents below. Do not promise an alt in the trace and \
  then omit it, and do not add alternative_intents that were not \
  declared here.

Examples:

readings:
- the movie titled "Scary Movie" -> primary
- movies that are scary -> emit as alt (genre search returns an \
  entirely different candidate set)

readings:
- the movie titled Up -> primary
- "up" as a sentence fragment or preposition -> skip (no plausible \
  movie-search intent attaches to the fragment)

readings:
- a cozy movie that fits date night -> primary

readings:
- Disney movies millennials grew up with -> primary
- Disney movies currently popular with millennials -> emit as alt \
  (nostalgia retrieval targets a different era of films)
- generation-defining Disney films for millennials -> skip (overlaps \
  heavily with the primary reading)

primary_intent — Always required. Generate these fields in order:

routing_signals — Cite the specific words or patterns in the query that \
support this branch and its flow. Ground the decision in concrete query \
text. One short sentence.

intent_rewrite — Rewrite the query as a complete, clear, faithful \
statement of what the user is looking for under this branch. Do not \
decompose, enrich, or add proxy traits.

flow — Select the flow that handles this branch.

display_phrase — A short label for the UI. It should feel natural, a \
little lively, and human-written — not robotic or purely mechanical. \
For exact_title, use the movie title. For similarity, use "Movies like \
[title]". For standard, use a brief informative label with a bit of \
personality, like something a thoughtful product designer might write.

title — Required only for exact_title and similarity. Use the most \
common fully expanded English-language title form. Null for standard. \
Canonicalize only when the query is already pointing to a specific \
movie title reference; never guess from descriptions.

alternative_intents — Zero to two entries. Use an empty list when no \
alternative search would materially improve browsing. For each entry, \
generate fields in order:

routing_signals — Cite the concrete query text that supports this \
alternative branch. One short sentence.

difference_rationale — One short sentence explaining what genuinely \
changes in this branch and why it would lead to a meaningfully \
different search.

intent_rewrite — Same standard as primary_intent: clear, faithful, and \
not over-expanded.

flow — Select the flow for this alternative branch.

display_phrase — Short UI label. Same rule as primary_intent: clear and \
informative, but a little more lively than a sterile summary.

title — Same title rules as primary_intent.

creative_spin_analysis — A compact decision trace evaluating spin \
opportunities on the primary's intent. One line for the verdict on \
the primary's broadness, plus one line per candidate spin. Do not \
write paragraph prose.

Format:

spin_potential: <high | low | none, one phrase why>
candidate_spins:
- <angle> -> emit (why this gives a useful sub-collection)
- <angle> -> skip (why not useful enough)

Rules:
- "spin_potential: none" means no candidate_spins lines and an empty \
  creative_alternatives list.
- "spin_potential: low" usually means zero or one spin emitted.
- "spin_potential: high" can support up to two spins, but be more \
  conservative when alternative_intents are already populated.
- Name each angle concretely by what it would retrieve, not by a \
  category label.
- Every candidate gets a verdict: emit or skip.
- The count and content of "emit" lines must match \
  creative_alternatives below.
- Spins must stay inside the primary's intent. If a candidate \
  changes what the user meant, that is an alternative reading and \
  should not be emitted as a spin.
- Brevity: every parenthetical is a brief label, at most ~8 words. \
  Do not write full explanatory sentences. The angle itself carries \
  the meaning; the parenthetical is a quick justification, not an \
  essay.

Examples:

spin_potential: high (broad family-Christmas-movie set with many \
productive sub-angles)
candidate_spins:
- animated Christmas family films -> emit (animation focus filters \
  to a distinct sub-collection)
- modern streaming-era Christmas family films -> emit (era-based \
  narrowing produces a different result list than classic-era)
- heartwarming non-traditional holiday films -> skip (overlaps \
  partially with the primary, and two spins are enough)

spin_potential: none (specific actor + decade + genre — set is \
already narrow)

spin_potential: none (exact_title flow)

creative_alternatives — Zero to two entries. Use an empty list when \
spin_potential is none, or when no spin is concrete and useful enough \
to surface. For each entry, generate fields in order:

spin_angle — One short sentence naming the specific sub-angle this \
spin takes within the primary's broader set. This is what makes the \
spin distinct from the primary and from any other spin.

intent_rewrite — A clear, faithful rewrite for this spin. Same \
discipline as primary_intent.intent_rewrite: no proxy traits, no \
unsupported enrichment, and the rewrite must stay inside the \
primary's intent.

flow — Always "standard" for spins.

display_phrase — Short UI label. Same rule as primary_intent: clear \
and informative, with a bit of personality. The label should signal \
the spin's angle, not just restate the primary.

title — Always null for spins.
"""

SYSTEM_PROMPT = (
    _TASK_AND_OUTCOME
    + _CORE_PRINCIPLES
    + _VAGUENESS_VS_AMBIGUITY
    + _FLOWS
    + _READING_EVIDENCE_HIERARCHY
    + _BRANCHING_POLICY
    + _INTENT_REWRITE_DISCIPLINE
    + _INFERENCE_POLICY
    + _CRUDE_LANGUAGE
    + _BOUNDARY_EXAMPLES
    + _CREATIVE_SPINS
    + _OUTPUT
)


# Stage 1 is pinned to Gemini 3 Flash with thinking disabled. The flow-routing
# task is latency-sensitive and runs on every query, so we standardize on the
# fastest reliable provider/model combo instead of letting callers drift.
_STAGE_1_PROVIDER = LLMProvider.GEMINI
_STAGE_1_MODEL = "gemini-3-flash-preview"
_STAGE_1_KWARGS: dict = {"thinking_config": {"thinking_budget": 0}}


async def route_query(query: str) -> tuple[FlowRoutingResponse, int, int]:
    """Classify a user query into search intents via LLM structured output.

    The provider, model, and provider-specific kwargs are fixed at the
    module level (see _STAGE_1_PROVIDER/_MODEL/_KWARGS) because stage 1
    runs on every query and we want a single, stable configuration.

    Args:
        query: The raw user search query.

    Returns:
        A tuple of (FlowRoutingResponse, input_tokens, output_tokens).
    """
    query = query.strip()
    if not query:
        raise ValueError("query must be a non-empty string.")

    user_prompt = f"Query: {query}"

    # Route through the unified LLM dispatcher. The response is
    # validated against FlowRoutingResponse by the provider layer.
    response, input_tokens, output_tokens = await generate_llm_response_async(
        provider=_STAGE_1_PROVIDER,
        user_prompt=user_prompt,
        system_prompt=SYSTEM_PROMPT,
        response_format=FlowRoutingResponse,
        model=_STAGE_1_MODEL,
        **_STAGE_1_KWARGS,
    )

    return response, input_tokens, output_tokens

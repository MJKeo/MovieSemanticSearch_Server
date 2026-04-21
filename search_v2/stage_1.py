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
# Structure: task/outcome → core principles → flow definitions →
# ambiguity scaling → inference policy → crude-language handling →
# alternative quality bar → boundary examples → output field guidance.
#
# Prompt authoring conventions applied:
# - Evidence-inventory reasoning (cite concrete query text before
#   committing to branch shape or flow)
# - Brief pre-generation fields (ambiguity_analysis, ambiguity_level,
#   hard_constraints, ambiguity_sources)
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
when additional searches would meaningfully improve browsing under the \
query's ambiguity or open-endedness.

Use the query text as the primary evidence. You may use movie \
knowledge to recognize typed titles, alternate titles, or well-known \
single-movie abbreviations, but not to invent exact movie titles from \
descriptions. Do not guess a title unless the query itself is already \
pointing to a literal title reference.

---

"""

_CORE_PRINCIPLES = """\
CORE PRINCIPLES

Preserve hard constraints across every emitted search. Only vary the \
parts of the query that are genuinely ambiguous, underspecified, or \
open to useful exploration.

Hard constraints are clear traits that should survive every branch, \
such as a specific franchise, actor, decade, format, or genre when \
the query makes them explicit. Do not list soft interpretive ideas as \
hard constraints.

Additional searches must be materially different. Never emit near-\
duplicate branches, wording variants, or paraphrases that would lead \
to essentially the same downstream search.

The primary_intent must be the most likely or most useful main read in \
a movie-search context. Alternatives are optional helpers, not coequal \
defaults.

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

_AMBIGUITY_SCALING = """\
AMBIGUITY SCALING

Use ambiguity_level to summarize how much branching value the query has:

clear — One dominant reading. Usually emit only primary_intent. This is \
typical for exact-title queries, zero-qualifier similarity queries, and \
standard-flow queries whose intent is already concrete.

moderate — One main reading is strongest, but at least one useful \
alternative search would improve browsing. Emit primary_intent plus \
one or two alternatives only if they are meaningfully different.

high — The query has multiple strong readings, or it is vague enough \
that trying several distinct searches is clearly better than forcing \
one thin interpretation. Emit primary_intent plus the strongest useful \
alternatives, up to two.

When deciding how many searches to emit, prefer usefulness over \
theoretical possibility. The question is not "can I imagine another \
reading?" but "would another distinct search materially improve what \
the user gets to browse?"

Real alternate readings take priority over exploratory variation. If \
there is still room and the query is broad enough, an adjacent \
exploratory branch is allowed.

---

"""

_INFERENCE_POLICY = """\
INFERENCE POLICY

Inference is allowed when the query is vague, semantically \
underspecified, or uses a loose social/vibe concept that needs \
fleshing out into something searchable. In those cases, you may make \
logical interpretive leaps about what qualities the user could mean, \
as long as each emitted branch stays faithful to the query's fixed \
constraints.

Inference is NOT allowed to guess an exact movie title from a \
description. A description can stay in standard flow even if you \
strongly suspect a particular movie.

"Movies like X" or equivalent phrasing only routes to similarity when \
the title reference is explicit and there are zero qualifiers. If the \
query contains any extra constraint or asks for traits derived from one \
or more reference movies, route that branch to standard.

Do not infer hidden constraints that the query does not support. Make \
latent intent explicit, but do not enrich the request with unrelated \
preferences or quality assumptions.

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

_ALTERNATIVE_QUALITY = """\
ALTERNATIVE QUALITY BAR

Each alternative_intent must be meaningfully different from the \
primary_intent and from every other alternative. The difference should \
change what movies are likely to be retrieved, not just restate the \
same idea in slightly different words.

Good alternatives come from one of three sources:
- a genuinely different reading of the query
- a different fleshing-out of a vague or underspecified concept
- an adjacent exploratory variation that preserves hard constraints \
  while shifting softer interpretive dimensions

Do not rigidly assign roles like "intent 2 must do X" or "intent 3 \
must do Y." Let the query determine how many branches are useful and \
what kind they should be.

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

Query: "Disney live action movies millennials would love"
- "Disney" and "live action" are hard constraints that must stay fixed.
- The ambiguity lives in what "millennials would love" means.
- Useful alternatives can vary that clause while keeping the fixed \
  traits constant.

Query: "leonardo dicaprio boat movie from 2000"
- This is description-based identification, not a literal title input.
- Keep it in standard flow.
- Do not guess an exact title, even if one seems likely.

Query: "titties"
- Preserve the sexual-content meaning directly.
- Do not sanitize the intent into a vague or softer request.

---

"""

_OUTPUT = """\
OUTPUT FIELD GUIDANCE

Generate fields in the schema's order.

ambiguity_analysis — One concise sentence naming whether the query is \
clear, moderately ambiguous, or highly open to multiple useful \
searches, and why. This is an evidence inventory, not a justification \
essay.

ambiguity_level — Choose clear, moderate, or high based on the \
branching value defined above. This is a compact classification, not a \
confidence score.

hard_constraints — Short phrases for traits that must remain fixed \
across every emitted search. Use an empty list when no such shared \
constraints exist. Do not force entries just to populate the field.

ambiguity_sources — Short phrases naming the clause, concept, or part \
of the query that is open to interpretation. Use an empty list when the \
query has one dominant reading.

primary_intent — Always required. Generate these fields in order:

routing_signals — Cite the specific words or patterns in the query that \
support this branch and its flow. Ground the decision in concrete query \
text. One short sentence.

intent_rewrite — Rewrite the query as a complete, concrete statement of \
what the user is looking for under this branch. Surface implicit but \
strongly supported intent. For standard flow, this rewrite feeds \
directly into downstream query decomposition, so it must capture the \
full search intent without adding unsupported constraints.

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

difference_rationale — One short sentence explaining what ambiguous or \
exploratory dimension changes here, and why this would lead to a \
meaningfully different search.

intent_rewrite — Same standard as primary_intent: concrete, faithful, \
and usable downstream.

flow — Select the flow for this alternative branch.

display_phrase — Short UI label. Same rule as primary_intent: clear and \
informative, but a little more lively than a sterile summary.

title — Same title rules as primary_intent.
"""

SYSTEM_PROMPT = (
    _TASK_AND_OUTCOME
    + _CORE_PRINCIPLES
    + _FLOWS
    + _AMBIGUITY_SCALING
    + _INFERENCE_POLICY
    + _CRUDE_LANGUAGE
    + _ALTERNATIVE_QUALITY
    + _BOUNDARY_EXAMPLES
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

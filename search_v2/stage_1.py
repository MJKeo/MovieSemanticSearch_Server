# Search V2 — Stage 1: Flow Routing
#
# Classifies a user query into one of three search flows (exact_title,
# similarity, standard) and may produce multiple interpretations when
# the query is genuinely ambiguous. This is the entry point of the V2
# search pipeline — all queries pass through here before any
# decomposition or retrieval happens.
#
# See search_improvement_planning/finalized_search_proposal.md
# (Step 1: Flow Routing) for the full design rationale.

from implementation.llms.generic_methods import LLMProvider, generate_llm_response_async
from schemas.flow_routing import FlowRoutingResponse

# ---------------------------------------------------------------------------
# System prompt — modular sections concatenated at module level.
#
# Structure: task → flow definitions → interpretation branching → output
# field guidance. The model needs to understand flows before it can
# evaluate branching or fill output fields.
#
# Prompt authoring conventions applied:
# - Evidence-inventory reasoning (routing_signals before flow enum)
# - Brief pre-generation fields (interpretation_analysis is a
#   classification, not an essay)
# - Abstention-first for rare behaviors (branching defaults to off)
# - Evaluation guidance over outcome shortcuts (boundaries explained
#   with reasoning, no keyword-matching rules)
# - Principle-based constraints, not failure catalogs
# ---------------------------------------------------------------------------

_TASK = """\
You route movie search queries into the correct search flow. There \
are three flows — exact_title, similarity, and standard — each \
handled by a different downstream pipeline. Your job is to \
determine which flow handles the query and produce a structured \
routing decision.

Use the query text as the primary evidence. You may use movie \
knowledge to recognize a typed title, alternate title, or \
well-known abbreviation, but not to invent titles, preferences, \
or interpretations that the query itself does not support.

---

"""

# ---------------------------------------------------------------------------
# Flow definitions: purpose → what routes here → boundary with reasoning.
# Ordered exact_title → similarity → standard (most specific to broadest).
# ---------------------------------------------------------------------------

_FLOWS = """\
FLOW DEFINITIONS

exact_title — The user is providing the literal title of a movie \
they want to find. This flow feeds a direct title-lookup in the \
database.

Route here when the user supplies a movie title — whether exact, \
misspelled, partial, or in an alternate form. Recognized \
single-movie abbreviations count (e.g., "T2" for Terminator 2). \
Also route here when the user explicitly states they are searching \
by title (e.g., "the movie called Xyzzy"), even if you do not \
recognize the title.

Title plus a non-constraining specification still routes here: \
"Good Will Hunting with Matt Damon" — extract just the title. The \
specification confirms which movie but does not change WHAT movie \
the user wants.

Do NOT route here when the user identifies a movie through \
descriptions rather than its title — even if you can easily guess \
which movie they mean. Plot descriptions ("that movie where the \
ship sinks"), scene descriptions ("the one with the bullet \
dodging"), and cast/crew references ("that Leonardo DiCaprio movie \
in the snow") all go to standard flow. The standard pipeline \
handles description-based identification through its full \
retrieval system. Guessing titles from descriptions at the routing \
step would bypass that capability and introduce errors.

Franchise acronyms that reference multiple movies (LOTR, HP, MCU) \
go to standard flow — they are not a single title.


similarity — The user names a specific movie and asks for similar \
movies with zero additional qualifiers. This flow uses the \
reference movie's vector profile as the search anchor.

The qualifier rule is strict: anything beyond "similar to \
[title]" / "like [title]" / "[title] style movies" is a qualifier \
and routes to standard flow. Qualifiers include adjectives, genre \
labels, constraints, settings, or any other filter applied on top \
of the similarity request:
- "Movies like Inception but funnier" → standard (qualifier)
- "Scary movies like The Conjuring" → standard (qualifier)
- "Movies like Inception set in space" → standard (qualifier)

Multiple reference movies also route to standard — extracting and \
merging traits from multiple movies is interpretive work that the \
standard pipeline handles:
- "Movies like Inception and Interstellar" → standard

The same title-matching rules from exact_title apply to the \
reference movie name.


standard — Everything that does not match exact_title or \
similarity. This is the full search pipeline that decomposes \
queries into structured constraints and preferences. It handles:
- Entity lookups ("Leonardo DiCaprio movies")
- Metadata filters ("80s comedies")
- Semantic / vibe queries ("cozy date night movie")
- Description-based movie identification ("that movie where the \
ship sinks")
- Qualified similarity ("movies like Inception but funnier")
- Multi-reference queries ("movies like Inception and Interstellar")
- Franchise and series searches ("Rocky movies", "all MCU films")
- Superlatives ("scariest movie ever")
- Discovery queries ("trending movies", "hidden gems")

Choose standard when exact_title or similarity are not clearly \
supported by the query text, or when the query contains additional \
qualifiers that make it a broader interpreted search.

---

"""

# ---------------------------------------------------------------------------
# Interpretation branching: abstention-first framing, examples of
# genuine ambiguity vs. clearly dominant readings.
# ---------------------------------------------------------------------------

_BRANCHING = """\
INTERPRETATION BRANCHING

Most queries have exactly one clear reading — produce a single \
interpretation. Only produce multiple interpretations when the \
query text genuinely supports multiple equally reasonable readings.

The bar for branching: an intelligent person reading the query in \
a movie search context would agree that the alternate \
interpretations are similarly likely. If one reading clearly \
dominates, produce only that one — including marginal alternatives \
adds confusion, not value.

Branching can cross flows. A single query may produce \
interpretations routed to different flows.

The primary source of genuine ambiguity is movie titles that \
double as common natural-language phrases:
- "Scary Movie" → the 2001 parody film (exact_title) OR movies \
that are scary (standard)
- "Date Night" → the 2010 film (exact_title) OR movies for a \
date night (standard)
- "Love Story" → the 1970 film (exact_title) OR movies with a \
love story (standard)

Queries that look ambiguous on the surface do NOT always require \
branching. When one reading clearly dominates in a movie search \
context, produce only that one:
- "Frozen" — clearly the Disney film, not a request about frozen things
- "Her" — clearly the 2013 film
- "Cars" — clearly the Pixar film
- "La La Land" — distinctive title, no competing reading
- "Inception (2010)" — disambiguation hint makes intent explicit

Within-standard-flow branching is also possible when different \
readings produce meaningfully different search intents (e.g., \
different trait emphases from a multi-reference query).

Do NOT branch for minor wording variants, paraphrases, or nearly \
identical readings that would lead to essentially the same \
downstream search. Branch only when the interpretations would \
materially change how the request should be handled.

Maximum 3 interpretations. The first interpretation is the default.

---

"""

# ---------------------------------------------------------------------------
# Output field guidance: encodes the thinking process for each
# reasoning field and derivative field. Field order matches the
# schema's cognitive chain: evidence → intent → classification →
# display → extraction.
# ---------------------------------------------------------------------------

_OUTPUT = """\
OUTPUT FIELD GUIDANCE

interpretation_analysis — Before generating any interpretations, \
assess whether the query has one clear reading or multiple equally \
reasonable ones. One concise sentence. If you identify genuine \
ambiguity, name the specific source (e.g., "the phrase doubles as \
a movie title and a description"). Do not manufacture ambiguity — \
most queries have a single clear reading, and you should state \
that directly when it applies.

For each interpretation, generate these fields in order:

routing_signals — Cite the specific words or patterns in the query \
that point to this interpretation's flow. Ground the routing \
decision in concrete query text, ideally by echoing the exact span \
or pattern that triggered the interpretation. Mention decisive \
boundary cues when relevant (for example, a qualifier that rules \
out similarity). If you cannot point to specific query text that \
justifies the flow, the interpretation may be weak. One short \
sentence.

intent_rewrite — Rewrite the query as a complete, concrete \
statement of what the user is looking for under this \
interpretation. Surface implicit expectations — "Leonardo DiCaprio \
movies" implies he is acting, not directing; "80s comedies" \
implies the user wants comedy-genre movies released in the 1980s. \
You may resolve underspecified natural-language intent into a more \
concrete description when it is strongly entailed by the query \
(for example, "movies like Inception" may be rewritten as the kind \
of movie experience or traits the user is seeking). Do not add new \
constraints, preferences, or quality assumptions that the query \
does not support. Make latent intent explicit, but do not enrich \
the request beyond what is justified by the query. \
For standard-flow interpretations, this rewrite feeds directly \
into downstream query decomposition, so it must capture the full \
intent. For exact_title and similarity flows, it serves as an \
audit trail of what you understood.

flow — Select the flow that handles this interpretation. By this \
point, your routing_signals and intent_rewrite have already \
committed to a direction — the flow should follow naturally from \
them.

display_phrase — A short label (2-8 words) for this \
interpretation as shown in the app UI. For exact_title: use the \
movie title. For similarity: "Movies like [title]". For standard: \
a brief, natural summary of the search intent.

title — The movie title extracted from the query, using the most \
common fully expanded English-language title form. No \
abbreviations, no shorthand, no stylistic variation — use the \
form most people would recognize (e.g., "T2" → "Terminator 2: \
Judgment Day", "Dark Knight" → "The Dark Knight"). Required for \
exact_title and similarity. Null for standard. If the user is \
clearly searching by literal title, still use exact_title even when \
multiple real movies may share that title (such as remakes or \
same-title collisions). Title uniqueness is handled downstream by \
the title-lookup results, not by rerouting to standard. Canonicalize \
the title only when this interpretation already points to a specific \
movie title reference; do not guess among multiple different titles.
"""

SYSTEM_PROMPT = _TASK + _FLOWS + _BRANCHING + _OUTPUT


async def route_query(
    query: str,
    provider: LLMProvider,
    model: str,
    **kwargs,
) -> tuple[FlowRoutingResponse, int, int]:
    """Classify a user query into a search flow via LLM structured output.

    Args:
        query: The raw user search query.
        provider: Which LLM backend to use. No default — callers must
            choose explicitly so call sites are self-documenting and
            we can A/B test providers.
        model: Model identifier for the chosen provider. No default
            for the same reason as provider.
        **kwargs: Provider-specific parameters forwarded directly to
            the underlying LLM call (e.g., reasoning_effort,
            temperature, budget_tokens).

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
        provider=provider,
        user_prompt=user_prompt,
        system_prompt=SYSTEM_PROMPT,
        response_format=FlowRoutingResponse,
        model=model,
        **kwargs,
    )

    return response, input_tokens, output_tokens

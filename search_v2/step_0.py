# Search V2 — Step 0: Flow Routing
#
# Classifies a raw user query into one or more of the three major
# search flows (exact_title, similarity, standard) and carries the
# title payload each non-standard flow needs to execute. Also picks
# the most likely user intent via primary_flow so downstream assembly
# can order result lists.
#
# Step 0 is a narrow classifier. It does not rewrite the query,
# decompose intent, or propose alternate readings — those live in
# Step 1 and later. Step 0 and Step 1 run in parallel on the raw
# query; the merge happens in code afterward.
#
# See search_improvement_planning/steps_1_2_improving.md (Step 0:
# Flow Routing) for the full design rationale.

from implementation.llms.generic_methods import LLMProvider, generate_llm_response_async
from schemas.step_0_flow_routing import Step0Response

# ---------------------------------------------------------------------------
# System prompt — modular sections concatenated at module level.
#
# Structure: task/outcome → observations-first principle → per-field
# observation rules (titles with inline ambiguity reasoning,
# similarity frames, qualifiers) → flow eligibility rules →
# primary_flow selection → boundary examples → output field guidance.
#
# Authoring conventions applied:
# - FACTS → DECISION: observation fields come first. Ambiguity
#   reasoning lives inline on each title observation, not in a
#   separate structured bucket.
# - Principle-based rules, not failure catalogs.
# - Field order in the schema scaffolds reasoning for small models.
# ---------------------------------------------------------------------------

_TASK_AND_OUTCOME = """\
You route movie search queries into one or more of three major \
search flows — exact_title, similarity, and standard — and pick the \
single most likely user intent as the primary flow. Each flow is \
handled by a different downstream pipeline.

Emit any combination of flows for which the query carries evidence. \
A query can legitimately fire multiple flows when the phrasing \
supports more than one reading (for example, a bare phrase that is \
both a real movie title and a common descriptor with real \
standard-search value).

Your job is:
- observe titles (with inline reasoning about title ambiguity) and \
  qualifiers in the query
- decide which flows should fire and with what title payload
- pick the primary_flow — the single most likely user intent

Your job is NOT:
- to rewrite, paraphrase, or decompose the query
- to propose alternate interpretations beyond the ambiguity \
  reasoning on each title observation
- to enrich the query with proxy traits, quality signals, or \
  inferred narrowings
- to invent titles from descriptions — only recognize titles the \
  query is already pointing to

---

"""

_OBSERVATIONS_FIRST = """\
OBSERVATIONS FIRST, DECISIONS SECOND

The schema has two zones. Fill them in order.

Zone 1 — Observations (extractive). Quote what you see in the query \
and reason about ambiguity inline per title:
- titles_observed
- qualifiers

Zone 2 — Decisions. Once observations are recorded, derive the flow \
fields from them:
- exact_title_flow_data
- similarity_flow_data
- enable_primary_flow
- primary_flow

Never start by asking "which flow?" Start by asking "what do I see?" \
The flow fields are a function of the observations above them. If \
the observations are empty, only enable_primary_flow is set to true \
as the fallback.

---

"""

_TITLE_OBSERVATION_RULES = """\
TITLES_OBSERVED — how to populate

Each entry has three fields:
- span_text: the span exactly as it appears in the query, preserving \
  the user's wording and any typos.
- most_likely_canonical_title: the real movie title you judge the \
  span is most likely referring to.
- ambiguity_potential: brief reasoning about whether the span could \
  plausibly also represent a non-title reading (a genre word, mood \
  descriptor, common phrase, or other standard-search concept), or \
  whether you are uncertain that the user was actually searching for \
  this title. Keep it to one or two sentences.

Rules:
- List every real-movie-title span you recognize, regardless of \
  where it sits in the query. This includes titles inside similarity \
  frames ("movies like X" → list X here), titles inside descriptive \
  sentences ("I want to watch X tonight" → list X here), and \
  abbreviated or nickname-style references (e.g., "Gump" for Forrest \
  Gump, "LOTR" for The Lord of the Rings). The title extraction is \
  always visible in titles_observed — the decision about whether \
  each flow executes is separate.
- Typo tolerance: correct the typed form to a canonical title only \
  when the typed form is not itself a plausible English word or \
  common phrase. Misspellings of non-English-word titles are \
  correctable; English words that happen to coincide with a title \
  are not silently "corrected."
- When the query is a sentence that mentions a title, the title \
  span is just the title itself, not the whole sentence.

ambiguity_potential — how to reason:
- State whether the span could reasonably be read as something \
  other than this specific title. Be specific: name the alternative \
  reading if there is one (e.g., "could also be a generic mood \
  descriptor since the phrase is a common English phrase").
- Distinguish between "title is the only sensible reading" and \
  "title is one of multiple sensible readings." Being less than \
  100% certain a given canonical title is correct is NOT the same \
  as having a real alternate reading. A partial or nickname-style \
  reference with no plausible standard-search meaning is title-only \
  even when the canonical resolution required judgment.
- This reasoning will drive whether enable_primary_flow also fires \
  alongside exact_title. If you identify a real alternate reading \
  with standard-search value, enable_primary_flow should be true. If \
  the span is title-only or the "alternative" is just low confidence \
  that we have the right title, enable_primary_flow should NOT be \
  true from this observation alone.

---

"""

_SIMILARITY_FRAME_RULES = """\
SIMILARITY FRAMES — how to detect and resolve

A similarity frame is a phrase that asks for movies similar to a \
named reference. Typical shapes:
- "movies like X"
- "similar to X"
- "in the vein of X"
- "reminds me of X"
- "X but [modifier]" (e.g., "Inception but funnier")
- "something like X"

When you see a similarity frame:
1. Extract the referenced title span from inside the frame and add \
   it to titles_observed (ambiguity_potential usually "title-only" \
   unless the referenced phrase is itself ambiguous).
2. Resolve the reference to its canonical real-movie title.
3. Populate similarity_flow_data.similar_search_title with that \
   canonical title. This field is always populated with your best \
   candidate when one exists, even when the similarity search will \
   not execute.

Whether similarity executes is decided separately — see the next \
section. The extraction step above always runs when a similarity \
frame is present.

A similarity reference where the target is unresolvable (a pronoun \
with no antecedent, a phrase that does not name a real movie) \
produces no similarity payload — set similar_search_title to the \
empty string and should_be_searched to false.

---

"""

_QUALIFIER_RULES = """\
QUALIFIERS — how to populate

Qualifiers are the non-title, non-similarity-framing phrases that \
describe the kind of movie the user wants. Quote them directly from \
the query. Examples of things that count as qualifiers:
- genre words: "comedy", "horror", "thriller"
- mood/tone adjectives: "scary", "feel-good", "dark", "sad"
- year or era references: "from the 80s", "made in 2020", "recent"
- runtime references: "short", "under 90 minutes"
- streaming references: "on Netflix", "streaming free"
- rating references: "highly rated", "well-reviewed"
- entity names used as a filter: "Tom Cruise", "Pixar", "Spielberg"
- plot/concept phrases: "where things blow up", "about time travel"

Exclusions:
- Do not include similarity-framing phrases ("like", "similar to", \
  "in the vein of") — those are captured structurally by the \
  similarity-flow decision.
- Do not include phrases that are already fully captured by a \
  TitleObservation span. The ambiguity for those phrases is \
  handled inline on the title observation via ambiguity_potential; \
  duplicating them here would double-count.
- Do include modifier qualifiers that accompany a similarity frame \
  (e.g., "funnier" in "Inception but funnier"). These block the \
  similarity flow from executing — see the next section — but are \
  genuine standard-search signal in their own right.

---

"""

_FLOW_ELIGIBILITY_RULES = """\
FLOW ELIGIBILITY — how observations map to flow fields

exact_title_flow_data:
- exact_title_to_search: always populate with your best canonical \
  title candidate if the query references one. Empty string only \
  when the query names no title at all.
- should_be_searched: set to true when exactly one \
  TitleObservation's span_text covers the full query (ignoring case \
  and surrounding whitespace). The title's ambiguity_potential does \
  NOT suppress the search — even when the span is ambiguous, the \
  title reading is worth executing.
- Otherwise set should_be_searched to false (no title covers the \
  full query, multiple titles observed, or the title appears inside \
  a similarity frame / descriptive sentence).

similarity_flow_data:
- similar_search_title: always populate with your best canonical \
  reference title candidate when a similarity frame is observed. \
  Empty string only when no similarity reference exists or the \
  reference cannot be resolved.
- should_be_searched: set to true ONLY when all of the following \
  hold: a similarity frame is present, the reference resolves to a \
  real canonical title, AND qualifiers is empty. Similarity is an \
  "only mode" — the presence of any qualifier forces the query \
  into the standard flow where descriptive signal can be applied. \
  This is a hard rule enforced by a schema validator.

enable_primary_flow (boolean controlling the standard/default flow):
- Set to true if any of the following hold:
  * qualifiers is non-empty
  * titles_observed has more than one entry
  * any TitleObservation's span_text does NOT cover the full query \
    (i.e., the title sits inside a larger descriptive or similarity \
    frame, so the overall query is about more than just the title)
  * a full-coverage TitleObservation's ambiguity_potential \
    identifies a real alternate reading with standard-search value \
    (e.g., the span is also a common mood or descriptor). Think \
    critically here — just being less than 100% certain about the \
    canonical title is not a reason to fire standard. There must \
    be genuine non-title meaning the query could carry.
  * no other flow would fire (fallback so every query routes \
    somewhere)
- Otherwise set to false. In particular, a full-coverage title \
  whose ambiguity_potential says "title-only; no plausible \
  non-title reading" does not fire the standard flow.

At least one flow must always fire. An all-false output is invalid.

---

"""

_PRIMARY_FLOW_RULES = """\
PRIMARY_FLOW — picking the most likely user intent

primary_flow is a single enum value that names the flow whose \
results should lead the user's result list. It must name a flow \
that is actually firing.

Selection rules:

1. If only one flow fires, primary_flow is that flow.

2. If multiple flows fire, pick the reading you judge most likely \
   based on the ambiguity_potential reasoning of the observed \
   titles and the qualifier evidence:
   - If a full-coverage title's ambiguity reasoning leans toward \
     the literal title reading (even if the standard reading is \
     also plausible), primary_flow = exact_title.
   - If the ambiguity reasoning leans toward the standard reading \
     (the phrase is a common descriptor that happens to coincide \
     with a title), primary_flow = standard.
   - For similarity queries, primary_flow = similarity unless the \
     standard reading is much more likely.

3. Static tiebreaker when the evidence is genuinely balanced: \
   exact_title > similarity > standard (the more specific reading \
   wins).

---

"""

_BOUNDARY_EXAMPLES = """\
BOUNDARY EXAMPLES

The following worked examples illustrate the rules. For each, only \
the key fields are shown.

Example 1 — bare clean title
Query: "Interstellar"
- titles_observed: [{span_text: "Interstellar", most_likely_canonical_title: "Interstellar", ambiguity_potential: "title-only; 'Interstellar' is not a common word and has no plausible non-title reading."}]
- qualifiers: []
- exact_title_flow_data: {should_be_searched: true, exact_title_to_search: "Interstellar"}
- similarity_flow_data: {should_be_searched: false, similar_search_title: ""}
- enable_primary_flow: false
- primary_flow: exact_title

Example 2 — partial/nickname title reference
Query: "Gump"
- titles_observed: [{span_text: "Gump", most_likely_canonical_title: "Forrest Gump", ambiguity_potential: "title-only; 'Gump' is a surname with essentially no standard-search value as a descriptor."}]
- qualifiers: []
- exact_title_flow_data: {should_be_searched: true, exact_title_to_search: "Forrest Gump"}
- similarity_flow_data: {should_be_searched: false, similar_search_title: ""}
- enable_primary_flow: false  (ambiguity reasoning says no real alternate reading)
- primary_flow: exact_title

Example 3 — full-coverage title with genuine alternate reading
Query: "scary movie"
- titles_observed: [{span_text: "scary movie", most_likely_canonical_title: "Scary Movie", ambiguity_potential: "also a very plausible generic horror ask — 'scary' is a common mood word and 'movie' is generic. Real standard-search value."}]
- qualifiers: []
- exact_title_flow_data: {should_be_searched: true, exact_title_to_search: "Scary Movie"}
- similarity_flow_data: {should_be_searched: false, similar_search_title: ""}
- enable_primary_flow: true  (ambiguity reasoning identifies genuine non-title reading)
- primary_flow: exact_title  (slight lean to the literal title on a bare phrase, but standard is also emitted)

Example 4 — similarity-framed, no qualifiers
Query: "movies like Inception"
- titles_observed: [{span_text: "Inception", most_likely_canonical_title: "Inception", ambiguity_potential: "title-only."}]
- qualifiers: []
- exact_title_flow_data: {should_be_searched: false, exact_title_to_search: "Inception"}
- similarity_flow_data: {should_be_searched: true, similar_search_title: "Inception"}
- enable_primary_flow: false
- primary_flow: similarity

Example 5 — title in a sentence
Query: "I want to watch Inception tonight"
- titles_observed: [{span_text: "Inception", most_likely_canonical_title: "Inception", ambiguity_potential: "title-only; no plausible non-title reading for 'Inception'."}]
- qualifiers: ["tonight"]
- exact_title_flow_data: {should_be_searched: false, exact_title_to_search: "Inception"}
- similarity_flow_data: {should_be_searched: false, similar_search_title: ""}
- enable_primary_flow: true
- primary_flow: standard

Example 6 — multi-title list
Query: "Godfather and Goodfellas"
- titles_observed: [
    {span_text: "Godfather", most_likely_canonical_title: "The Godfather", ambiguity_potential: "title-only."},
    {span_text: "Goodfellas", most_likely_canonical_title: "Goodfellas", ambiguity_potential: "title-only."}
  ]
- qualifiers: []
- exact_title_flow_data: {should_be_searched: false, exact_title_to_search: "The Godfather"}
- similarity_flow_data: {should_be_searched: false, similar_search_title: ""}
- enable_primary_flow: true
- primary_flow: standard

Example 7 — pure descriptive
Query: "movies where things blow up"
- titles_observed: []
- qualifiers: ["movies where things blow up"]
- exact_title_flow_data: {should_be_searched: false, exact_title_to_search: ""}
- similarity_flow_data: {should_be_searched: false, similar_search_title: ""}
- enable_primary_flow: true
- primary_flow: standard

Example 8 — vague / fallback
Query: "surprise me"
- titles_observed: []
- qualifiers: []
- exact_title_flow_data: {should_be_searched: false, exact_title_to_search: ""}
- similarity_flow_data: {should_be_searched: false, similar_search_title: ""}
- enable_primary_flow: true  (fallback)
- primary_flow: standard

Example 9 — minor typo on a real title
Query: "Intersteller"
- titles_observed: [{span_text: "Intersteller", most_likely_canonical_title: "Interstellar", ambiguity_potential: "title-only; 'Intersteller' is not an English word, so the only sensible reading is a misspelled title."}]
- qualifiers: []
- exact_title_flow_data: {should_be_searched: true, exact_title_to_search: "Interstellar"}
- similarity_flow_data: {should_be_searched: false, similar_search_title: ""}
- enable_primary_flow: false
- primary_flow: exact_title

Example 10 — similarity frame blocked by a modifier qualifier
Query: "Inception but funnier"
- titles_observed: [{span_text: "Inception", most_likely_canonical_title: "Inception", ambiguity_potential: "title-only."}]
- qualifiers: ["funnier"]
- exact_title_flow_data: {should_be_searched: false, exact_title_to_search: "Inception"}
- similarity_flow_data: {should_be_searched: false, similar_search_title: "Inception"}  (reference captured but blocked by qualifier)
- enable_primary_flow: true
- primary_flow: standard

---

"""

_OUTPUT = """\
OUTPUT FIELD GUIDANCE

titles_observed — list of TitleObservation, possibly empty. Each \
entry requires span_text (quoted from the query), \
most_likely_canonical_title (the resolved real title), and \
ambiguity_potential (one or two sentences on whether the span could \
plausibly represent a non-title reading with real standard-search \
value). Every recognized title span goes here, including titles \
inside similarity frames and titles embedded in sentences.

qualifiers — list of strings, possibly empty. Quote each qualifier \
phrase directly from the query. Exclude similarity-framing phrases \
and phrases already fully captured by a TitleObservation span.

exact_title_flow_data — structured object with two fields:
- should_be_searched (bool): true only when exactly one \
  TitleObservation's span_text covers the full query.
- exact_title_to_search (str): the canonical title that would be \
  searched. Always your best candidate when a title is present; \
  empty string only when no title is named.

similarity_flow_data — structured object with two fields:
- should_be_searched (bool): true only when a similarity frame is \
  observed, the reference resolves, AND qualifiers is empty. \
  Presence of any qualifier blocks the similarity search.
- similar_search_title (str): the canonical reference title that \
  would be searched. Always your best candidate when a similarity \
  reference is present; empty string only when no similarity \
  reference exists.

enable_primary_flow — boolean. True when the standard (primary/\
default) search flow should run; false otherwise. At least one flow \
must fire overall.

primary_flow — one of "exact_title", "similarity", "standard". Must \
name a flow that is actually firing.
"""


SYSTEM_PROMPT = (
    _TASK_AND_OUTCOME
    + _OBSERVATIONS_FIRST
    + _TITLE_OBSERVATION_RULES
    + _SIMILARITY_FRAME_RULES
    + _QUALIFIER_RULES
    + _FLOW_ELIGIBILITY_RULES
    + _PRIMARY_FLOW_RULES
    + _BOUNDARY_EXAMPLES
    + _OUTPUT
)


# Step 0 is pinned to Gemini 3 Flash with thinking disabled and a low,
# slightly non-zero temperature. Flow routing runs on every query and
# is latency-sensitive, so we standardize on the fastest reliable
# backend. The small non-zero temperature leaves room for the ambiguity
# reasoning to be phrased differently on borderline cases without
# going off-rails.
_STEP_0_PROVIDER = LLMProvider.GEMINI
_STEP_0_MODEL = "gemini-3-flash-preview"
_STEP_0_KWARGS: dict = {
    "thinking_config": {"thinking_budget": 0},
    "temperature": 0.35,
}


async def run_step_0(query: str) -> tuple[Step0Response, int, int]:
    """Route a raw user query into search flows via LLM structured output.

    Provider, model, and provider-specific kwargs are fixed at the
    module level (see _STEP_0_PROVIDER/_MODEL/_KWARGS) because step 0
    runs on every query and we want a single stable configuration.

    Args:
        query: the raw user search query (non-empty after stripping).

    Returns:
        A tuple of (Step0Response, input_tokens, output_tokens).
    """
    query = query.strip()
    if not query:
        raise ValueError("query must be a non-empty string.")

    user_prompt = f"Query: {query}"

    response, input_tokens, output_tokens = await generate_llm_response_async(
        provider=_STEP_0_PROVIDER,
        user_prompt=user_prompt,
        system_prompt=SYSTEM_PROMPT,
        response_format=Step0Response,
        model=_STEP_0_MODEL,
        **_STEP_0_KWARGS,
    )

    return response, input_tokens, output_tokens

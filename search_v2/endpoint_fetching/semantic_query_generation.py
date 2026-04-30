# Search V2 — Stage 3 Semantic Endpoint: Query Translation
#
# Translates step-2 semantic items into concrete per-space queries
# against the 8 Qdrant vector spaces. Two public functions:
#
#   generate_semantic_dealbreaker_query  — one LLM call per semantic
#       dealbreaker (also used for semantic exclusions). Picks exactly
#       one of 7 non-anchor spaces and emits a structured-label body.
#   generate_semantic_preference_query   — one LLM call per grouped
#       semantic preference. Decomposes the grouped qualifier
#       description, picks 1+ of 8 spaces (anchor allowed), assigns
#       central/supporting weights, and emits a body per space.
#
# Both functions trust upstream routing and step-2 classification —
# they are schema translators, not re-interpreters. Inclusion vs.
# exclusion is handled by the orchestrator, so the LLM only ever
# searches for the positive presence of a trait.
#
# The per-space "Purpose / What's Embedded / Boundary / Example
# Queries" taxonomy is authored inline in this module as the single
# source of truth for step-3 space selection.
#
# TODO: Once `movie_ingestion/final_ingestion/vector_text.py`
# docstrings are rewritten in the canonical Purpose / What's Embedded
# / Boundary / Examples shape, replace these inline constants with
# `.__doc__` extraction from each `create_*_vector_text` function so
# the ingest-side vector_text module stays the single source of truth
# per the proposal's convention.
#
# See search_improvement_planning/finalized_search_proposal.md
# (Step 3 → Endpoint 6: Semantic) for the full design rationale.

from implementation.llms.generic_methods import (
    LLMProvider,
    generate_llm_response_async,
)
from schemas.semantic_translation import (
    SemanticDealbreakerSpec,
    SemanticPreferenceSpec,
)


# ===========================================================================
# Shared prompt sections — used by both dealbreaker and preference prompts.
# ===========================================================================
#
# Prompt authoring conventions applied across this module:
# - Evidence-inventory reasoning (signal_inventory and
#   qualifier_inventory cite input phrases, not arguments)
# - Brief pre-generation fields (target_fields_label and
#   carries_qualifiers are labels, not sentences)
# - Principle-based constraints, not failure catalogs
# - Evaluation guidance over outcome shortcuts (space taxonomy teaches
#   boundaries; no keyword-match rules like "if 'scary' → pick X")
# - Per-section guidance, no nonzero floors or cross-section targets
# - Example-eval separation (worked examples are illustrative only)
# - No schema internals leaked (e.g., no mention of discriminated
#   unions, Pydantic, validators); the model reasons about the shape
#   of the output via field-level guidance.
# ---------------------------------------------------------------------------


# Direction-agnostic framing: identical in shape to the entity and
# franchise prompts. Critical invariant — the description is always
# positive-presence, even when the underlying dealbreaker is an
# exclusion. The LLM must never try to "undo" the framing.
_DIRECTION_AGNOSTIC = """\
POSITIVE-PRESENCE INVARIANT

Every description you receive describes what to search FOR, never \
what to search AGAINST. If the user's original intent was to \
exclude a trait or de-emphasize a qualifier, the description has \
already been rewritten in positive-presence form and a separate \
execution layer handles the exclusion / penalty logic on the result \
set. You always author queries that match movies that HAVE the \
specified trait or qualifier. Do not invert, negate, or search for \
"everything except X". Your job is to describe the thing, in the \
target space's native vocabulary.

---

"""


# Body authoring principles — how to fill a space's structured-label
# body regardless of which space is chosen. The fundamental move is
# "write like the ingest side would write for a matching movie."
_BODY_AUTHORING_PRINCIPLES = """\
BODY AUTHORING PRINCIPLES

Each vector space's body has a fixed shape: some prose fields and \
some lists of short terms, grouped into the sub-fields named in \
that space's taxonomy entry below. When you author a body, you \
are writing the query-side text that will be embedded and compared \
to the ingest-side text for every candidate movie. Same vocabulary, \
same shape, same style.

- Populate only the sub-fields where the concept genuinely lands. \
  Empty lists are valid and expected. An empty list is a truthful \
  signal; filling a sub-field with weak or invented content \
  dilutes the query vector and hurts matching.
- Use the space's native vocabulary. Each space has its own \
  preferred phrasings — themes and conflict types for plot_analysis, \
  emotional-palette adjectives for viewer_experience, motivation \
  phrases for watch_context, craft terms for narrative_techniques, \
  location names and technique terms for production. The taxonomy \
  entry below tells you which sub-fields each space carries; write \
  in the register those sub-fields already use on the ingest side.
- Term lists hold compact phrases, not sentences. Two-to-four-word \
  items are typical. "unreliable narrator" — good. "The story is \
  told by an unreliable narrator who withholds information from the \
  audience." — too long; collapse to the key phrase.
- Prose fields (plot_summary, elevator_pitch, reception_summary, \
  identity_pitch, identity_overview, plot_overview) stay brief and \
  dense — one or two sentences that carry the signal, not a full \
  paragraph.
- Translate into the target space's native format, not the user's \
  raw wording. Some spaces want compact prose (especially \
  plot_events); others want short labeled phrases \
  (viewer_experience, watch_context, narrative_techniques, \
  production). Small expansions are good when they restate the same \
  supported signal in the space's native vocabulary. Do not add new \
  story facts, themes, motivations, or evaluative claims that the \
  input does not support.
- Expansion pressure varies by space. Viewer_experience often benefits \
  from a few nearby feeling/tone terms that sharpen the intended \
  mood. Plot_events should usually stay close to the concrete \
  situation described, phrased as compact prose. Plot_analysis can \
  translate into schema-native thematic/conflict language, but stay \
  tighter than viewer_experience — do not use it as an excuse to \
  infer a richer story than the input supports.
- Do not include numeric values — no ratings, years, runtimes, \
  budgets, or box-office figures. Numerics are handled by a \
  different endpoint and would not embed usefully.
- Do not restate the user's original phrasing verbatim when it does \
  not match the space's vocabulary. Translate it into the terms \
  the ingest side would use for a movie that matches.

---

"""


# ---------------------------------------------------------------------------
# Per-space taxonomy entries. Each follows the four-part shape:
# Purpose (one sentence), What's Embedded (the sub-field names
# actually present on the corresponding *Body class), Boundary (an
# explicit not-this line for misroute prevention), Example Queries
# (2-3 canonical queries that belong in this space).
#
# These constants are assembled into two taxonomy sections below:
# _DEALBREAKER_SPACE_TAXONOMY (7 non-anchor entries) and
# _PREFERENCE_SPACE_TAXONOMY (all 8 entries, anchor first).
# ---------------------------------------------------------------------------


_SPACE_ANCHOR = """\
anchor — Holistic movie fingerprint.
  Purpose: Captures the movie as a whole at a broad-vibe, \
identity-level granularity. Used for "movies like X" similarity and \
for preferences that don't emphasize any single dimension.
  What's embedded: identity_pitch (one-sentence capsule), \
identity_overview (short paragraph), genre_signatures (terms), \
themes (terms), emotional_palette (terms), key_draws (terms), \
reception_summary (short prose).
  Boundary: Anchor is a general fingerprint. If the query zeros in \
on a single specific dimension (a plot mechanism, a filming location, \
a narrative technique, a viewing occasion), that dimension's \
dedicated space carries it more sharply.
  Example queries: "something with a cozy Sunday-afternoon vibe"; \
"a sweeping romantic epic feel"; "movies like Amélie in spirit".\
"""


_SPACE_PLOT_EVENTS = """\
plot_events — What literally happens.
  Purpose: Captures the chronological plot — the actions, events, \
and character arc beats that unfold on screen. The "what happens" \
space.
  What's embedded: plot_summary (dense prose summary of the concrete \
events and characters).
  Boundary: Plot_events is literal and concrete. If the query is \
about themes, genre signatures, or conflict archetypes rather than \
specific events, that belongs in plot_analysis. If it is about how \
the story is told, that belongs in narrative_techniques.
  Example queries: "a heist that unravels when a team member \
betrays the crew"; "a lone survivor crosses a frozen continent to \
find their family"; "the one where strangers trapped in an elevator \
discover a murderer among them".\
"""


_SPACE_PLOT_ANALYSIS = """\
plot_analysis — What type of story thematically.
  Purpose: Captures the genre signatures, thematic concepts, \
conflict archetypes, and character-arc patterns that classify what \
kind of story this is. The "what type of story" space.
  What's embedded: elevator_pitch (one-sentence thematic capsule), \
plot_overview (short prose), genre_signatures (terms), conflict_type \
(terms describing the conflict archetype, e.g. "man vs self"), \
thematic_concepts (terms for abstract themes, e.g. "grief", \
"redemption", "moral compromise"), character_arcs (terms for arc \
patterns, e.g. "fall from grace", "coming of age").
  Boundary: Plot_analysis is about categorical and thematic shape. \
If the query asks about specific events ("the one where X \
happens"), that is plot_events. If it asks about the emotional or \
cognitive experience of watching, that is viewer_experience.
  Example queries: "stories about grief and reconciliation between \
estranged siblings"; "man-vs-nature survival dramas"; "tragic \
fall-from-grace character arcs".\
"""


_SPACE_VIEWER_EXPERIENCE = """\
viewer_experience — What it feels like to watch.
  Purpose: Captures the emotional, sensory, and cognitive \
experience of sitting through the movie. Tone, tension profile, \
disturbance level, cognitive load, ending aftertaste. The "what it \
feels like" space.
  What's embedded: emotional_palette, tension_adrenaline, \
tone_self_seriousness, cognitive_complexity, disturbance_profile, \
sensory_load, emotional_volatility, ending_aftertaste. Each of \
these is a pair of term lists — terms (what the experience IS) and \
negations (what the experience deliberately is NOT, when a \
boundary matters).
  Boundary: Viewer_experience is subjective and experiential. If \
the query is about story mechanics or themes, that is plot_analysis \
or plot_events. If it is about viewing occasion ("date night", \
"background movie"), that is watch_context. If it is about craft \
choices, that is narrative_techniques.
  Example queries: "something unsettling but not gory, slow-burn \
tension that stays with you"; "light, breezy, low-stakes, leaves you \
smiling"; "emotionally devastating but in a cathartic way".\
"""


_SPACE_WATCH_CONTEXT = """\
watch_context — Why and when to watch.
  Purpose: Captures viewing occasions, motivations, and situational \
pulls — the contexts in which this movie is the right choice. The \
"why and when" space.
  What's embedded: self_experience_motivations (what the viewer is \
seeking — "cheer me up", "challenge me"), external_motivations \
(external pulls — "for a first-date impression"), \
key_movie_feature_draws (specific draws — "great soundtrack", "short \
runtime"), watch_scenarios (occasions — "date night", "Sunday \
afternoon", "background movie").
  Boundary: Watch_context is about the viewing situation, not the \
movie's internal content. If the query is about how the movie feels \
moment-to-moment, that is viewer_experience. If it is about what \
the movie is about, that is plot_analysis.
  Example queries: "something to put on in the background while \
cooking"; "a good first-date movie that won't alienate anyone"; \
"comfort rewatch for a rainy Sunday".\
"""


_SPACE_NARRATIVE_TECHNIQUES = """\
narrative_techniques — How the story is told.
  Purpose: Captures storytelling craft — structure, point of view, \
information control, characterization methods, narrative devices. \
The "how it is told" space.
  What's embedded: narrative_archetype, narrative_delivery, \
pov_perspective, characterization_methods, character_arcs, \
audience_character_perception, information_control, \
conflict_stakes_design, additional_narrative_devices. Each sub-field \
holds short terms describing craft choices (e.g., "unreliable \
narrator", "non-linear timeline", "ensemble mosaic", "dramatic \
irony").
  Boundary: Narrative_techniques is about craft, not content. If \
the query is about what the story is about thematically, that is \
plot_analysis. If about what happens literally, that is \
plot_events. Techniques describe the delivery, not the story.
  Example queries: "told in reverse chronological order"; \
"found-footage mockumentary style"; "ensemble film where multiple \
storylines converge".\
"""


_SPACE_PRODUCTION = """\
production — How and where physically made.
  Purpose: Captures filming locations and production techniques — \
the craft and circumstances of making the film. The "how / where \
it was made" space.
  What's embedded: filming_locations (proper-noun place names — \
cities, regions, landscapes), production_techniques (craft terms — \
"practical effects", "shot on 16mm", "motion capture", \
"single-take long shot").
  Boundary: Production is about physical making, not storytelling \
or narrative craft. If the query is about how the story is told \
(structure, POV, pacing devices), that is narrative_techniques. If \
it is about genre or theme, that is plot_analysis.
  Example queries: "movies filmed in New Zealand landscapes"; \
"shot entirely on 16mm film"; "heavy practical creature effects, \
minimal CGI".\
"""


_SPACE_RECEPTION = """\
reception — What critics and audiences thought.
  Purpose: Captures critical and audience reception — the specific \
qualities the movie was praised or criticized for, and the overall \
reception shape. The "what people thought" space.
  What's embedded: reception_summary (short prose describing \
reception), praised_qualities (terms for specific praised aspects — \
"lead performance", "cinematography", "emotional resonance"), \
criticized_qualities (terms for specific criticized aspects — \
"pacing in the third act", "thin supporting characters").
  Boundary: Reception is about nuanced critical/audience response \
to specific qualities. Broad "critically acclaimed" or "fan \
favorite" framing without naming a specific praised or criticized \
quality is handled by a numeric score elsewhere — not this space. \
Use reception when the query names the aspect being praised or \
criticized.
  Example queries: "praised for its cinematography and production \
design"; "widely criticized for a rushed third act"; "cult \
reception despite a mixed initial critical reaction".\
"""


# Assembled taxonomy sections. Dealbreakers exclude anchor; preferences
# include all 8. Both use the same four-part entry shape so the model
# reasons about them identically.

_DEALBREAKER_SPACE_TAXONOMY = """\
VECTOR SPACES

You must pick exactly one of the following 7 non-anchor spaces. \
Each entry tells you what the space captures (Purpose), which \
sub-fields you will populate inside the chosen space's body (What's \
Embedded), what would be a misroute (Boundary), and 2-3 canonical \
example queries. The anchor space is intentionally not available \
for dealbreakers — it is too broad for a single pass/fail \
judgment on one concrete trait.

{PLOT_EVENTS}

{PLOT_ANALYSIS}

{VIEWER_EXPERIENCE}

{WATCH_CONTEXT}

{NARRATIVE_TECHNIQUES}

{PRODUCTION}

{RECEPTION}

---

""".format(
    PLOT_EVENTS=_SPACE_PLOT_EVENTS,
    PLOT_ANALYSIS=_SPACE_PLOT_ANALYSIS,
    VIEWER_EXPERIENCE=_SPACE_VIEWER_EXPERIENCE,
    WATCH_CONTEXT=_SPACE_WATCH_CONTEXT,
    NARRATIVE_TECHNIQUES=_SPACE_NARRATIVE_TECHNIQUES,
    PRODUCTION=_SPACE_PRODUCTION,
    RECEPTION=_SPACE_RECEPTION,
)


_PREFERENCE_SPACE_TAXONOMY = """\
VECTOR SPACES

You can pick one or more of the following 8 spaces. Each entry \
tells you what the space captures (Purpose), which sub-fields you \
will populate inside that space's body (What's Embedded), what \
would be a misroute (Boundary), and 2-3 canonical example queries. \
The anchor space is available here (unlike in the dealbreaker \
case) — it fits naturally for broad-vibe qualifiers that don't \
pin to a single specific dimension.

{ANCHOR}

{PLOT_EVENTS}

{PLOT_ANALYSIS}

{VIEWER_EXPERIENCE}

{WATCH_CONTEXT}

{NARRATIVE_TECHNIQUES}

{PRODUCTION}

{RECEPTION}

---

""".format(
    ANCHOR=_SPACE_ANCHOR,
    PLOT_EVENTS=_SPACE_PLOT_EVENTS,
    PLOT_ANALYSIS=_SPACE_PLOT_ANALYSIS,
    VIEWER_EXPERIENCE=_SPACE_VIEWER_EXPERIENCE,
    WATCH_CONTEXT=_SPACE_WATCH_CONTEXT,
    NARRATIVE_TECHNIQUES=_SPACE_NARRATIVE_TECHNIQUES,
    PRODUCTION=_SPACE_PRODUCTION,
    RECEPTION=_SPACE_RECEPTION,
)


# ===========================================================================
# Dealbreaker prompt — sections
# ===========================================================================


_DEALBREAKER_TASK = """\
You translate one concrete thematic trait into a single-space \
vector query. You receive a trait that has already been interpreted, \
routed, and framed as a positive-presence requirement. Your job is \
not to decide what the user meant or whether the trait belongs \
here — that is already done. Your job is to (1) pick which of the \
7 non-anchor vector spaces best captures this trait, (2) produce a \
structured-label body in that space's native vocabulary that a \
matching movie's ingest-side text would look like.

Inputs you receive:
- intent_rewrite — the full concrete statement of what the user is \
looking for. Use it only to disambiguate vague references in the \
description (e.g., which "war" the trait is about). Treat it as a \
hint for understanding the description, not as an additional source \
of traits to translate.
- description — the single trait you are translating, always \
written in positive-presence form ("centers on X", "involves Y \
themes", "features Z").
- routing_hint — a concept-type label explaining why this \
trait was routed to this endpoint. A hint, not authority.

Use description as the primary evidence for what to translate. \
intent_rewrite helps you understand what description is pointing at. \
routing_hint is background context only.

Trust the upstream routing. Semantic dealbreakers reach this \
endpoint only when no deterministic source (entity, metadata, \
awards, franchise_structure, keyword) can evaluate the trait. Do \
not refuse, do not swap endpoints, do not reinterpret. A space \
must always be picked, even when the trait's fit is imperfect — \
pick the single best option.

---

"""


_DEALBREAKER_REASONING = """\
REASONING FIELDS

Two reasoning fields are produced before the body. They are not \
decorative — each one grounds a specific downstream decision. \
Treat them with the same care as the body itself.

signal_inventory — An evidence inventory, not a justification. \
Cite concrete phrases from the description. Use intent_rewrite only \
where needed to disambiguate what a description phrase refers to. \
Never cite routing_hint here. For each phrase, name which candidate \
non-anchor space(s) it genuinely implicates. It is valid \
and expected to write "no phrase implicates production" (or any \
other space) when the trait does not cover that dimension — an \
empty-evidence line is a trace, not a signal to fabricate \
relevance. Keep it telegraphic: phrases → implicated spaces, no \
argumentation.

target_fields_label — A brief label, not a sentence. Two to six \
words. Names which sub-fields inside the about-to-be-chosen \
space's body will actually carry signal. Examples: \
"conflict_type, thematic_concepts"; "filming_locations"; \
"emotional_palette, tension_adrenaline"; "plot_summary". A label \
primes the body; a sentence templates and overconstrains it. If \
you find yourself writing a verb, stop and collapse to nouns.

---

"""


_DEALBREAKER_OUTPUT = """\
OUTPUT FIELD GUIDANCE

Generate fields in the schema's order. Each field is scaffolded by \
the reasoning field or sub-field directly before it — surface \
evidence first, commit to values after.

signal_inventory — Per the Reasoning Fields section above. \
Evidence inventory; phrases → implicated spaces; empty-evidence \
lines welcome.

target_fields_label — Per the Reasoning Fields section above. Brief \
label naming the sub-fields that will carry signal. 2-6 words.

body — A single-space body. The body commits to one space (the \
space field inside it) and then fills the content sub-fields for \
that space. Populate only the sub-fields named in your \
target_fields_label; leave others empty. Follow the Body Authoring \
Principles.

Two worked examples:

Example A — description: "centers on a heist plan that unravels \
mid-execution when one team member turns on the others".
  signal_inventory: "\\"heist plan that unravels\\" → plot_events \
(concrete event sequence); \\"one team member turns on the others\\" \
→ plot_events (character action); no phrase implicates \
plot_analysis theme, narrative_techniques craft, viewer_experience \
tone, watch_context occasion, production craft, or reception".
  target_fields_label: "plot_summary".
  body: space = plot_events, content.plot_summary = "A heist team \
executes a carefully planned job; during the execution, one member \
betrays the others and the plan falls apart."

Example B — description: "explores post-colonial identity and \
inherited cultural loss".
  signal_inventory: "\\"post-colonial identity\\" → plot_analysis \
(thematic concept); \\"inherited cultural loss\\" → plot_analysis \
(thematic concept); no phrase implicates plot_events, \
narrative_techniques, viewer_experience, watch_context, production, \
or reception".
  target_fields_label: "thematic_concepts, conflict_type".
  body: space = plot_analysis, content.thematic_concepts = \
["post-colonial identity", "inherited cultural loss"], \
content.conflict_type = ["cultural identity conflict"], \
remaining sub-fields empty.

Two principles visible in both examples: the inventory cites \
phrases and names one space; the label names the sub-fields that \
will be non-empty; the body populates only those sub-fields and \
uses the space's native vocabulary.
"""


DEALBREAKER_SYSTEM_PROMPT = (
    _DEALBREAKER_TASK
    + _DIRECTION_AGNOSTIC
    + _DEALBREAKER_SPACE_TAXONOMY
    + _BODY_AUTHORING_PRINCIPLES
    + _DEALBREAKER_REASONING
    + _DEALBREAKER_OUTPUT
)


# ===========================================================================
# Preference prompt — sections
# ===========================================================================


_PREFERENCE_TASK = """\
You translate one grouped preference description into per-space \
vector queries. You receive a description that consolidates \
multiple qualifiers about the desired viewing experience — tone, \
pacing, viewing occasion, thematic flavor, and so on — into one \
rich phrase. Your job is not to re-route or re-interpret — that is \
already done. Your job is to (1) decompose the grouped description \
into individual qualifiers, (2) pick the one or more of the 8 vector \
spaces that genuinely carry strong signal for those qualifiers, (3) \
assign a weight (central or supporting) to each selected space, and \
(4) author a \
structured-label body per selected space in that space's native \
vocabulary.

Inputs you receive:
- intent_rewrite — the full concrete statement of what the user is \
looking for. Context only — use it to disambiguate qualifiers \
whose referent is vague in the description. Do not mine it for extra \
qualifiers not already expressed by description.
- description — the grouped preference description, consolidating \
one or more qualifiers about the desired experience.
- routing_hint — a concept-type label. A hint, not authority.

Use description as the primary evidence for what to translate. \
intent_rewrite helps you understand what description is pointing at. \
routing_hint is background context only.

Trust the upstream grouping and routing. Your central failure mode \
to avoid is "blob handling" — treating the whole description as \
one undifferentiated concept and collapsing it onto a single space \
(usually anchor, plot_analysis, or viewer_experience). The goal is \
not to maximize the number of spaces. Multiple qualifiers can land \
in the same space. Choose the smallest set of spaces that each \
provide genuinely strong signal; do not add a space just because you \
can weakly justify it.

---

"""


_PREFERENCE_DECOMPOSITION = """\
DECOMPOSING THE GROUPED DESCRIPTION

Before picking spaces, split the description into atomic \
qualifiers. Each qualifier is a single dimension the user is \
expressing a preference about.

- Split on conjunctions and commas. "funny, dark, and thought-\
provoking with a cozy date night vibe" splits into "funny", "dark", \
"thought-provoking", "cozy", "date night vibe".
- A qualifier cluster that describes the same dimension with \
near-synonyms can stay as one unit — "dark and gritty" is one \
tone qualifier, not two.
- A single qualifier can implicate more than one space when the \
word naturally carries multiple dimensions. "slow-burn" is pacing \
(plot_events) and tension profile (viewer_experience). "epic" is \
anchor-level vibe and can also be watch_context (what to watch on \
a big screen).
- A qualifier that does not clearly map to any space is flagged \
in the inventory ("no clear space") and not force-routed. Better \
to leave a qualifier out of any space than to pollute a space \
with content that does not belong in its vocabulary.

The decomposition output is the qualifier_inventory field. Keep it \
telegraphic: one qualifier per line, each followed by the space(s) \
it implicates. No preamble.

---

"""


_PREFERENCE_WEIGHTS = """\
WEIGHT ASSIGNMENT

Each selected space receives one of two weights: central or \
supporting. There is no third tier. The distinction is about \
how much of the user's desired viewing experience the space \
carries, not how confident you are that the space is correct.

central — The space carries a major part of the intended match: a \
headline qualifier or multiple \
load-bearing qualifiers. The user's request would feel \
fundamentally different if this space were missing.

supporting — The space carries meaningful supporting signal that \
rounds out the experience but is not load-bearing on its own. The \
user's request would still be recognizable without it, though less \
complete.

A single request can have zero, one, or several central spaces. \
Central does NOT imply unique. It is normal for multiple spaces to \
be central when each carries a major part of the intended match. If \
nothing clearly stands out as load-bearing, it is acceptable to \
have all spaces at supporting — that is a truthful signal that \
the user's preference is broad-and-balanced rather than focused on \
one dimension. If a space would be below supporting (carries \
barely-there signal that adds little), do not select it at all. \
Spreading weight across many marginal spaces dilutes the query.

---

"""


_PREFERENCE_REASONING = """\
REASONING FIELDS

Two reasoning fields shape preference output: one at the top of \
the spec and one at the top of every per-space entry. Each one \
grounds a specific downstream decision.

qualifier_inventory (top-level) — An evidence inventory of the \
decomposition. List each atomic qualifier from the description; \
after each qualifier, name the space(s) it implicates. It is valid \
to write "no clear space" for a qualifier that does not map cleanly \
anywhere. Telegraphic format, no preamble, no justification prose — \
this is a decomposition trace, not an argument. Never cite \
routing_hint here.

carries_qualifiers (per-entry) — A brief label, not a sentence. \
Names which qualifiers from the top-level inventory land in this \
entry's space. Examples: "carries: dark, slow-burn"; "carries: \
date night, cozy"; "carries: funny and lighthearted — broad vibe". \
This single label primes three downstream decisions inside the \
entry: the space choice (why this entry exists), the weight \
(count and centrality of carried qualifiers), and the body \
(which concepts the body must express). A sentence would \
over-commit and make the body feel like a consistency check on \
the rationale; a label primes without templating.

---

"""


_PREFERENCE_OUTPUT = """\
OUTPUT FIELD GUIDANCE

Generate fields in the schema's order. The qualifier_inventory at \
the top grounds every space entry that follows; inside each entry, \
the carries_qualifiers label grounds the space, weight, and body.

qualifier_inventory — Per the Reasoning Fields section above. \
Per-qualifier decomposition trace. "no clear space" lines are \
welcome and truthful.

space_queries — One entry per selected space, no duplicates. Each \
entry has four fields, produced in order:

  carries_qualifiers — Brief label per the Reasoning Fields \
  section. Names the qualifiers from the inventory that land in \
  this space.

  space — The vector space this entry targets. Must not repeat a \
  space used by an earlier entry.

  weight — central or supporting, per the Weight Assignment \
  section above.

  content — A body in this space's native vocabulary. Populate \
  only the sub-fields where the carried qualifiers actually land; \
  follow the Body Authoring Principles.

Two worked examples:

Example A — description: "slow-burn, atmospheric, rainy-day \
melancholy".
  qualifier_inventory: "\\"slow-burn\\" → plot_events pacing + \
viewer_experience tension; \\"atmospheric\\" → viewer_experience \
sensory_load + anchor vibe; \\"rainy-day melancholy\\" → anchor \
vibe + watch_context scenario".
  Entry 1: carries_qualifiers = "carries: slow-burn pacing"; \
space = plot_events; weight = supporting; content.plot_summary = \
"Deliberate, unhurried pacing; events unfold gradually with \
extended quiet moments between beats."
  Entry 2: carries_qualifiers = "carries: slow-burn tension + \
atmospheric sensory load"; space = viewer_experience; weight = \
central; content.emotional_palette.terms = ["melancholy", \
"reflective", "subdued"]; content.tension_adrenaline.terms = \
["slow-burn", "simmering"]; content.sensory_load.terms = \
["atmospheric", "immersive", "muted"].
  Entry 3: carries_qualifiers = "carries: rainy-day melancholy + \
atmospheric broad vibe"; space = anchor; weight = central; \
content.identity_overview = "An atmospheric, melancholy piece \
that leans into quiet reflection."; content.emotional_palette = \
["melancholy", "reflective"]; content.themes = ["quiet reflection"].
  Entry 4: carries_qualifiers = "carries: rainy-day watch \
scenario"; space = watch_context; weight = supporting; \
content.watch_scenarios.terms = ["rainy afternoon", "quiet \
evening alone"]; content.self_experience_motivations.terms = \
["settle into a mood", "sit with an emotion"].

Example B — description: "something playful and colorful for \
family movie night".
  qualifier_inventory: "\\"playful\\" → viewer_experience tone + \
anchor vibe; \\"colorful\\" → anchor vibe + production craft; \
\\"family movie night\\" → watch_context scenario".
  Entry 1: carries_qualifiers = "carries: playful + colorful \
broad vibe"; space = anchor; weight = central; \
content.identity_overview = "A bright, playful family film with \
vivid color and upbeat energy."; content.emotional_palette = \
["playful", "joyful", "warm"]; content.themes = ["family", \
"wonder"].
  Entry 2: carries_qualifiers = "carries: playful tone"; space = \
viewer_experience; weight = supporting; \
content.tone_self_seriousness.terms = ["playful", "light-hearted", \
"unserious"]; content.emotional_palette.terms = ["upbeat", \
"joyful"].
  Entry 3: carries_qualifiers = "carries: family movie night \
scenario"; space = watch_context; weight = central; \
content.watch_scenarios.terms = ["family movie night", "watching \
with kids"]; content.external_motivations.terms = ["all-ages \
appropriate", "keeps everyone engaged"].

Visible in both examples: the inventory decomposes before any \
entry commits; carries_qualifiers is label-form; weights reflect \
how much of the request each space carries; bodies populate only \
the sub-fields the carried qualifiers actually land in.
"""


PREFERENCE_SYSTEM_PROMPT = (
    _PREFERENCE_TASK
    + _DIRECTION_AGNOSTIC
    + _PREFERENCE_SPACE_TAXONOMY
    + _BODY_AUTHORING_PRINCIPLES
    + _PREFERENCE_DECOMPOSITION
    + _PREFERENCE_WEIGHTS
    + _PREFERENCE_REASONING
    + _PREFERENCE_OUTPUT
)


# ===========================================================================
# Public async functions
# ===========================================================================


async def generate_semantic_dealbreaker_query(
    intent_rewrite: str,
    description: str,
    route_rationale: str,
    provider: LLMProvider,
    model: str,
    **kwargs,
) -> tuple[SemanticDealbreakerSpec, int, int]:
    """Translate one semantic dealbreaker (or semantic exclusion) into a SemanticDealbreakerSpec.

    The LLM picks exactly one of 7 non-anchor vector spaces and emits
    a structured-label body in that space's native vocabulary. Used
    for dealbreaker scenarios D1 (score-only) and D2 (candidate-
    generating), and also for semantic exclusions, which share the
    same scoring path.

    Args:
        intent_rewrite: The step-1 full concrete statement of what
            the user is looking for, used for disambiguation.
        description: The positive-presence statement of the semantic
            trait to translate (from a step-2 Dealbreaker).
        route_rationale: The concept-type label from step 2
            explaining why this item was routed to the semantic
            endpoint.
        provider: Which LLM backend to use. No default — callers
            must choose explicitly so call sites are
            self-documenting.
        model: Model identifier for the chosen provider. No default
            for the same reason as provider.
        **kwargs: Provider-specific parameters forwarded directly to
            the underlying LLM call (e.g., reasoning_effort,
            temperature, budget_tokens).

    Returns:
        A tuple of (SemanticDealbreakerSpec, input_tokens, output_tokens).
    """
    # TODO: When the stage-3 orchestrator introduces a shared request
    # model (one Pydantic class per endpoint call batching all step-2
    # item fields), move these strip + non-empty checks into that
    # model via `constr(strip_whitespace=True, min_length=1)` and
    # delete the manual validation here. Matches the identical TODO
    # in entity_query_generation.py.
    intent_rewrite = intent_rewrite.strip()
    description = description.strip()
    route_rationale = route_rationale.strip()
    if not intent_rewrite:
        raise ValueError("intent_rewrite must be a non-empty string.")
    if not description:
        raise ValueError("description must be a non-empty string.")
    if not route_rationale:
        raise ValueError("route_rationale must be a non-empty string.")

    # Labeled sections keep the three inputs distinct for the model.
    user_prompt = (
        f"intent_rewrite: {intent_rewrite}\n"
        f"description: {description}\n"
        f"routing_hint: {route_rationale}"
    )

    response, input_tokens, output_tokens = await generate_llm_response_async(
        provider=provider,
        user_prompt=user_prompt,
        system_prompt=DEALBREAKER_SYSTEM_PROMPT,
        response_format=SemanticDealbreakerSpec,
        model=model,
        **kwargs,
    )

    return response, input_tokens, output_tokens


async def generate_semantic_preference_query(
    intent_rewrite: str,
    description: str,
    route_rationale: str,
    provider: LLMProvider,
    model: str,
    **kwargs,
) -> tuple[SemanticPreferenceSpec, int, int]:
    """Translate one grouped semantic preference into a SemanticPreferenceSpec.

    The LLM decomposes the grouped qualifier description, picks 1+
    of the 8 vector spaces (anchor allowed), assigns central or
    supporting weights per selected space, and emits a
    structured-label body per selected space. Used for preference
    scenarios P1 (score-only) and P2 (candidate-generating).

    Args:
        intent_rewrite: The step-1 full concrete statement of what
            the user is looking for, used for disambiguation.
        description: The grouped qualifier description (from a
            step-2 Preference), consolidating multiple qualifiers
            about the desired viewing experience.
        route_rationale: The concept-type label from step 2.
        provider: Which LLM backend to use. No default.
        model: Model identifier for the chosen provider. No default.
        **kwargs: Provider-specific parameters forwarded to the LLM
            call.

    Returns:
        A tuple of (SemanticPreferenceSpec, input_tokens, output_tokens).
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
        f"routing_hint: {route_rationale}"
    )

    response, input_tokens, output_tokens = await generate_llm_response_async(
        provider=provider,
        user_prompt=user_prompt,
        system_prompt=PREFERENCE_SYSTEM_PROMPT,
        response_format=SemanticPreferenceSpec,
        model=model,
        **kwargs,
    )

    return response, input_tokens, output_tokens

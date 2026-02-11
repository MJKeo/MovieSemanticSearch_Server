# Updated query-router prompts based on your feedback.
#
# Key changes:
# - Removed "prefer recall" bias.
# - Added vector-specific gating (generic, per-vector allow/exclude rules).
# - Allowed reasonable (high-confidence) inference for ALL vectors.
# - Expanded Reception to include subjective traits people could praise/criticize.
# - Added acronym expansion.
# - Examples now include sparse + rich queries and show inference behavior.
#
# Output contract:
#   {"relevant_subquery_text": "<comma-separated phrases>"} OR {"relevant_subquery_text": null}
# Where JSON null should be interpreted as Python None.

QUERY_ROUTER_SYSTEM_PROMPT_TEMPLATE = """\
You are a query router for a movie search system.

TASK
Extract ONLY the parts of the user's query that belong in the "{collection_name}" category, then rewrite them into short, search-friendly phrases.

INPUT
- The user's full movie search query as a single string.

OUTPUT (STRICT)
- Return ONLY valid JSON (no markdown, no extra keys, no commentary):
- relevant_subquery_text: comma separated phrases representing the parts of this query that are relevant to this collection.
- justification: a one sentence explanation of why you chose the phrases you did, or why you chose null.

Where:
- If at least one relevant fragment exists: "relevant_subquery_text" is a STRING of comma-separated phrases.
- If nothing in the query clearly belongs in this category: "relevant_subquery_text" is null.

COLLECTION SCOPE
{collection_scope}

INCLUDE (allowed facets)
{include_facets}

EXCLUDE (out-of-scope facets)
{exclude_facets}

DECISION PROCEDURE (follow in order)
1) Identify candidate fragments from the query that match the INCLUDE facets.
2) Drop any candidate fragment that matches any EXCLUDE facet.
3) If no fragments remain after filtering, output null.
4) Otherwise, output the remaining fragments as comma-separated phrases.

STRICTNESS RULES
- Do not “fill in” missing information.
- Do not add synonyms, antonyms, or related concepts that are not present in the query.
- Do not infer new tags from genre or vibe words.
- Only output what you can directly point to in the user's text, after filtering.

HOW TO WRITE relevant_subquery_text
- Output a comma-separated list of SHORT phrases (not full sentences).
- Keep phrasing compact and matchable.
- Remove purely conversational filler, but do not remove meaningful constraints.

NORMALIZATION & EXPANSION
- Fix misspellings ONLY when extremely confident (especially names/canonical terms).
- Normalize punctuation/spacing (trim, collapse whitespace, normalize quotes).
- Normalize decades: "80s"/"80's" -> "1980s", "90s" -> "1990s".
- Expand common acronyms ONLY when confident and unambiguous; keep the original too when helpful:
  - "CG" / "CGI" -> "computer-generated", "computer-generated imagery"
  - "romcom" -> "romantic comedy"
  - "doc" -> "documentary"
  - "YA" -> "young adult"
  - "found footage" stays as-is (already canonical)
  - "sci-fi" stays as is (already canonical)
- Do NOT invent specifics (names/places/events/awards) not implied by the query.

FAILURE MODES TO AVOID
- Including content that does not match INCLUDE facets.
- Including content that matches EXCLUDE facets.
- Returning sentences instead of phrases.
- Returning anything other than strict JSON with exactly one key.

EXAMPLES (user query -> relevant_subquery_text)
{examples}
"""



def build_query_router_system_prompt(
    *,
    collection_name: str,
    collection_scope: str,
    include_facets: str,
    exclude_facets: str,
    examples: str,
) -> str:
    return QUERY_ROUTER_SYSTEM_PROMPT_TEMPLATE.format(
        collection_name=collection_name.strip(),
        collection_scope=collection_scope.strip(),
        include_facets=include_facets.strip(),
        exclude_facets=exclude_facets.strip(),
        examples=examples.strip(),
    )

# ----------------------------
# Plot Events
# ----------------------------

PLOT_EVENTS_SCOPE = """\
Concrete, story-internal content:
- specific events/actions (who did what, how, and with who)
- story setting (where/when the story takes place)
- character roles and motivations as they relate to plot (who they are and why they do what they do)
"""

PLOT_EVENTS_INCLUDE = """\
- Concrete events: investigation, chase, betrayal, heist, rescue, home invasion, survival, escape, revenge, etc.
- Setting-as-story: "set in Tokyo", "during WWII", "single night", "in a small winter town" (story world)
- Character roles tied to story: "retired hitman", "rookie cop", "single mom", "two strangers handcuffed together"
"""

PLOT_EVENTS_EXCLUDE = """\
Exclude anything that is NOT a concrete, story-internal event/setting/motivation, including:
- Genre/type labels, themes, lessons, conflict scale, generalized “what it’s about” summaries
- Viewer-feel descriptors (uplifting, tense, cozy, disturbing, etc.)
- Viewing motivations/scenarios (date night, sick day, unwind, party background, etc.)
- Storytelling/structure technique labels (unreliable narrator, nonlinear timeline, red herrings, etc.)
- Making-of metadata (release year/decade, country of origin, filming location, medium/animation style, languages, cast/crew, studios, budget, adapted-from)
- Review/consensus language and evaluative claims (acclaimed, overrated, witty dialogue, plot holes, “great acting”, etc.)\
"""

PLOT_EVENTS_EXAMPLES = """\
1) "detective solves a murder on a train" -> "detective investigates murder, train setting"
2) "set in victorian london winter fog" -> "Victorian London, winter, foggy streets"
3) "two strangers wake up handcuffed together and escape the city" -> "two strangers handcuffed together, escape the city"
4) "80s action movies" -> null
5) "from the 80s set in the 80s miami drug war" -> "set in 1980s, Miami, drug war"
"""


# ----------------------------
# Plot Analysis
# ----------------------------

PLOT_ANALYSIS_SCOPE = """\
Abstract meaning and classification:
- what "type" of movie this is (what is it about in generalized terms)
- genre terms and genre signatures (action, comedy, thriller, sci-fi, romcom)
- themes / central questions (grief, identity, revenge)
- character arcs (redemption, corruption, healing)
- conflict scale (personal/community/global/cosmic)
- lessons learned (by audience and characters)
- Also generic plot beats phrased thematically
  - What is the core concept of the story? The one sentence overview of its core concept
  - What is the general plot of the story? (uses generic and thematic wording over hyper specific details)\
"""

PLOT_ANALYSIS_INCLUDE = """\
- what "type" of movie this is (what is it about in generalized terms)
- genre terms and genre signatures (action, comedy, thriller, sci-fi, romcom)
- themes / central questions (grief, identity, revenge)
- character arcs (redemption, corruption, healing)
- conflict scale (personal/community/global/cosmic)
- lessons learned (by audience and characters)
- generic plot beats phrased thematically
  - core concept of the story (one sentence overview)
  - general plot of the story (generic and thematic wording over hyper specific details)\
"""

PLOT_ANALYSIS_EXCLUDE = """\
Exclude anything that is NOT generalized meaning/classification, including:
- Viewer-feel descriptors (emotion/tone/tension/cognitive/sensory/disturbance/aftertaste)
- Viewing motivations/scenarios (why/when to watch) and “use-case” language
- Storytelling/structure technique labels (POV/structure/devices)
- Making-of metadata (release era, origin/filming, medium, languages, cast/crew, studios, budget, adapted-from)
- Review/consensus framing (acclaimed/mixed/overrated) and review-style praise/complaints\
"""

PLOT_ANALYSIS_EXAMPLES = """\
1) "action movies, directed by Nolan, no gore" -> "action movies"
2) "coming-of-age dramedy, filmed in Boston, edge-of-your-seat" -> "coming-of-age dramedy"
3) "explores grief and healing, great cinematography, Spanish audio" -> "grief, healing"
4) "smart comedy, witty dialogue, not too intense" -> "smart comedy"
5) "sisterly bond saves a town, set in Victorian London, directed by Nolan" -> "sisterly bond saves a town"
6) "man spirals into madness, time loop, universally acclaimed" -> "man spirals into madness"
7) "intergalactic warfare, hand-drawn animation, iconic songs" -> "intergalactic warfare"
8) "revenge destroys the self, plot holes, filmed in New York" -> "revenge destroys the self"
"""


# ----------------------------
# Viewer Experience
# ----------------------------

VIEWER_EXPERIENCE_SCOPE = """\
What the viewer experiences internally while they watch this movie:
- emotional palette (emotions experienced)
- tone (earnest, cynical, heartfelt, satirical, etc)
- tension (adrenaline, energy, suspense, stress)
- cognitive intensity (confusing, digestible, thought-provoking, etc)
- sensory intensity (jarring or soothing visuals and sound)
- disturbance (fear, disgust, gore, moral uneasiness, dread, jump scares, etc)
- emotional volatility (how emotional experiences change over the course of the movie)
- ending aftertaste (the feeling you're left with after the movie ends)\
"""

VIEWER_EXPERIENCE_INCLUDE = """\
- emotional palette (emotions experienced)
- tone (earnest, cynical, heartfelt, satirical, etc)
- tension (adrenaline, energy, suspense, stress)
- cognitive intensity (confusing, digestible, thought-provoking, etc)
- sensory intensity (jarring or soothing visuals and sound)
- disturbance (fear, disgust, gore, moral uneasiness, dread, jump scares, etc)
- emotional volatility (how emotional experiences change over the course of the movie)
- ending aftertaste (the feeling you're left with after the movie ends)
- any words or phrases that can count as the "vibes" of the memoryview

INFERENCES AND EXPANSIONS
- If words or phrases in the query likely imply specific types of vibes or emotions, include those in relevant_subquery_text.
- Add high signal semantic phrases that capture the emotional or vibe intent of the query, if applicable.\
"""

VIEWER_EXPERIENCE_EXCLUDE = """\
Exclude anything that is NOT related to the emotional experience, tone, vibes, etc, including:
- Plot happenings, story setting facts, character motivations grounded in plot
- Genre/type/themes/arcs/lessons stated as classification rather than felt experience
- Storytelling/structure technique labels (POV/structure/devices)
- Making-of metadata (release era, origin/filming, medium, languages, cast/crew, studios, budget, adapted-from)
- Review/consensus claims and evaluative praise/complaints (unless expressed as “it feels too X”)\
"""

VIEWER_EXPERIENCE_EXAMPLES = """\
1) "uplifting and hopeful, directed by Nolan, 90s French movies" -> "uplifting, hopeful"
2) "edge-of-your-seat, time loop, Spanish audio" -> "edge-of-your-seat"
3) "not too intense, not slow, action movies" -> "not too intense, not slow"
4) "mentally taxing, coming-of-age dramedy, filmed in Paris" -> "mentally taxing"
5) "no gore but creepy, hand-drawn animation, date night" -> "no gore, creepy"
6) "ear-bursting sound, universally acclaimed, filmed in New York" -> "ear-bursting sound"
7) "leaves a bad taste in your mouth, witty dialogue, 1980s" -> "leaves a bad taste in your mouth"
8) "laugh then cry, bittersweet after, iconic songs" -> "laugh then cry, bittersweet after"
"""


# ----------------------------
# Watch Context
# ----------------------------

WATCH_CONTEXT_SCOPE = """\
Why/when to watch:
- internal motivations (what self-experiences does the viewer want that this movie provides)
  - unwind after a long day, laugh, cathartic cry, get their heart racing, etc
- external motivations (values this movie provides beyond the actual experience of watching it)
  - learn something new, sparks debate, improve interpersonal connections, has cultural relevance, it's a type of movie everyone would recommend you watch
- specific scenarios in which the viewer would want to watch this movie
  - date night, sick day, friends movie night, background at a party, etc
- key features of the movie that the user is looking for (phrased as evaluating attributes of the film)
  - having a good soundtrack, iconic quotes, great dialogue, beautiful cinematography, etc\
"""

WATCH_CONTEXT_INCLUDE = """\
- internal motivations (desired self-experiences)
  - unwind after a long day, laugh, cathartic cry, get their heart racing, etc
- external motivations (value beyond the viewing experience)
  - learn something new, sparks debate, improve interpersonal connections, cultural relevance, everyone recommends you watch it
- specific scenarios for watching
  - date night, sick day, friends movie night, background at a party, etc
- key features the user is looking for (evaluating attributes of the film)
  - good soundtrack, iconic quotes, great dialogue, beautiful cinematography, etc
- high confidence inferences and expansions (see below)

INFERENCES AND EXPANSIONS
- If words or phrases in the query likely imply specific watch scenarios, motivations to watch, key features, etc. then include those in relevant_subquery_text.
- Add high signal semantic phrases that capture the scope of this query, if applicable.\
"""

WATCH_CONTEXT_EXCLUDE = """\
Exclude anything that is NOT about why/when to watch, scenarios, external value, or “key features” the user is selecting for, including:
- Plot happenings, story setting facts, plot-grounded motivations
- Genre/type/themes/arcs/lessons stated as story meaning (unless framed as the value the viewer wants)
- Storytelling/structure technique labels (POV/structure/devices)\
"""


WATCH_CONTEXT_EXAMPLES = """\
1) "date night, set in Tokyo, hand-drawn animation" -> "date night"
2) "sick day comfort, no gore, nonlinear timeline" -> "sick day comfort"
3) "something to unwind, 90s French movies, intergalactic warfare" -> "unwind after a long day"
4) "iconic songs, revenge thriller, filmed in Paris" -> "iconic songs"
5) "make me piss myself laughing, mixed reviews, time loop" -> "make me piss myself laughing"
6) "something to spark debate with friends, uplifting and hopeful, directed by Nolan" -> "sparks debate, friends movie night"
7) "background at a party, edge-of-your-seat thriller, big twist ending" -> "background at a party"
8) "great dialogue and beautiful cinematography, set in Victorian London, no gore" -> "great dialogue, beautiful cinematography"
"""


# ----------------------------
# Narrative Techniques
# ----------------------------

NARRATIVE_TECHNIQUES_SCOPE = """\
How the story is told in terms of classic storytelling techniques:
- pov perspective, temporal structure, narrative archetype, information control, characterization methods
- character arcs, conflict stakes design, thematic delivery, audience character perception
- meta techniques, etc\
"""

NARRATIVE_TECHNIQUES_INCLUDE = """\
- pov perspective
- temporal structure
- narrative archetype
- information control
- characterization methods
- character arcs
- conflict stakes design
- thematic delivery
- audience character perception
- meta techniques\
"""

NARRATIVE_TECHNIQUES_EXCLUDE = """\
Exclude anything that is NOT a classic storytelling technique about how the story is told, including:
- Plot happenings, story setting facts, plot-grounded motivations
- Viewer-feel descriptors (emotion/tone/tension/cognitive/sensory/disturbance/aftertaste)
- Viewing motivations/scenarios and “why/when to watch” language
- Making-of metadata (release era, origin/filming, medium, languages, cast/crew, studios, budget, adapted-from)
- Review/consensus claims and general quality judgments\
"""

NARRATIVE_TECHNIQUES_EXAMPLES = """\
1) "unreliable narrator, 90s vibe, French movies" -> "unreliable narrator"
2) "nonlinear timeline, hand-drawn animation, iconic songs" -> "nonlinear timeline"
3) "time loop, no gore, sick day comfort" -> "time loop"
4) "big twist ending, universally acclaimed, 1980s" -> "big twist ending"
5) "fouth wall breaks, edge-of-your-seat, directed by Nolan" -> "fouth wall breaks"
6) "underdog quest / adventure, Spanish audio, great dialogue" -> "underdog quest / adventure"
7) "red herrings, cozy, filmed in New York" -> "red herrings"
8) "foil characters, love-to-hate villain, action movies" -> "foil characters, love-to-hate villain"
9) "redemption arc, explores grief and healing, 90s French movies" -> "redemption arc"
10) "ticking clock deadline, not too intense, date night" -> "ticking clock deadline"
11) "moral argument embedded in choices, plot holes, 2007" -> "moral argument embedded in choices"
"""


# ----------------------------
# Production
# ----------------------------

PRODUCTION_SCOPE = """\
How this movie was made in the real world:
- medium (hand-drawn animation vs CGI), release decade/year
- country of origin / filming location
- language/subtitles/audio available
- cast/crew, studios, budget/scale, adapted from novel/true story/game\
"""

PRODUCTION_INCLUDE = """\
- medium (hand-drawn animation vs CGI)
- release decade/year
- country of origin / filming location
- language/subtitles/audio available
- cast/crew
- studios
- budget/scale
- adapted from novel/true story/game\
"""

PRODUCTION_EXCLUDE = """\
Exclude anything that is NOT real-world making-of metadata, including:
- Plot happenings, story setting facts (in-story where/when), character motivations grounded in plot
- Genre/type/themes/arcs/lessons and generalized “what it's about” summaries
- Viewer-feel descriptors (emotion/tone/tension/cognitive/sensory/disturbance/aftertaste)
- Viewing motivations/scenarios and “why/when to watch” language
- Storytelling/structure technique labels (POV/structure/devices)
- Review/consensus language and evaluative praise/complaints\
"""

PRODUCTION_EXAMPLES = """\
1) "90s French movies, no gore, edge-of-your-seat" -> "90s French movies"
2) "hand-drawn animation not CGI, grief and healing, date night" -> "hand-drawn animation, not CGI"
3) "Spanish audio, time loop, funny but not dumb" -> "Spanish audio"
4) "directed by Nolan, edge-of-your-seat, big twist ending" -> "directed by Nolan"
5) "low budget indie, intergalactic warfare, uplifting and hopeful" -> "low budget indie"
6) "based on a true story, no gore but creepy, plot holes" -> "based on a true story"
7) "filmed in New York, sisterly bond saves a town, witty dialogue" -> "filmed in New York"
8) "English subtitles, coming-of-age dramedy, sick day comfort" -> "English subtitles"
"""


# ----------------------------
# Reception (expanded)
# ----------------------------

RECEPTION_SCOPE = """\
How the movie is received and discussed in reviews:
- acclaim tier (acclaimed, mixed, disliked)
- evaluative traits people praise/criticize (smart/funny, acting, writing, pacing, plot holes, iconic songs, overrated)
- key attributes people are likely to discuss and evaluate while reviewing this movie\
"""

RECEPTION_INCLUDE = """\
- acclaim tier (acclaimed, mixed, disliked)
- evaluative traits people praise/criticize (smart/funny, acting, writing, pacing, plot holes, iconic songs, overrated)
- key attributes people are likely to discuss and evaluate while reviewing this movie\
"""

RECEPTION_EXCLUDE = """\
- Technique/structure labels stated neutrally (unless judged: “clever misdirection”, “cheap twist”)
- Raw making-of facts (year/country/medium/cast/crew/languages/studios/budget/adaptation) unless evaluated (“great acting”, “cheap CGI”)\
"""

RECEPTION_EXAMPLES = """\
1) "universally acclaimed, set in Victorian London, 1990s" -> "universally acclaimed"
2) "mixed reviews, stunning hand-drawn animation, time loop" -> "mixed reviews, stunning hand-drawn animation"
3) "overrated, directed by Nolan, action movies" -> "overrated"
4) "witty dialogue, filmed in Paris, no gore" -> "witty dialogue"
5) "plot holes, edge-of-your-seat, Spanish audio" -> "plot holes"
6) "funny but not dumb, date night, 90s French movies" -> "funny, not dumb"
7) "like harry potter but with guns, 1980s, no gore" -> "like harry potter but with guns"
8) "bad pacing, big twist ending, low budget indie" -> "bad pacing, big twist ending"
"""

# ----------------------------
# Final assembled prompts
# ----------------------------

PLOT_EVENTS_QUERY_SYSTEM_PROMPT = build_query_router_system_prompt(
    collection_name="Plot Events",
    collection_scope=PLOT_EVENTS_SCOPE,
    include_facets=PLOT_EVENTS_INCLUDE,
    exclude_facets=PLOT_EVENTS_EXCLUDE,
    examples=PLOT_EVENTS_EXAMPLES,
)

PLOT_ANALYSIS_QUERY_SYSTEM_PROMPT = build_query_router_system_prompt(
    collection_name="Plot Analysis",
    collection_scope=PLOT_ANALYSIS_SCOPE,
    include_facets=PLOT_ANALYSIS_INCLUDE,
    exclude_facets=PLOT_ANALYSIS_EXCLUDE,
    examples=PLOT_ANALYSIS_EXAMPLES,
)

VIEWER_EXPERIENCE_QUERY_SYSTEM_PROMPT = build_query_router_system_prompt(
    collection_name="Viewer Experience",
    collection_scope=VIEWER_EXPERIENCE_SCOPE,
    include_facets=VIEWER_EXPERIENCE_INCLUDE,
    exclude_facets=VIEWER_EXPERIENCE_EXCLUDE,
    examples=VIEWER_EXPERIENCE_EXAMPLES,
)

WATCH_CONTEXT_QUERY_SYSTEM_PROMPT = build_query_router_system_prompt(
    collection_name="Watch Context",
    collection_scope=WATCH_CONTEXT_SCOPE,
    include_facets=WATCH_CONTEXT_INCLUDE,
    exclude_facets=WATCH_CONTEXT_EXCLUDE,
    examples=WATCH_CONTEXT_EXAMPLES,
)

NARRATIVE_TECHNIQUES_QUERY_SYSTEM_PROMPT = build_query_router_system_prompt(
    collection_name="Narrative Techniques",
    collection_scope=NARRATIVE_TECHNIQUES_SCOPE,
    include_facets=NARRATIVE_TECHNIQUES_INCLUDE,
    exclude_facets=NARRATIVE_TECHNIQUES_EXCLUDE,
    examples=NARRATIVE_TECHNIQUES_EXAMPLES,
)

PRODUCTION_QUERY_SYSTEM_PROMPT = build_query_router_system_prompt(
    collection_name="Production",
    collection_scope=PRODUCTION_SCOPE,
    include_facets=PRODUCTION_INCLUDE,
    exclude_facets=PRODUCTION_EXCLUDE,
    examples=PRODUCTION_EXAMPLES,
)

RECEPTION_QUERY_SYSTEM_PROMPT = build_query_router_system_prompt(
    collection_name="Reception",
    collection_scope=RECEPTION_SCOPE,
    include_facets=RECEPTION_INCLUDE,
    exclude_facets=RECEPTION_EXCLUDE,
    examples=RECEPTION_EXAMPLES,
)
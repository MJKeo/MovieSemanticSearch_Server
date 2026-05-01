# Endpoint: Keyword (Unified Classification)

## Purpose

Resolves the parts of the call's intent that map onto the closed UnifiedClassification registry — 259 canonical members across 21 families covering genre / sub-genre, story-engine archetypes, source material, cultural tradition, animation form, seasonal framing, narrative mechanics (endings, twists, POV tricks), viewer-response tags, and audience tier. Each finalized member dispatches onto its backing posting list (keyword, source-material, or concept-tag index); execution returns a per-movie hit count which the schema's `scoring_method` converts into a [0,1] score.

## What does NOT belong here

Routing already committed this call to the keyword endpoint — do not refuse. But when decomposing `attributes` from `retrieval_intent` + `expressions`, ignore content from these out-of-scope domains rather than coercing it into a registry member:

- Named real entities (persons, characters, franchises, studios) → entity / franchise / studio endpoints.
- Structured numeric / factual attributes (release date, runtime, rating, country of origin, streaming availability, budget, box office, popularity, reception) → metadata endpoint.
- Free-form thematic / tonal / experiential qualifiers without a registry member ("cozy", "unsettling but not gory", "rainy-day melancholy") → semantic endpoint.
- Awards → award endpoint.

A sibling category handler in the same firing owns those facets.

## Classification registry

{{CLASSIFICATION_REGISTRY}}

When a concept could plausibly fit more than one family, compare candidate definitions directly — the definition that names the concept specifically wins over one that only covers it incidentally.

## Reading inputs as keyword facets

A single `retrieval_intent` + expression set can carry multiple registry-relevant facets simultaneously. Decompose into one `attribute` per distinct facet — not per expression. Typical combinations:

- Genre + cultural tradition: "scary Hindi films" → two attributes (horror feel; Hindi cinema tradition).
- Genre + source material: "biographical dramas" → two attributes (drama feel; biographical source).
- Sub-genre + ending shape: "slasher with a twist" → two attributes (slasher sub-form; twist mechanic).
- Combined member: when the registry has a single member that absorbs both facets (e.g., `TEEN_HORROR` for "teen horror," `HOLIDAY_ROMANCE` for "Christmas rom-com"), collapse to one attribute. The decomposition serves the registry, not the surface phrasing.

Out-of-scope content per the boundaries above is ignored at this step, even if it appears in the inputs.

## Surface forms and aliases

User phrasing often paraphrases canonical labels. Match by meaning, not literal echo:

- "Bollywood" → `HINDI` (the Hindi film tradition, not the audio track).
- "Biopic" → `BIOGRAPHY`.
- "Does the dog die?" / "animal death" → `ANIMAL_DEATH`.
- "Short films" / "shorts" → `SHORT` (form-factor classification, not runtime cutoff).
- "Twist ending" → `PLOT_TWIST`, unless the phrasing names a specific ending type.

These examples illustrate the principle; the right registry match is often definitionally clear even when the inputs do not literally echo the label. Do not force them mechanically.

## Near-collision disambiguation

Within a family, several members often cover overlapping territory. Four principles decide near-collision picks — apply them when surfacing `potential_keywords` and again when committing `finalized_keywords`:

**Breadth vs. specificity.** Prefer the broader member absent signal for a narrower one. "Scary movies" → `HORROR`, not `SLASHER_HORROR` / `PSYCHOLOGICAL_HORROR` / `SUPERNATURAL_HORROR`, unless the inputs cite the sub-form's defining premise. Picking narrow on weak evidence silently rejects every movie in the broad category that lacks the specific tag.

**Explicit premise signal.** Prefer a narrower member only when the inputs cite the premise that defines it. `SLASHER_HORROR` needs stalker/killer phrasing. `ZOMBIE_HORROR` needs zombies cited. `HEIST` needs the theft / crew / plan premise. If the premise is not in the text, the broader family member wins.

**Cross-family proximity.** Some concepts sit on family boundaries. "Coming-of-age" (family 12 audience/life-stage) vs "teen drama" (family 5 drama-with-teen-audience). "True story" (family 18 real-world-basis) vs "biography" (same family, narrowed to one named person). "Remake" (generic retelling, family 18) vs a tracked franchise's remake (franchise endpoint's concern). Decide by which feature the inputs emphasize — audience, life stage, person, retelling motion — and pick the family whose definition names that feature.

**Mutually exclusive ending / viewer-response pairs.** Members inside families 19 (endings) and 21 (viewer response) are near-mutually-exclusive. "Makes you cry" → `TEARJERKER`. "Leaves you uplifted" → `FEEL_GOOD`. "Unexpected ending" → `PLOT_TWIST`, not a specific ending-type. Cite the input phrase that names the effect, not your own summary.

## Reading retrieval_intent for scoring_method

`scoring_method` is a downstream commitment driven by `retrieval_intent`'s framing of how the finalized members combine.

**ANY.** We only care if the movie has at least one of the finalized members, like an "or" case. Movies score equally high for matching 1+ values. Cues: "any kind of", "some flavor of", "or", "either", listing alternatives, one concept paraphrased several ways.

**ALL.** We care how many finalized members a given movie matches. Movies score higher depending on how many values they match. Cues: "and", "as well as", "both", "needs to be", separate facets each independently named, AND-style coverage requirements.

When `retrieval_intent` is silent on combination — typical for single-attribute calls — ANY is the safe default.

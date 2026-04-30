# Endpoint: Keyword (Unified Classification)

## Purpose

Resolves a requirement onto **exactly one member** of the unified classification registry — a closed vocabulary of 259 canonical concept tags, source-material labels, and cultural/format classifications. Execution dispatches the chosen member onto its backing posting list (keyword, source-material, or concept-tag index) and returns movies whose ingest-time classification includes it.

## Canonical question

"Which single registry member best identifies the user's described concept?"

## Capabilities

- Closed-enum lookup. The registry is finite; every emission must be one of its 259 members.
- Covers genre, sub-genre, story-engine archetype, source material / adaptation type, cultural/national cinema tradition, animation form, seasonal framing, narrative mechanics (endings, twists, POV tricks), viewer-response tags, and audience tier.
- Tolerates paraphrased input — match by the member's definition, not by literal label overlap with the query text.

## Boundaries (what does NOT belong here)

- Named real entities (persons, characters, franchises, studios) → entity / franchise / studio endpoints.
- Structured numeric / factual-logistical attributes (release date, runtime, rating, country of origin, streaming availability, budget, box office, popularity, reception) → metadata endpoint.
- Free-form thematic / tonal / experiential qualifiers without a canonical registry member ("cozy", "unsettling but not gory", "rainy-day melancholy") → semantic endpoint.
- Awards → award endpoint.

## Classification registry

Every member of the registry is listed below, grouped into the twenty-one canonical concept families. Each entry shows the exact member name to emit, followed by its definition. The `classification` parameter must match one of these names exactly. When a concept could plausibly fit more than one family, compare the candidate definitions directly — the definition that names the concept specifically wins over one that only covers it incidentally.

{{CLASSIFICATION_REGISTRY}}

## Surface forms and aliases

User phrasing often paraphrases canonical labels. Match by meaning, not literal echo:

- "Bollywood" → HINDI (the Hindi film tradition, not the audio track).
- "Biopic" → BIOGRAPHY.
- "Does the dog die?" / "animal death" → ANIMAL_DEATH.
- "Short films" / "shorts" → SHORT (form-factor classification, not runtime cutoff).
- "Twist ending" → PLOT_TWIST, unless the phrasing clearly names a specific ending type instead.

Do not force these examples mechanically. They illustrate that the right match can be definitionally clear even when the query does not literally echo the label.

## Near-collision disambiguation

Four comparison principles decide within-family and cross-family close calls:

**Breadth vs. specificity.** Prefer the broader member absent signal for a narrower one. "Scary movies" picks HORROR, not SLASHER_HORROR / PSYCHOLOGICAL_HORROR / SUPERNATURAL_HORROR, unless the query cites that sub-form's defining premise. Picking a narrow member on weak evidence silently drops every movie in the broad category that lacks the specific tag.

**Explicit premise signal.** Prefer a narrower member only when the query cites its defining premise. SLASHER_HORROR needs stalker/killer phrasing. ZOMBIE_HORROR needs zombies cited. HEIST needs the theft / crew / plan premise. If the premise is not in the text, the broader family member wins.

**Cross-family proximity.** Some concepts sit on family boundaries. "Coming-of-age" (family 12 audience/life-stage) vs. "teen drama" (family 5 drama-with-teen-audience). "True story" (family 18 real-world-basis) vs. "biography" (same family, narrowed to one person's named life). "Remake" (generic retelling, family 18) vs. a tracked franchise's remake (franchise endpoint's concern). Decide by which feature the query emphasizes — the audience, the life stage, the person, the retelling motion — and pick the family whose definition names it.

**Mutually exclusive ending / tag pairs.** Members inside families 19 (endings) and 21 (viewer response) are near-mutually-exclusive. "Makes you cry" → TEARJERKER. "Leaves you uplifted" → FEEL_GOOD. "Unexpected ending" → PLOT_TWIST, not a specific ending-type. Cite the query phrase that names the effect, not your own summary.

## Scope discipline

Exactly one member per firing. No list, no abstention at member-picking time (the bucket-level guardrails govern whether the endpoint fires at all). If no member is a clean fit but the bucket decides the endpoint should still fire, pick the closest definitionally supported member — do not force a broader label that the definition does not really support.

## 1) Project overview

You’re building a **movie search/recommendation system** where a user types a freeform query (phone search bar style), and the system returns the best-matching movies.

The system blends three sources of signal:

* **Lexical entity search** (titles, people, characters, franchises, studios)
* **Metadata preference scoring** (year, runtime, genre, language, platforms, etc.)
* **Vector semantic search** across multiple vector spaces (each vector is a different “lens”)

The system uses **8 vector collections**:

1. **Anchor** (everything / “movie card” broad recall) — treated as always on with fixed weight, not evaluated here
2. **Plot Events**
3. **Plot Analysis**
4. **Viewer Experience**
5. **Watch Context**
6. **Narrative Techniques**
7. **Production**
8. **Reception**

---

## 2) Search pipeline (MVP-level)

### Phase 1: Query understanding (parallel)

All run in parallel for latency:

1. **Lexical entity extraction** (find entity-like substrings; expand aliases/phonetics; categorize as PERSON/CHARACTER/TITLE/STUDIO, etc.)
2. **Metadata preference extraction** (structured preferences: year ranges, runtime ranges, genre must/must-not, languages, etc.)
3. **Vector query routing (per vector)** ✅ *(this is what you’re evaluating)*

   * For each vector space, produce a per-vector **subquery** containing only the relevant parts of the raw query.
4. **Channel weights** (separate model) — determines relative weight of lexical vs metadata vs vector in final scoring (not part of this evaluation).

### Candidate generation

* UI “hard filters” apply first.
* **Lexical retrieval** returns ranked candidates by entity match.
* **Vector retrieval** runs per active vector using **two queries**:

  * the **original raw query**
  * the **router-produced per-vector rewritten query**
* **80/20 rule**: per vector, 80% of candidates come from rewritten query and 20% from original query (preserve recall).

### Candidate unioning

* Progressive union & dedupe until up to ~1000 candidates.

### Reranking

* Recompute exact similarities.
* **Per-vector z-score normalization** of similarity (clamp → scale to [0,1]).
* Combine original+rewritten for each vector: **0.2 original + 0.8 rewritten**.
* Fuse across vectors with per-vector weights (from a separate model) to get a single **vector_score** in [0,1].
* Combine with lexical_score + metadata_score (+ session penalty).

---

## 3) The LLM under evaluation: “Vector Query Router” (per vector)

### Purpose

Given the user’s **raw query**, generate a **single string**: a comma-separated list of **only the fragments relevant to one vector space**, with light normalization. If nothing is relevant, output null.

### Input

* One string: the user’s raw query.

### Output (strict JSON)

Exactly:

```json
{"relevant_subquery_text": "comma, separated, phrases"}
```

or

```json
{"relevant_subquery_text": null}
```

### Core constraints

* **Extraction only**: Output must be derived directly from the user’s text.
* **No inference**: Do not add tags, themes, synonyms, related ideas, antonyms, or “likely” implications.
* **No expansions except the explicit normalization rules below.**
* **Do not “salvage” output**: If no fragment clearly belongs in this category, return null.
* Output should be **phrases**, not sentences.

### Normalization (allowed)

The router may:

* fix misspellings **only when extremely confident** (especially canonical names/terms)
* normalize punctuation/spacing
* normalize decades: `"80s"` → `"1980s"`, `"90s"` → `"1990s"`
* expand common acronyms **only when confident & unambiguous**, optionally keeping the original too:

  * `"CG" / "CGI"` → `"computer-generated", "computer-generated imagery"`
  * `"romcom"` → `"romantic comedy"`
  * `"doc"` → `"documentary"`
  * `"YA"` → `"young adult"`
  * `"found footage"` stays as-is
  * `"sci-fi"` stays as-is

### Required decision procedure

1. Identify fragments that match the vector’s **INCLUDE** definition.
2. Remove any fragments that match the vector’s **EXCLUDE** definition.
3. If nothing remains: output null.
4. Else: output remaining fragments as comma-separated phrases.

---

## 4) Ground-truth vector definitions (authoritative)

Use these definitions as the “what belongs here” truth.

### A) Plot Events

Concrete, story-internal content:

* **all events of the plot** (who did what, where, when, why, how)
* **setting in the story world** (where/when it takes place)
* **character motivations** grounded in the plot (what they do and why they do it)

Examples of belonging:

* “detective solves a murder on a train”
* “set in Victorian London during winter”
* “two strangers escape the city handcuffed together”
* “a dog pees on a fire hydrant”

**Should NOT include:** vibes, genre labels, production facts, review language, storytelling-technique labels unless they directly describe plot events (generally they don’t).

### B) Plot Analysis

Abstract meaning and classification:

* what “type” of movie it is (generalized)
* genre terms/signatures (action, comedy, thriller, sci-fi, romcom)
* themes / central questions (grief, identity, revenge)
* character arcs (redemption, corruption, healing)
* conflict scale (personal/community/global/cosmic)
* lessons learned (by audience and characters)
* generic plot beats phrased thematically:

  * core concept (one sentence)
  * general plot overview in generalized/thematic wording

Examples of belonging:

* “coming-of-age dramedy”
* “explores grief and healing”
* “action movies” / “smart comedy”
* “sisterly bond saves a town”
* “man spirals into madness”
* “intergalactic warfare”

### C) Viewer Experience

What the viewer experiences internally:

* emotional palette
* tone (earnest/cynical/heartfelt/satirical)
* tension (adrenaline/energy/suspense/stress)
* cognitive intensity (confusing/digestible/thought-provoking)
* sensory intensity (jarring/soothing visuals and sound)
* disturbance (fear/disgust/gore/moral uneasiness/dread/jump scares)
* emotional volatility
* ending aftertaste

Examples of belonging:

* “uplifting and hopeful”
* “edge-of-your-seat”
* “not too intense, not slow”
* “mentally taxing”
* “no gore but creepy”
* “ear-bursting sound”
* “leaves a bad taste in your mouth”

### D) Watch Context

Why/when to watch:

* internal motivations (unwind, laugh, cathartic cry, heart racing)
* external motivations (learn something new, sparks debate, cultural relevance, “everyone recommends it”)
* specific viewing scenarios (date night, sick day, friends movie night, background at a party)
* key features the user is selecting for (evaluative attributes like good soundtrack, iconic quotes, great dialogue, beautiful cinematography)

Examples of belonging:

* “date night”
* “sick day comfort”
* “something to unwind”
* “iconic songs”
* “make me piss myself laughing”

### E) Narrative Techniques

How the story is told via classic techniques:

* POV/perspective, temporal structure, narrative archetype, information control, characterization methods
* character arcs, conflict stakes design, thematic delivery, audience character perception
* meta techniques, etc.

Examples of belonging:

* “unreliable narrator”
* “nonlinear timeline”
* “time loop”
* “big twist ending”
* “fouth wall breaks”
* “underdog quest / adventure”
* “red herrings”
* “foil characters”
* “redemption arc”
* “love-to-hate villain”
* “ticking clock deadline”
* “moral argument embedded in choices”

### F) Production

How the movie was made in the real world:

* medium (hand-drawn vs CGI), release decade/year
* country of origin / filming location
* language/subtitles/audio available
* cast/crew, studios, budget/scale, adapted from novel/true story/game

Examples of belonging:

* “90s French movies”
* “hand-drawn animation not CGI”
* “Spanish audio”
* “directed by Nolan”
* “low budget indie”
* “based on a true story”

### G) Reception

How it’s received and discussed in reviews:

* acclaim tier (acclaimed, mixed, disliked)
* evaluative traits people praise/criticize (smart/funny, acting, writing, pacing, plot holes, iconic songs, overrated)
* key attributes people are likely to discuss and evaluate while reviewing

Examples of belonging:

* “universally acclaimed”
* “mixed reviews”
* “overrated”
* “witty dialogue”
* “plot holes”
* “funny but not dumb”
* “like harry potter but with guns”

---

## 5) What the evaluation should measure (for the router)

For each vector’s router output given a query, evaluate:

1. **Precision (no bleed):** output contains *only* fragments that belong per the definition.
2. **Recall (extraction):** if the query contains relevant fragments, they are included.
3. **Null correctness:** outputs null when the query has no relevant fragments for that vector.
4. **Normalization correctness:** decades/acronyms/misspellings corrected only under the allowed rules.
5. **Format correctness:** strict JSON, correct key, phrases not sentences.
# Question → Endpoint Mapping

Maps each of the 59 question-atoms in [query_questions.md](query_questions.md)
to the endpoints that carry genuinely valuable signal for it, flags
data gaps, and groups questions by *both* conceptual flavor *and*
database retrieval shape.

## Endpoint legend

Lexical / structured:
- **ENT** — entity (person, character, title substring; lex posting tables)
- **FRA** — franchise (lineage / shared universe / subgroup arrays)
- **STU** — studio (ProductionBrand enum or freeform company tokens)
- **KW** — keyword (UnifiedClassification: OverallKeyword / SourceMaterialType / ConceptTag, three GIN array columns)
- **META** — metadata (release_ts, runtime, maturity, genre_ids, language, country, streaming, popularity, reception_score, budget_bucket, box_office_bucket)
- **AWD** — award (ceremony / award_name / category / outcome / year)

Semantic (Qdrant, 8 vector spaces):
- **ANC** — dense_anchor (holistic identity)
- **P-EVT** — plot_events (literal events, setting, subject)
- **P-ANA** — plot_analysis (themes, conflict, arc archetypes)
- **VWX** — viewer_experience (moment-to-moment feel, tone, pacing)
- **CTX** — watch_context (occasion, motivation, feature-draws)
- **NRT** — narrative_techniques (structure, POV, devices, craft-how)
- **PRD** — production_techniques (filming location, production method)
- **RCP** — reception (praised / criticized qualities, reception prose)

Notation: **Primary** (best-fit channel) · Secondary (useful support) · ⚠ gap
note when nothing covers cleanly.

---

## Objective tier

1. **Who made it (person credits)** → **ENT** (actor/director/writer/producer/composer postings).
2. **Below-the-line creative credits** → **ENT** for composer only. ⚠ Gap: cinematographer, editor, production designer, costume designer have no posting tables; the only way they surface today is incidentally in **RCP** text ("praised cinematography by Deakins") — unreliable for lookup.
3. **Studio / brand** → **STU** (brand path + freeform fallback).
4. **Named character** → **ENT** (character postings). Secondary **FRA** when the character *is* the franchise anchor (Batman, Wolverine) — need cross-endpoint coordination.
5. **Top-level genre** → **META** (genre_ids GIN overlap). Secondary **KW** (genre-flavor families) and **P-ANA** (genre_signatures).
6. **Release date / era** → **META** (release_ts).
7. **Runtime** → **META** (runtime_minutes).
8. **Maturity rating** → **META** (maturity_rank).
9. **Audio language** → **META** (audio_language_ids).
10. **Streaming availability** → **META** (watch_offer_keys).
11. **Budget scale** → **META** (budget_bucket).
12. **Box office** → **META** (box_office_bucket).
13. **Trending** → **META** (popularity_score). ⚠ Gap-ish: no dedicated trending endpoint surfaced in the audit; popularity is a static field, not a "right now" signal unless refreshed.
14. **Formal awards** → **AWD**.
15. **Curated canon (Criterion, AFI, Sight & Sound, IMDb 250)** → ⚠ **Data gap**. Not encoded as awards, not a METADATA column, not present in the UnifiedClassification vocabulary. Might incidentally surface in **RCP** ("Criterion pick") but not queryable.
16. **Numeric reception score** → **META** (reception_score).
17. **Format (doc, short, anime, mockumentary)** → **KW** (SourceMaterialType + format-flavor OverallKeywords). Secondary **META** (runtime for shorts).
18. **Visual-format specifics (B&W, 70mm, found-footage, widescreen)** → **KW** for the canonical tags that exist (FOUND_FOOTAGE, BLACK_AND_WHITE, ANIMATION, etc.). Secondary **PRD** for "shot on 16mm" style prose.
19. **Title text pattern** → **ENT** (title ILIKE path).
20. **Production country** → **META** (country_of_origin_ids).
21. **Filming location** → **PRD** (filming_locations field). ⚠ Note: not **META.country_of_origin** — that's legal/financial origin, not where cameras rolled.

## Murky tier

1. **Franchise membership** → **FRA** (lineage + shared_universe arrays).
2. **Franchise lineage (sequel/prequel/spinoff/reboot/remake)** → **FRA** (spinoff/crossover/launched_franchise flags). Secondary **KW** (SourceMaterialType.REMAKE for the remake case specifically).
3. **Cultural tradition (Korean cinema, Bollywood, neorealism)** → **META** (country_of_origin) as base. Secondary **KW** (ConceptTag for "Italian Neorealism"-style schools if present). **P-ANA** can catch prose-level tradition but weakly. ⚠ Partial gap: "cultural tradition" ≠ country; no clean channel for "Hong Kong action" as an aesthetic tradition vs. the literal country code.
4. **Sub-genre** → **KW** (families 1–11 are genre-flavor buckets). Secondary **P-ANA** (genre_signatures).
5. **Setting time (narrative era)** → **P-EVT** (prose mentions era). Secondary **KW** if period-setting ConceptTags exist. ⚠ Partial gap: no structured "story_era" column, so queries over the 1940s (narrative) vs. 1940s (release) bleed together unless prose carries the weight.
6. **Setting place (narrative location)** → **P-EVT** (prose). Secondary **KW** (settings ConceptTags if present).
7. **Plot events** → **P-EVT** (the space's entire reason for existing).
8. **Source material / adaptation** → **KW** (SourceMaterialType is literally designed for this).
9. **Real-world people/events depicted (subject-of)** → **P-EVT** (subject usually in synopsis). Secondary **KW** (SourceMaterialType.TRUE_STORY, OverallKeyword.BIOGRAPHY). ⚠ Important distinction: this is *what the movie is about*, not *who made it* (ENT) and not *whether it's adapted* (KW source-material flag alone).
10. **Narrative devices (plot twist, nonlinear, unreliable narrator)** → **NRT**. Secondary **KW** (some devices are OverallKeyword tags).
11. **Story archetype (revenge, underdog, con-artist, post-apoc)** → **KW** (ConceptTag archetype family 20). Secondary **P-ANA** (character_arcs).
12. **Cast composition (ensemble, two-hander, all-female)** → **KW** (if ENSEMBLE-style tags) + **NRT** (ensemble structure). ⚠ Partial gap: "stacked A-list cast" needs ENT-level popularity-of-cast aggregation that no single endpoint does today.
13. **Character prominence** → **ENT** (actor zones + character CENTRAL/DEFAULT).
14. **Era feel (classics / old / modern)** → **META** (release_ts as proxy).
15. **Sensitive content (gore, nudity, violence)** → **KW** (content ConceptTags / OverallKeywords) + **META** (maturity_rank). Secondary **VWX** (disturbance_profile).
16. **Seasonal / holiday** → **KW** (family 14 is exactly this).
17. **Audience / life-stage (family, teen, coming-of-age)** → **KW** (family 12) + **META** (maturity_rank).

## Subjective tier

1. **Kind of story** → **P-ANA** (primary — thematic_concepts, elevator_pitch, conflict_type). Secondary **KW** (ConceptTag thematic families) and **ANC** (when the "kind" is a vibe spanning multiple axes).
2. **Thematic weight ("has something to say")** → **ANC** + **P-ANA**. Secondary **RCP** (reviews often say "substantive", "has something to say"). ⚠ Partial gap: this meta-quality isn't directly encoded anywhere; it's inferred.
3. **Character-focused vs plot-focused** → **NRT** (characterization_methods, information_control). ⚠ Partial gap: weakly captured; no explicit axis.
4. **How told (slow burn, frenetic, dreamlike)** → **NRT** (narrative_delivery) + **VWX** (pacing aspect of tension/tone).
5. **Realism / stylization mode** → **VWX** (tone_self_seriousness) + **NRT**.
6. **Scale and scope (epic, intimate)** → **ANC** (best captures scope as identity). Secondary **P-ANA**. ⚠ Partial gap: "epic" vs "intimate" isn't an explicit field.
7. **Feel to watch** → **VWX** (the entire reason this space exists: emotional_palette, tension, disturbance, sensory_load).
8. **Cognitive demand** → **VWX** (cognitive_complexity).
9. **Post-viewing resonance (lingers, haunting)** → **VWX** (ending_aftertaste). Secondary **RCP**.
10. **Rewatch value** → **RCP** (reviews often mention rewatchability). Secondary **CTX** (feature-draws for "go-to rewatch"). ⚠ Partial gap: no explicit rewatch signal; inferred from prose.
11. **Occasion / use-case (date night, background)** → **CTX** (watch_scenarios is exactly this).
12. **Gateway / entry-level** → ⚠ **Data gap**. Nothing captures "good first anime" or "accessible arthouse" directly. Sometimes bleeds into **CTX** motivations or **RCP** ("accessible") but unreliable.
13. **Comfort-watch archetype** → **CTX** (self_experience_motivations + scenarios both carry this).
14. **Craft: visual / technical** → **RCP** (praised_qualities: cinematography, production design). Secondary **PRD** (when about the *how*, not the aesthetic judgment).
15. **Craft: music / score** → **RCP** (praised "score"). Secondary **ENT** (composer presence, but that's a different question).
16. **Craft: dialogue** → **RCP** (praised "dialogue", "screenplay"). Secondary **NRT** ("Sorkin-style" is a craft-delivery pattern).
17. **Reception qualitative (cult, acclaimed, underrated, divisive)** → **RCP** (reception_summary + praised/criticized tags). Secondary **META** (reception_score for baseline).
18. **Cultural influence / historical importance** → ⚠ **Mostly a gap**. Nothing directly captures "era-defining" or "invented the genre." Leaks into **RCP** when reviews say it, and maybe **KW** if a "CULT_CLASSIC"-style tag exists.
19. **Still holds up today** → ⚠ **Partial gap**. Weak **RCP** signal if reviews discuss aging. Otherwise uncaptured.
20. **Quality superlative ("best", "scariest", "funniest")** → **META** (reception_score for generic "best") + **AWD** + **RCP**. Per-axis superlatives ("scariest") need the genre constraint from **KW/META** combined with a reception ranking.
21. **Tonal aesthetic (dark, whimsical, gritty, wholesome)** → **VWX** (tone_self_seriousness + emotional_palette). Secondary **ANC**.

---

## Data gaps (consolidated)

Flagging so they're not forgotten.

**Hard gaps** (nothing answers this today):
- **Below-the-line credits** beyond composer: cinematographer, editor, production designer, costume designer.
- **Curated canon**: Criterion, AFI, Sight & Sound, IMDb Top 250 membership.
- **Gateway / entry-level** designation: "good first anime", "accessible arthouse."

**Partial gaps** (captured incidentally but not queryable cleanly):
- Cultural tradition as distinct from production country (Hong Kong action, Bollywood).
- Narrative setting time — bleeds into release date when prose doesn't carry it.
- Cast composition quality ("stacked A-list") — no cross-endpoint cast-popularity aggregate.
- "Something to say" / thematic weight as a meta-quality.
- Character-vs-plot focus axis.
- Scale/scope (epic vs intimate).
- Rewatch value.
- Cultural influence / historical importance.
- Still holds up today.
- Trending *right now* (popularity_score is static, not a live trending feed).

---

## Groupings

Grouped by *both* conceptual flavor *and* database retrieval shape
(same question-kind + same DB mechanism). The mechanism column is
how each group actually lands in the database.

### G1 · Named-entity lookup
**Mechanism:** normalize → token index → lex posting-table intersect → score by prominence where applicable.
**Questions:** Obj-1 (person credits), Obj-2 (below-the-line, partial), Obj-4 (named character), Obj-19 (title text), Murky-13 (character prominence).
**Justification:** All five resolve a proper-noun-ish surface form to an `entity_id` set via `lex.inv_*_postings` (or ILIKE for titles) and score by posting-list membership / prominence zone. Same normalization pipeline, same posting shape, same scoring skeleton.

### G2 · Franchise-lineage membership
**Mechanism:** token-resolve franchise name → GIN `&&` on `lineage_entry_ids` / `shared_universe_entry_ids` → structural flag filter.
**Questions:** Murky-1 (franchise membership), Murky-2 (franchise lineage).
**Justification:** Same two-phase token-to-entry resolution and same array-overlap scoring. Lineage-vs-universe split and spinoff/crossover/launched flags are orthogonal modifiers over the same retrieval shape. Character-anchored franchises (Batman) span G1+G2 and need a dispatch-layer fan-out.

### G3 · Studio attribution
**Mechanism:** ProductionBrand enum → brand_id posting, or normalize → DF-filtered token intersect → `production_company_ids && [...]`.
**Questions:** Obj-3 (studio).
**Justification:** Singleton group. Brand and freeform paths collapse to the same column overlap; distinct from G1 because the DF-ceiling and per-name intersection semantics don't match the prominence-zone logic.

### G4 · Award record lookup
**Mechanism:** ceremony/category/outcome filter on `movie_awards` + fast path on `award_ceremony_win_ids`.
**Questions:** Obj-14 (formal awards). Partially Subj-20 (quality superlative via awards).
**Justification:** Singleton-plus. The retrieval shape (structured row filter + count threshold) is unique to this endpoint.

### G5 · Structured scalar / range / array attribute
**Mechanism:** direct `movie_card` column predicate; ranges via decay, array_ids via GIN overlap, numeric via tanh/linear compression.
**Questions:** Obj-5 (top genre), Obj-6 (release date), Obj-7 (runtime), Obj-8 (maturity), Obj-9 (audio language), Obj-10 (streaming), Obj-11 (budget), Obj-12 (box office), Obj-13 (trending), Obj-16 (numeric reception), Obj-20 (production country), Murky-14 (era feel), Murky-17 (partial — maturity side).
**Justification:** All live on a first-class `movie_card` column and share the soften-where-needed / graceful-decay scoring pattern. Top-level genre technically straddles G5 and G6 (genre_ids column is structured but genre flavor overlaps with KW families) — leaving it here because the structured path is the primary one.

### G6 · Taxonomy tag overlap
**Mechanism:** LLM picks a UnifiedClassification member → resolve to (backing_column, source_id) → single GIN `&&` against `keyword_ids` / `source_material_type_ids` / `concept_tag_ids`.
**Questions:** Obj-17 (format), Obj-18 (visual-format specifics — canonical side), Murky-8 (source material / adaptation), Murky-11 (story archetype), Murky-15 (sensitive content — partial), Murky-16 (seasonal), Murky-17 (audience life-stage — vocabulary side), Murky-12 (cast composition — tag side), Murky-10 (narrative devices — tag side), Murky-4 (sub-genre — canonical families side).
**Justification:** All resolve to a *single discrete vocabulary member* and score binary-match. The three-way backing-column split is hidden from the LLM (one enum, one overlap). Distinct from G5 because genre_ids is a closed structured enum with hand-curated IDs; KW is a much wider vocabulary with soft thematic meaning.

### G7 · Literal / categorical narrative semantics
**Mechanism:** Qdrant cosine against **plot_events** and **plot_analysis** spaces; structured-label query text.
**Questions:** Murky-5 (setting time), Murky-6 (setting place), Murky-7 (plot events), Murky-9 (real-world subject depicted), Subj-1 (kind of story, P-ANA side), Murky-4 (sub-genre, P-ANA side).
**Justification:** These ask *what it's literally about* (events, setting, subject) or *what category of story it is* (archetype, thematic pitch). Both spaces embed narrative/thematic prose and share the same Qdrant retrieval shape with structured-label input. Splitting them into sub-groups per space if dispatch needs to, but they're the same flavor from the user's perspective.

### G8 · Experiential semantics
**Mechanism:** Qdrant cosine against **viewer_experience**; embeds feeling/tone/pacing labels with explicit negations.
**Questions:** Subj-4 (how told — pacing side), Subj-5 (realism / stylization), Subj-7 (feel to watch), Subj-8 (cognitive demand), Subj-9 (post-viewing resonance — first-viewing side), Subj-21 (tonal aesthetic), Murky-15 (sensitive content — intensity side).
**Justification:** All ask what it *feels like* — emotional / sensory / cognitive — not what happens. Same space, same structured-label query, same calibration.

### G9 · Situational / occasion semantics
**Mechanism:** Qdrant cosine against **watch_context**; embeds motivations, scenarios, feature-draws.
**Questions:** Subj-11 (occasion), Subj-13 (comfort-watch archetype), Subj-12 (gateway — partial, leaks here).
**Justification:** Asks *why* or *when* someone watches, not what the movie is. Deliberately plot-free embedding space; any question where the answer depends on the viewing situation lands here.

### G10 · Craft / how-told semantics
**Mechanism:** Qdrant cosine against **narrative_techniques**.
**Questions:** Murky-10 (narrative devices — prose side), Murky-12 (cast composition — structural side), Subj-3 (character vs plot focus), Subj-4 (how told — craft side), Subj-16 (dialogue style — craft side).
**Justification:** All ask *how* the story is delivered at the craft level (structure, POV, characterization method), orthogonal to *what* (G7) or *feel* (G8).

### G11 · Production / physical-making semantics
**Mechanism:** Qdrant cosine against **production_techniques**.
**Questions:** Obj-18 (visual-format specifics — prose side), Obj-21 (filming location), Subj-14 (craft visual — production side).
**Justification:** Physical making of the film. Filming location belongs here (not META) because META.country_of_origin is legal/financial, not geographic where-shot.

### G12 · Reception / response semantics
**Mechanism:** Qdrant cosine against **reception**; also leans on META.reception_score for numeric baseline.
**Questions:** Subj-2 (thematic weight — partial), Subj-9 (post-viewing resonance — critical side), Subj-10 (rewatch value), Subj-14 (craft visual — acclaim side), Subj-15 (music / score), Subj-16 (dialogue — acclaim side), Subj-17 (reception qualitative), Subj-18 (cultural influence — weak), Subj-19 (still holds up — weak), Subj-20 (quality superlative — reception side).
**Justification:** All ask about *response* (critical or audience) to specific qualities, not intrinsic properties of the film. Reception space embeds praised/criticized qualities; numeric METADATA.reception_score is the baseline companion.

### G13 · Holistic identity semantics
**Mechanism:** Qdrant cosine against **dense_anchor**; intended default when query spans multiple axes.
**Questions:** Subj-1 (kind of story — vibe side), Subj-2 (thematic weight — vibe side), Subj-6 (scale / scope).
**Justification:** When the ask doesn't zero in on one dimension (events / feel / occasion / craft / response), ANC is the right default. Fallback catch for fuzzy preference queries.

---

## Justification of the grouping structure

Groups are organized along two axes that happen to align:

1. **What the user is asking about** (named entity, structured attribute, narrative semantic, experiential, etc.).
2. **How the database answers it** (posting-table intersect, GIN array overlap, Qdrant cosine against a specific space, etc.).

These axes align because the endpoints were deliberately designed
around user-facing question-kinds. The tight alignment means each
group has a dominant mechanism, even when multiple endpoints carry
signal (e.g. top-level genre sits in G5 structured but also feeds
G6 taxonomy and G7 narrative semantics — the *primary* group is
determined by which mechanism is cheapest / most authoritative).

The 13 groups split roughly:
- **G1–G4** lexical / proper-noun (exact-match flavor)
- **G5–G6** structured / vocabulary (closed-schema flavor)
- **G7–G13** seven semantic sub-groups, one per vector space that
  participates as a *primary* answerer (anchor, plot events+analysis,
  viewer experience, watch context, narrative techniques, production,
  reception). Plot_events and plot_analysis collapse into G7 because
  the user-level flavor ("what's it about / what kind of story") is
  shared and dispatch can decide between them internally.

Questions appearing in multiple groups (Subj-1, Murky-4, Obj-18,
etc.) are genuinely multi-endpoint and the dispatch layer needs to
fan out — that's the expected pattern, not a flaw in the grouping.

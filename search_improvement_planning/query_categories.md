# Query Categories (final)

Each category names a conceptual flavor of question AND the concrete
set of endpoints the dispatcher fans out to when that category fires.
Categories **compose**: a single user query routinely invokes several
(e.g. "Tom Hanks comedies from the 90s" hits Cat 1 + Cat 10 + Cat 9).
Splits were made wherever a sub-case would call a *different* set of
endpoint families — per the rule that different endpoint sets =
different categories.

**Endpoint-family shorthand.** ENT · FRA · STU · KW · META · AWD ·
TRENDING · ANC · P-EVT · P-ANA · VWX · CTX · NRT · PRD · RCP · INTERP.
Semantic sub-spaces (ANC, P-EVT, …) are treated as distinct endpoint
families for category-split purposes because each one embeds a
different information surface.

---

## Structured / lexical categories

### 1. Credit + title text
**Endpoints:** ENT.
**Handles:** actor, director, writer, producer, composer credits; title substring matches.
**Mechanism:** normalize → `lex.inv_<role>_postings` intersect (or `movie_card.title_normalized ILIKE` for title).
**Why single-endpoint:** posting tables are authoritative for indexed roles. No other channel carries reliable proper-noun credit signal — adding semantic fan-out would *dilute* precision, not enrich it.
**Below-the-line fallthrough:** cinematographer / editor / production designer / costume designer aren't indexed as postings — those queries route to Cat 28 (Below-the-line creator lookup) instead.

### 2. Named character lookup
**Endpoints:** ENT + FRA.
**Handles:** named-character presence and prominence ("Batman movies," "any Wolverine appearance").
**Mechanism:** ENT character postings (with CENTRAL/DEFAULT prominence) ∪ FRA lineage/universe overlap on the character's franchise_entry_id.
**Why fan out:** many iconic characters *are* franchise anchors. A cameo or crossover may escape the character-postings index but still register as MCU/DC/Star-Wars membership in FRA. Fan-out catches the union.
**Why split from Cat 1:** person credits need only ENT; characters genuinely need FRA too. Different endpoint set → separate category.

### 3. Studio / brand attribution
**Endpoints:** STU.
**Handles:** production-company and curated-brand queries.
**Mechanism:** ProductionBrand enum path (brand_id posting, with rename-chain time-bounding) OR freeform token intersect with DF ceiling.
**Why single-endpoint:** brand rename handling and DF-filtered token intersection are unique to STU; no other channel carries production-company signal with any discipline.

### 4. Franchise / universe lineage
**Endpoints:** FRA + KW.
**Handles:** franchise/universe membership, lineage position (sequel/prequel/spinoff/crossover/launched), mainline-vs-offshoot, reboot/remake positioning.
**Mechanism:** FRA two-phase token resolution → array overlap on lineage / shared_universe / subgroup. KW contributes SourceMaterialType.REMAKE when the remake-or-not axis is load-bearing.
**Why fan out:** "the original, not the remake" is a FRA lineage query; "movies that are remakes" is a KW source-material query; they blur at the edges and fanning out both channels catches both.

### 5. Adaptation source flag
**Endpoints:** KW.
**Handles:** "novel adaptation," "comic book movie," "based on a true story," "video-game adaptation," "biography," "remake" (as an adaptation flag rather than lineage positioning).
**Mechanism:** KW (SourceMaterialType family) single-overlap.
**Why split from Cat 4:** Cat 4 asks "where does this sit in a named franchise?" Cat 5 asks "what's the *origin medium*?" — different conceptual flavor and the primary endpoint differs (FRA vs KW). When both axes are asked simultaneously the query fires both categories.

### 6. Specific subject / element / motif presence
**Endpoints:** P-EVT + KW.
**Handles:**
- Real-world subjects: "about JFK," "Titanic movie," "Watergate," "Princess Diana biopic," "Vietnam War."
- Fictional element / motif presence: "movies with clowns," "zombie movies," "shark movies," "robots."
- Presence of the *thing* in the story, as distinct from Cat 2 (specific *named* character like Batman) and Cat 14/Cat-new (character *archetype* like "lovable rogue").

**Mechanism:** P-EVT prose typically carries subject and element presence + KW supports with canonical tags (BIOGRAPHY, TRUE_STORY for real subjects; ZOMBIE, SHARK, CLOWN for motifs).

**Why fan out:** discrete tags give precision for common elements; P-EVT prose catches long-tail subjects/elements that never made it to the vocabulary and narrows matches to movies where the subject is actually central, not a passing mention. KW alone can't distinguish *which* real event; P-EVT alone risks matching on coincidental name mentions.

**Why split from Cat 5:** source-flag (Cat 5) is a yes/no flag; subject-presence (Cat 6) is specific-subject identity. Different endpoint set.

### 6b. Character archetype
**Endpoints:** KW + P-EVT + NRT.
**Handles:** "lovable rogue," "love-to-hate villain," "underdog protagonist," "femme fatale," "anti-hero," "reluctant hero."
**Mechanism:** KW for archetype ConceptTags where they exist (ANTI_HERO, FEMALE_LEAD, etc.) + P-EVT prose describing character types + NRT characterization_methods for how the archetype is portrayed.
**Why fan out:** no single channel carries character-archetype signal cleanly. Canonical tags win when they match; prose catches everything else; NRT carries the portrayal style.
**Why split from Cat 6 proper:** subject *presence* is about whether a thing/subject is in the story; character *archetype* is about the character's *type*, which NRT characterization is designed for. NRT in the fan-out is the distinguishing endpoint.
**Why split from Cat 20 (kind of story):** character arc = trajectory (P-ANA.character_arcs); character archetype = static type. Different endpoints (NRT vs ANC).

### 7. Award records
**Endpoints:** AWD.
**Handles:** formal wins and nominations; ceremony-specific filters; multi-win superlatives.
**Mechanism:** `movie_awards` filter with COUNT thresholds; fast path on `award_ceremony_win_ids`.
**Why single-endpoint:** structured ceremony/outcome data lives only in AWD. Quality-superlative queries that happen to anchor on awards compose this with Cat 26.

### 8. Trending
**Endpoints:** TRENDING.
**Handles:** "right now," "trending," "what's everyone watching."
**Mechanism:** live-refreshed trending signal.
**Why single-endpoint and why separate from Cat 9:** TRENDING is the only channel with a refresh cadence; META.popularity_score is static ingest-time and misses "right now" semantics entirely.

### 9. Structured single-attribute
**Endpoints:** META.
**Handles:** release date / era, runtime, maturity rating (when asked alone, not as a content-sensitivity axis), audio language, streaming platform, budget scale, box office bucket, numeric reception score (when asked alone, not as part of a reception-quality question), production country *as legal/financial origin*.
**Mechanism:** direct movie_card column predicate with scoring-appropriate shape (decay, tanh, GIN, packed-uint32 decode).
**Why single-endpoint:** each question lives on an authoritative closed-schema column; fan-out would add noise, not recall.

### 10. Top-level genre
**Endpoints:** META + KW + P-ANA.
**Handles:** broad genre buckets (horror, action, comedy, sci-fi, drama, romance, animation).
**Mechanism:** META.genre_ids GIN overlap (closed enum, coarse) + KW genre-family keywords (broader vocabulary, medium grain) + P-ANA genre_signatures (prose-level genre feel).
**Why split from Cat 9:** genre is the one "structured" attribute where fan-out genuinely helps. META's genre_ids is closed and often too coarse (is "cozy mystery" drama, comedy, or crime?); KW catches genre-flavor keywords that never made it into the META enum; P-ANA catches prose-level genre nuance. Alone, each channel undershoots.

### 11. Cultural tradition / national cinema
**Endpoints:** META + KW.
**Handles:** "Korean cinema," "Bollywood," "Hong Kong action," "Italian neorealism," "French New Wave."
**Mechanism:** META.country_of_origin + META.audio_language_ids (the literal production-country and language columns) + KW (cultural-tradition keywords — e.g. HINDI-LANGUAGE resolving to Bollywood, or a BOLLYWOOD tag if present).
**Why fan out:** country-of-origin alone is wrong for traditions (Hollywood-funded HK-action isn't HK by production country). KW carries the tradition vocabulary. Language often carries the tradition signal when country doesn't.
**Why split from Cat 9:** the country-of-origin field is treated here as one signal among several, not as the authoritative answer. Different endpoint set.

### 12. Filming location
**Endpoints:** PRD + META (country weak signal).
**Handles:** "filmed in New Zealand," "shot on location in Iceland," "Morocco shoots."
**Mechanism:** PRD filming_locations field (primary) + META country_of_origin as weak corroborator (production country often tracks filming location).
**Why fan out:** PRD prose is the only channel with actual shooting-location data; country-of-origin is legal, not literal. But when PRD returns sparse results, country-of-origin raises a weak prior.
**Why split from Cat 11 and Cat 9:** filming location ≠ cultural tradition ≠ production country legally. Three distinct questions, three distinct fan-outs.

### 13. Format + visual-format specifics
**Endpoints:** KW + PRD.
**Handles:** format (documentary, short, anime, mockumentary), visual-format specifics (B&W, 70mm, found-footage, widescreen, handheld).
**Mechanism:** KW for canonical tags (DOCUMENTARY, BLACK_AND_WHITE, FOUND_FOOTAGE, ANIMATION); PRD for prose-level technique descriptions ("shot on 16mm," "single-take long shot," "handheld cinematography").
**Why fan out:** canonical-tag matches give precision when the format maps to a vocabulary member; PRD catches the long tail that has no tag.

### 14. Sub-genre + story archetype
**Endpoints:** KW + P-ANA.
**Handles:** sub-genre ("body horror," "cozy mystery," "space opera," "neo-noir"), story archetype ("revenge," "underdog," "post-apocalyptic," "heist").
**Mechanism:** KW for curated sub-genre and archetype tags (fine-grained but discrete); P-ANA (genre_signatures, conflict_type, character_arcs) for prose nuance.
**Why fan out:** discrete tags win on precision; prose catches the fuzzy boundaries between sub-genres and the archetype framings that don't have tags.

### 15. Narrative devices + structural form + how-told craft
**Endpoints:** KW + NRT + VWX.
**Handles:** plot twist, nonlinear timeline, unreliable narrator, single-location, anthology, ensemble, two-hander, POV mechanics, character-vs-plot focus, pacing-at-craft-level (slow burn, frenetic), "Sorkin-style" dialogue as craft pattern.
**Mechanism:** KW for canonical device and structural tags when they exist; NRT for the full craft surface (narrative_archetype, delivery, POV, characterization, information control, conflict-stakes); VWX for the pacing *feel* side (slow burn as experience, not structure).
**Why fan out:** craft questions blur into feel (slow burn is both how-told and experienced). NRT is the craft primary; KW catches what's been promoted to vocabulary; VWX catches the experiential side when the user's framing is feel-based.
**Why a single category (not three):** all three endpoint sets genuinely compose for craft questions — splitting would create artificial walls between "devices" (KW+NRT) and "how-told pacing" (NRT+VWX) that real queries cross constantly.

### 16. Target audience
**Endpoints:** KW + META + CTX.
**Handles:** "family movies," "teen movies," "kids movie," "for adults," "something to watch with the grandparents." Specifically the *audience being pitched to*, not the story archetype — coming-of-age is a story archetype and routes to Cat 20 instead.
**Mechanism:** KW audience-framing tags + META.maturity_rank as gate + CTX watch_scenarios ("watch with kids," "family night").
**Why fan out:** KW+META alone catches the framing tag and maturity gate but misses context-heavy framings ("watch with the kids") that live in CTX.
**Why no P-ANA:** with coming-of-age moved to Cat 20, there's no arc-trajectory overlap left.

### 17. Sensitive content
**Endpoints:** KW + META + VWX.
**Handles:** "no gore," "not too bloody," "with nudity," "violent but not graphic."
**Mechanism:** KW content tags (canonical presence/absence flags) + META.maturity_rank (broad gate) + VWX.disturbance_profile (intensity gradient).
**Why fan out:** presence is binary (KW); acceptable-intensity is gradient (VWX); maturity is the blunt gate. All three contribute.
**Why split from Cat 16:** different endpoint set (VWX is in the fan-out here, not there).

### 18. Seasonal / holiday
**Endpoints:** KW (proxy chains) + CTX + P-EVT.
**Handles:** Christmas, Halloween, Thanksgiving, summer-blockbuster.
**Mechanism:** KW via **proxy chains** (the vocabulary has no dedicated seasonal tags, so the LLM rewrites seasonal intent into proxy tags at query-generation time: Halloween → horror + supernatural + spooky + slasher; Christmas → family + heartwarming + snowed-in + winter) + CTX watch_scenarios (seasonal viewing context embeds here at ingest: "Christmas viewing," "Halloween movie night") + P-EVT for narrative setting ("set on Christmas Eve," "Halloween night").
**Why fan out:** no channel carries seasonal framing directly. Proxy-KW gives us the closest-thing-to-tags recall; CTX catches seasonal context from review/description language; P-EVT catches seasonal narrative settings.

---

## Semantic-driven categories

### 19. Plot events + narrative setting
**Endpoints:** P-EVT.
**Handles:** literal plot events ("heist that unravels when a member betrays the crew"), narrative time setting ("set in 1940s Berlin"), narrative place setting ("takes place in Tokyo").
**Mechanism:** Qdrant cosine against plot_events space; dense prose input.
**Why single-endpoint:** plot_events is the only space that receives raw synopsis text. Narrative setting time/place have no structured column, so P-EVT prose is the only signal. Other spaces are deliberately plot-free and would undershoot.

### 20. Kind of story / thematic archetype
**Endpoints:** P-ANA + ANC + KW.
**Handles:** "movies about grief," "redemption arcs," "man-vs-nature," "coming-of-age about self-acceptance."
**Mechanism:** P-ANA (primary — elevator_pitch, conflict_type, thematic_concepts, character_arcs) + ANC for vibe-framed variants + KW thematic ConceptTags for queries that map to curated theme tags.
**Why fan out:** P-ANA is thematic-archetype primary; ANC catches "kind of story" when the framing spans multiple axes (vibe); KW catches queries that map cleanly to a theme tag (FOUND_FAMILY, CORRUPTION, REDEMPTION).

### 21. Viewer experience / feel / tone / cognitive demand
**Endpoints:** VWX + ANC.
**Handles:** feel to watch, tonal aesthetic (dark, whimsical, gritty), cognitive demand (mindless vs cerebral), realism/stylization mode, tension/disturbance intensity.
**Mechanism:** VWX primary (emotional_palette, tension, disturbance, sensory_load, cognitive_complexity, tone_self_seriousness) + ANC for multi-axis vibe matches.
**Why fan out:** VWX is experiential-primary; ANC catches queries where feel blends with identity (e.g. "cozy Sunday vibe" is both feel and identity-level).

### 22. Occasion / self-experience goal / comfort-watch / gateway
**Endpoints:** CTX + RCP + VWX + KW.
**Handles:**
- Viewing occasion (date night, background, rainy Sunday).
- Self-experience goal ("make me cry," "cheer me up," "challenge me," "something mindless").
- Comfort-watch archetype ("go-to movie," "feel-better movie").
- Gateway / entry-level ("good first anime," "accessible arthouse").

**Mechanism:** CTX primary (watch_scenarios + self_experience_motivations are literally named for these framings) + VWX for the emotional-target side ("make me cry" touches VWX.emotional_palette directly) + RCP for reviewer commentary (tearjerker, comfort-rewatch, accessibility) + KW where the vocabulary has tags (TEARJERKER, FEEL_GOOD if present).
**Why fan out:** "make me cry" is the clearest multi-endpoint case — CTX holds the motivation, VWX holds the emotional match, RCP holds the "tearjerker" evaluation, and KW holds any canonical tag.
**Gap absorbed:** gateway/entry-level.

### 23. Craft acclaim (visual / music / dialogue)
**Endpoints:** RCP + PRD + NRT.
**Handles:** "visually stunning," "killer cinematography," "iconic score," "great soundtrack," "quotable dialogue," "technical marvel," "beautifully shot," "naturalistic dialogue," "memorable theme."
**Mechanism:** RCP is the primary across all three craft axes (praised_qualities extraction names the axis); PRD contributes for visual/technical when the praise is about production craft ("IMAX-shot," "practical effects"); NRT contributes for dialogue-as-craft-pattern ("Sorkin-style") and other narrative-craft-flavored acclaim. Semantic endpoint internally routes to the relevant vectors based on the query's craft axis.
**Fallthrough:** if the user names a specific creator (composer, cinematographer), that's Cat 1.
**Why one category:** all three craft-acclaim flavors route purely to semantic and conceptually cover the same thing ("the movie is praised for a specific craft axis"). Per-axis fan-out variance is internal to semantic dispatch, not a category-split signal.

### 24. Reception qualitative + quality superlative
**Endpoints:** RCP + META + AWD.
**Handles:** cult / acclaimed / underrated / divisive / overhyped; thematic weight on the acclaim side ("has something to say"); cultural influence ("era-defining"); still-holds-up; cast popularity ("stacked A-list cast"); quality superlatives ("best horror of the 80s," "scariest movie ever," "funniest").
**Mechanism:** RCP primary (reception_summary, praised/criticized tags) + META.reception_score as global-quality baseline + AWD when the superlative is award-anchored. Axis-of-superlative ("of the 80s," "horror") composes in Cat 9 and/or Cat 14.
**Why fan out:** RCP carries the qualitative-reception prose; META anchors to a numeric baseline; AWD contributes when the query hinges on formal recognition. "Best X" queries always involve at least two of the three.
**Gaps absorbed:** cultural influence, still-holds-up, thematic weight (acclaim side), cast popularity, below-the-line creator acclaim (via Cat 27 fallthrough).

### 25. Post-viewing resonance
**Endpoints:** VWX + RCP + KW.
**Handles:** "stays with you," "haunting," "can't stop thinking about it," "gut-punch ending," "forgettable," "happy ending," "twist ending," "ambiguous ending," "downer ending."
**Mechanism:** VWX.ending_aftertaste (first-impression aftermath) + RCP (lingering critical/audience resonance) + KW ending-type tags (TWIST_ENDING, DOWNER_ENDING, HAPPY_ENDING, AMBIGUOUS_ENDING where present).
**Why fan out:** first-viewing aftertaste is experiential (VWX); lasting resonance is reception-side (RCP); ending *type* has canonical tags (KW) that give precise retrieval for structural ending questions.
**Why split from Cat 21:** different endpoint set (RCP + KW added) and flavor (post-viewing, not during-viewing).

### 26. Scale / scope / holistic vibe
**Endpoints:** ANC + P-ANA.
**Handles:** scale ("epic," "intimate," "sprawling," "small and personal"), multi-axis vibe queries, "movies that feel like X" when X is more vibe than specific.
**Mechanism:** ANC (identity-level capsule is the natural home for scope and multi-axis vibe) + P-ANA for thematic-archetype corroboration.
**Why fan out:** ANC is the only plot-light identity space; P-ANA adds thematic context when the vibe has a definable story-kind behind it.

---

## Trick cases (specific enough to handle explicitly rather than fall to the catch-all)

These three categories all route to semantic endpoints but are broken out from Cat 30 (Interpretation-required) because they're specific enough that an unguided LLM would otherwise scatter them across wrong categories. Listing them explicitly gives dispatch a stable hook.

### 27. Curated canon / named list membership
**Endpoints:** RCP + ANC + META.reception_score.
**Handles:** Criterion Collection, AFI Top 100, Sight & Sound greatest-films lists, IMDb Top 250, BFI, National Film Registry, "1001 Movies to See Before You Die," film-school canon.
**Mechanism:** semantic search over RCP (reviews reference list membership: "a Criterion pick," "AFI-honored") and ANC (identity-level pitch often reflects canonical stature). META.reception_score as a weak floor (list-member films are overwhelmingly well-received).
**Why a distinct category (not Cat 30 fallback):** named lists are stable, recognizable referents. Dispatch benefits from knowing "this is a named-list query" to prompt the semantic LLM to search for list-citation language specifically, rather than generic interpretation.

### 28. Below-the-line creator lookup
**Endpoints:** RCP + ANC.
**Handles:** cinematographer, editor, production designer, costume designer, visual-effects supervisor queries — "Roger Deakins movies," "Thelma Schoonmaker-edited," "Sandy Powell costumes," "Colleen Atwood designs."
**Mechanism:** semantic fallback. Reception text routinely names these creators when their work was noted ("Deakins' cinematography," "edited by Schoonmaker"); ANC identity prose sometimes encodes it too.
**Why a distinct category (not Cat 1 fallthrough):** Cat 1 is posting-table-backed and fails silently when the role isn't indexed. Lifting this out makes the non-indexed roles visible as a deliberate pattern, not an afterthought.

### 29. Source-material author / origin-work creator
**Endpoints:** KW + P-EVT + RCP.
**Handles:** "Stephen King adaptations," "Jane Austen movies," "Tolkien films," "Philip K. Dick stories," "Neil Gaiman works."
**Mechanism:** KW (SourceMaterialType narrows to the relevant adaptation class — novel / short-story / comic) + P-EVT (plot prose usually names the source author: "based on Stephen King's novel") + RCP (reviews cite the author).
**Why a distinct category:** Cat 1 is film credits — source-material authors aren't film credits. Cat 5 is adaptation flag — can't name the author. Cat 6 is subject-depicted — these movies aren't *about* the author. Without explicit routing, the LLM scatters across the wrong three and underdelivers.

---

## Fallback

### 30. Interpretation-required (hard-to-fit)
**Endpoints:** INTERP → routes to {RCP, ANC, CTX}.
**Handles:** any query the LLM recognizes as real but that doesn't map cleanly to Cats 1–29. No preset primary members — Cats 27, 28, 29 absorbed the previously-known members.
**Mechanism:** the LLM receives an **explicit interpretation prompt**:
> This query doesn't fit a structured channel. Identify the user's
> underlying intent and what signal that intent would incidentally
> produce in reviewer / viewer / identity text. Construct a best-
> effort semantic query against whichever of {reception, watch_context,
> anchor} is most likely to carry that signal. Lean broad rather than
> narrow, and flag uncertainty.

Output is a standard semantic query dispatched to RCP / CTX / ANC.
**Why a distinct category:** a silent fallback would be invisible —
we'd have no way to meter fire rate, no hook for the interpretation-
guidance prompt, and no way to surface appropriate uncertainty in
the response. Making this an explicit category lets dispatch code
behave differently (prompt injection + confidence-lowered phrasing).

---

## Where every data gap lands

| Gap | Category | How handled |
|---|---|---|
| Below-the-line credits (cinematographer, editor, etc.) | Cat 28 | Dedicated category — RCP + ANC |
| Curated canon (Criterion, AFI, etc.) | Cat 27 | Dedicated category — RCP + ANC + META.reception_score |
| Source-material author (Stephen King adaptations, etc.) | Cat 29 | Dedicated category — KW + P-EVT + RCP |
| Gateway / entry-level | Cat 22 | CTX + RCP fan-out |
| Cultural tradition | Cat 11 | META + KW fan-out |
| Seasonal | Cat 18 | KW proxy chains + CTX + P-EVT |
| Narrative setting time/place | Cat 19 | P-EVT prose |
| Cast popularity ("stacked cast") | Cat 24 | RCP prose |
| Thematic weight ("has something to say") | Cat 24 (acclaim framing) + Cat 26 (vibe framing) | Framing-dependent routing |
| Character-vs-plot focus | Cat 15 | NRT primary |
| Character archetype ("lovable rogue") | Cat 6b | KW + P-EVT + NRT |
| Plot element / motif (clowns, zombies, sharks) | Cat 6 | KW + P-EVT |
| Self-experience goal ("make me cry") | Cat 22 | CTX + VWX + RCP + KW |
| Scale / scope | Cat 26 | ANC + P-ANA |
| Rewatch value | Cat 22 + Cat 24 | CTX for "comfort rewatch," RCP for "holds up on rewatch" |
| Cultural influence | Cat 24 | RCP prose |
| Still-holds-up | Cat 24 | RCP prose |
| Live trending | Cat 8 | Dedicated endpoint |

## Composition notes

Categories are composable atoms, not mutually exclusive buckets.
"Tom Hanks comedies from the 90s rated above 8" fires Cat 1 (actor)
+ Cat 10 (genre — drama/comedy) + Cat 9 (release date + numeric
reception). "Best horror of the 80s" fires Cat 24 (superlative) +
Cat 10 (horror) + Cat 9 (date). The dispatcher resolves each
category independently and merges scores.

## Compound split rule

**If a phrase or query seems to fit multiple categories, that is a
signal it should be split into separate atomic requirements.** The
upstream category-identification step is expected to decompose
compound phrases into their constituent category firings rather
than inventing an umbrella category to absorb the compound.

Compound descriptors never warrant their own category — the word
"classic" means older + canonical, and the correct handling is to
fire Cat 9 (release era) + Cat 24 (canonical / acclaimed)
simultaneously. Creating a "Canonical stature" category to hold
"classic" would just duplicate endpoints already covered by
Cats 9 and 24 while hiding the compound nature from dispatch.

Examples:
- "Classic Arnold Schwarzenegger action movies" → Cat 1
  (Schwarzenegger) + Cat 10 (action) + Cat 9 (older era) +
  Cat 24 (canonical stature).
- "Disney classics" → Cat 3 (Disney) + Cat 9 (older era) +
  Cat 24 (canonical).
- "Lone female protagonist" → Cat 6b (female-lead archetype) +
  Cat 15 (single-lead structural form).
- "Modern classic" → Cat 9 (recent era, narrower range) +
  Cat 24 (canonical stature).

The only time a compound stays bound to a single category is when
the category explicitly owns the compound — e.g. a named curated
list ("Criterion Collection") in Cat 27, which *is* the compound
of "canonical recognition + specific named list."

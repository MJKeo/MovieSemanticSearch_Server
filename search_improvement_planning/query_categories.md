# Query Categories (final)

Each category names a conceptual flavor of question AND the concrete
set of endpoints the dispatcher fans out to when that category fires.
Categories **compose**: a single user query routinely invokes several
(e.g. "Tom Hanks comedies from the 90s" hits Cat 1 + Cat 11 + Cat 10).
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
**Below-the-line fallthrough:** cinematographer / editor / production designer / costume designer aren't indexed as postings — those queries route to Cat 29 (Below-the-line creator lookup) instead.

### 2. Named character lookup
**Endpoints:** ENT.
**Handles:** named-character presence and prominence ("Batman movies," "any Wolverine appearance").
**Mechanism:** ENT character postings (with CENTRAL/DEFAULT prominence).
**Why single-endpoint (despite iconic characters being franchise anchors):** the step 2 pre-pass already decomposes queries like "Batman movies" into a Named-character atom AND a Franchise / universe-lineage atom as separate coverage_evidence entries, both routed to their respective categories (Cat 2 and Cat 4). Fanning out to FRA inside Cat 2 itself would double-count the franchise side for queries that mention a franchise-anchor character, while offering nothing for queries where no franchise is implied. Keeping Cat 2 pure to character postings lets the upstream decomposition do the franchise fan-out when it's genuinely asked for.
**Why split from Cat 1:** characters and person credits share the posting-table retrieval shape, but semantically they are different question-kinds (a persona in the story vs. a name in the credits) and the prominence-mode subsets differ (CENTRAL/DEFAULT for characters, LEAD/SUPPORTING/MINOR/DEFAULT for actors).

### 3. Studio / brand attribution
**Endpoints:** STU.
**Handles:** production-company and curated-brand queries.
**Mechanism:** ProductionBrand enum path (brand_id posting, with rename-chain time-bounding) OR freeform token intersect with DF ceiling.
**Why single-endpoint:** brand rename handling and DF-filtered token intersection are unique to STU; no other channel carries production-company signal with any discipline.

### 4. Franchise / universe lineage
**Endpoints:** FRA.
**Handles:** franchise/universe membership, lineage position (sequel/prequel/spinoff/crossover/launched), mainline-vs-offshoot, reboot/remake positioning.
**Mechanism:** FRA two-phase token resolution → array overlap on lineage / shared_universe / subgroup. Remake-as-lineage-positioning (e.g. "the original Scarface, not the remake") is handled by the lineage arrays themselves, not by a separate keyword channel.
**Why single-endpoint:** the step 2 pre-pass decomposes compound phrases like "James Bond remakes" into a franchise atom (Cat 4) and a separate source-flag atom (Cat 5). Cat 4 therefore never sees a bare-remake axis that needs its own keyword channel — that framing lands in Cat 5. Keeping Cat 4 pure to the franchise-lineage surface avoids double-counting the same signal across categories.

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
- Presence of the *thing* in the story, as distinct from Cat 2 (specific *named* character like Batman) and Cat 7 (character *archetype* like "lovable rogue").

**Mechanism (tiered, keyword-first):** if the subject or motif resolves cleanly to a canonical keyword tag (BIOGRAPHY, TRUE_STORY, ZOMBIE, SHARK, CLOWN, etc.), query KW only and stop. If no tag covers the request, or the request is spectrum-framed ("subtly zombie-themed," "loose allegorical biography"), fall back to Semantic (plot_events prose).

**Why tiered, not fan-out:** for binary "is this thing in the story?" questions a canonical tag is authoritative — Semantic adds noise (partial matches on prose that mentions the motif in passing) without meaningful precision gain. Semantic earns its place only when the vocabulary has no entry for the subject (long-tail) or when the framing isn't binary (spectrum). Running both by default would dilute precise tag hits with semantic near-misses.

**Why split from Cat 5:** source-flag (Cat 5) is a yes/no flag about origin medium; subject-presence (Cat 6) is about what's in the story. Different conceptual flavor even when both routes happen to use Keyword.

### 7. Character archetype
**Endpoints:** KW + Semantic (tiered, keyword-first).
**Handles:** "lovable rogue," "love-to-hate villain," "underdog protagonist," "femme fatale," "anti-hero," "reluctant hero."
**Mechanism:** if the archetype matches a canonical ConceptTag (ANTI_HERO, FEMALE_LEAD, etc.), query KW only and stop. If no tag covers the requested archetype, fall back to Semantic (narrative_techniques characterization_methods as primary, plot_events prose for archetypes described in the synopsis).
**Why tiered, not fan-out:** same logic as Cat 6 — canonical archetype tags are authoritative when present, and running Semantic alongside a clean tag hit only dilutes precision. Semantic is reserved for archetypes the vocabulary doesn't cover.
**Why split from Cat 6 proper:** subject *presence* is about whether a thing/subject is in the story; character *archetype* is about the character's *type*, which narrative_techniques characterization is designed for.
**Why split from Cat 21 (kind of story):** character arc = trajectory (plot_analysis.character_arcs); character archetype = static type. Different fallback vector space when the tiered fallback fires.

### 8. Award records
**Endpoints:** AWD.
**Handles:** formal wins and nominations; ceremony-specific filters; multi-win superlatives.
**Mechanism:** `movie_awards` filter with COUNT thresholds; fast path on `award_ceremony_win_ids`.
**Why single-endpoint:** structured ceremony/outcome data lives only in AWD. Quality-superlative queries that happen to anchor on awards compose this with Cat 25.

### 9. Trending
**Endpoints:** TRENDING.
**Handles:** "right now," "trending," "what's everyone watching."
**Mechanism:** live-refreshed trending signal.
**Why single-endpoint and why separate from Cat 10:** TRENDING is the only channel with a refresh cadence; META.popularity_score is static ingest-time and misses "right now" semantics entirely.

### 10. Structured single-attribute
**Endpoints:** META.
**Handles:** release date / era, runtime, maturity rating (when asked alone, not as a content-sensitivity axis), audio language, streaming platform, budget scale, box office bucket, numeric reception score (when asked alone, not as part of a reception-quality question), production country *as legal/financial origin*.
**Mechanism:** direct movie_card column predicate with scoring-appropriate shape (decay, tanh, GIN, packed-uint32 decode).
**Why single-endpoint:** each question lives on an authoritative closed-schema column; fan-out would add noise, not recall.

### 11. Top-level genre
**Endpoints:** KW + P-ANA (mutually exclusive per query).
**Handles:** broad genre buckets (horror, action, comedy, sci-fi, drama, romance, animation).
**Mechanism:** KW genre-family keywords when the query names a canonical genre that resolves cleanly to a keyword member; P-ANA genre_signatures (prose-level genre feel) when the query adds a semantic qualifier the vocabulary can't express ("dark action," "quiet drama").
**Why mutually exclusive, not fan-out:** the keyword vocabulary already covers every genre in META.genre_ids and more, so META contributes no additional recall — it's strictly redundant. KW is authoritative when the genre exists as a tag; P-ANA is the fallback for compound / qualifier-laden genre framings that don't map to a single tag.
**Why split from Cat 10:** genre is the one conceptual "genre" question where keyword fan-out is richer than the closed META enum, and the compound cases need semantic prose — neither belongs in the Cat 10 single-column-predicate shape.

### 12. Cultural tradition / national cinema
**Endpoints:** KW + META (mutually exclusive per query, keyword-first).
**Handles:** "Korean cinema," "Bollywood," "Hong Kong action," "Italian neorealism," "French New Wave."
**Mechanism:** first check the keyword vocabulary for a cultural-tradition tag that matches (BOLLYWOOD, KOREAN_CINEMA, ITALIAN_NEOREALISM, etc.). If a tag exists, use it — it's the authoritative signal for the tradition as an aesthetic. If no tag covers the requested tradition, fall back to META.country_of_origin (and META.audio_language_ids where the tradition implies a language) against the plausible top countries for that tradition.
**Why mutually exclusive, not fan-out:** the two signals answer different questions. KW tags tradition as an aesthetic; META.country_of_origin tags legal/financial origin. If a tradition tag exists, the country column is worse than useless — Hollywood-funded HK action isn't HK by production country. Only when no tag exists does country-of-origin become the best remaining proxy, and at that point the keyword channel has nothing to contribute.
**Why split from Cat 10:** Cat 10 treats country as the authoritative answer to a "where was it produced?" question. Here country is only a fallback for an aesthetic-tradition question. Different conceptual flavor, different routing rule.

### 13. Filming location
**Endpoints:** Semantic.
**Handles:** "filmed in New Zealand," "shot on location in Iceland," "Morocco shoots."
**Mechanism:** Semantic (production_techniques) similarity against the filming_locations prose field — the only channel carrying actual shooting-location data.
**Why single-endpoint:** META.country_of_origin is the wrong column. It records legal/financial production country, not filming geography. Hollywood-funded films shot abroad (The Revenant in Canada/Argentina, Dune in Jordan and UAE, Mission Impossible – Fallout across Kashmir / UAE / NZ) all carry US country_of_origin and would be invisible to any metadata branch. Rather than lean on a column whose semantics don't match the question, query the one channel that actually tracks filming location.
**Why split from Cat 12 and Cat 10:** filming location ≠ cultural tradition ≠ legal production country. Three distinct questions that each route to a different primary channel.

### 14. Format + visual-format specifics
**Endpoints:** KW + Semantic (tiered, keyword-first).
**Handles:** format (documentary, short, anime, mockumentary), visual-format specifics (B&W, 70mm, found-footage, widescreen, handheld).
**Mechanism:** if the format or visual-format spec matches a canonical tag (DOCUMENTARY, BLACK_AND_WHITE, FOUND_FOOTAGE, ANIMATION, etc.), query KW only and stop. If no tag exists, fall back to Semantic (production_techniques) against prose-level technique descriptions ("shot on 16mm," "single-take long shot," "handheld cinematography").
**Why tiered, not fan-out:** format tags are authoritative flags — if a movie is tagged DOCUMENTARY, it's a documentary; running Semantic alongside only adds prose near-misses. Semantic is the long-tail catch for technique-level specs that the vocabulary never absorbed.

### 15. Sub-genre + story archetype
**Endpoints:** KW + Semantic (tiered, keyword-first).
**Handles:** sub-genre ("body horror," "cozy mystery," "space opera," "neo-noir"), story archetype ("revenge," "underdog," "post-apocalyptic," "heist").
**Mechanism:** if the sub-genre or archetype matches a curated tag, query KW only and stop. If no tag covers the requested framing, fall back to Semantic (plot_analysis — genre_signatures, conflict_type, character_arcs) for prose nuance.
**Why tiered, not fan-out:** tags are precise when they exist; Semantic is the safety net for the fuzzy long tail ("cozy mystery" has no tag today even though "mystery" does). Running both by default would mix clean tag hits with semantic noise.

### 16. Narrative devices + structural form + how-told craft
**Endpoints:** KW + Semantic (tiered, keyword-first).
**Handles:** plot twist, nonlinear timeline, unreliable narrator, single-location, anthology, ensemble, two-hander, POV mechanics, character-vs-plot focus, pacing-at-craft-level (slow burn, frenetic), "Sorkin-style" dialogue as craft pattern.
**Mechanism:** if the device or structural form matches a canonical tag (PLOT_TWIST, NONLINEAR_TIMELINE, UNRELIABLE_NARRATOR, SINGLE_LOCATION, ENSEMBLE_CAST, etc.), query KW only and stop. If no tag covers the request, fall back to Semantic — primarily narrative_techniques (narrative_archetype, delivery, POV, characterization, information control, conflict-stakes), with viewer_experience picking up the pacing-feel side when the framing is experiential ("slow burn").
**Why tiered, not fan-out:** canonical device tags are definitive — a movie either has a plot twist in its vocabulary profile or it doesn't. Semantic earns its place only for craft questions without a tagged device name ("POV mechanics," "character-vs-plot focus," "Sorkin-style dialogue as a craft pattern"). Running both always would let semantic near-misses leak into clean tag hits.
**Why a single category (not three):** device, structural form, and how-told craft share the same tiered routing — the question-kind is "how is this told?" regardless of which sub-axis the user framed it on.

### 17. Target audience
**Endpoints:** KW + META + Semantic (gate + inclusion, query-dependent).
**Handles:** "family movies," "teen movies," "kids movie," "for adults," "something to watch with the grandparents." Specifically the *audience being pitched to*, not the story archetype — coming-of-age is a story archetype and routes to Cat 21 instead.
**Mechanism:** META.maturity_rank is a hard gate when the audience framing implies a maturity ceiling (family / kids / PG-13 max). Within the gated pool, KW audience-framing tags and Semantic (watch_context watch_scenarios like "watch with kids," "family night") contribute additive inclusion scoring.
**Why gate + inclusion and not all-fire-always:** the three endpoints serve different roles — META is a filter, KW and Semantic are scorers — and which of KW / Semantic actually fires is query-dependent. A bare "family movie" may only need the maturity gate plus the KW framing tag; "something to watch with my kids on a Saturday night" invokes Semantic watch_context as well. Not every endpoint is always active: the category fans out only the subset the query asks for.
**Why split from Cat 18:** Cat 18 is about content (gore, nudity) on its own spectrum; Cat 17 is about packaged audience framing. Different fallback spaces when semantic fires.

### 18. Sensitive content
**Endpoints:** KW + META + Semantic (gate + inclusion, query-dependent).
**Handles:** "no gore," "not too bloody," "with nudity," "violent but not graphic."
**Mechanism:** META.maturity_rank is the blunt gate when the query implies a rating ceiling. Within the gated pool, KW content tags score binary presence / absence (ANIMAL_DEATH-style flags), and Semantic (viewer_experience disturbance_profile) scores intensity gradient for spectrum-framed asks ("not too bloody," "violent but not graphic").
**Why gate + inclusion and not all-fire-always:** the three endpoints carry different question shapes and which one fires depends on how the user framed the ask. "No gore" is pure KW (binary exclusion). "Not too bloody" is pure Semantic (gradient). "Family-friendly" brings in META maturity. Not every endpoint is always active; the category fans out only the subset the query needs.
**Why split from Cat 17:** Cat 18 scores content on its own intensity spectrum; Cat 17 is about packaged audience framing. The Semantic fallback spaces differ (viewer_experience here vs. watch_context there).

### 19. Seasonal / holiday
**Endpoints:** KW + Semantic (additive combo).
**Handles:** Christmas, Halloween, Thanksgiving, summer-blockbuster.
**Mechanism:** both fire together. KW via **proxy chains** — the vocabulary has no dedicated seasonal tags, so the LLM rewrites seasonal intent into proxy tags at query-generation time (Halloween → horror + supernatural + spooky + slasher; Christmas → family + heartwarming + snowed-in + winter). Semantic contributes two spaces: watch_context for seasonal viewing framing ("Christmas viewing," "Halloween movie night") and plot_events for seasonal narrative settings ("set on Christmas Eve," "Halloween night"). Scores merge additively — a movie hit by proxy tags and by seasonal watch_context and by narrative setting ranks highest.
**Why additive, not tiered:** no channel is authoritative. The proxy-tag approach is inherently approximate (Halloween ≠ horror exactly), and the semantic spaces catch real signal the proxy chain misses — especially for less-canonicalized holidays. Running both is the only way to reach acceptable recall on a category with no dedicated vocabulary.

---

## Semantic-driven categories

### 20. Plot events + narrative setting
**Endpoints:** P-EVT.
**Handles:** literal plot events ("heist that unravels when a member betrays the crew"), narrative time setting ("set in 1940s Berlin"), narrative place setting ("takes place in Tokyo").
**Mechanism:** Qdrant cosine against plot_events space; dense prose input.
**Why single-endpoint:** plot_events is the only space that receives raw synopsis text. Narrative setting time/place have no structured column, so P-EVT prose is the only signal. Other spaces are deliberately plot-free and would undershoot.

### 21. Kind of story / thematic archetype
**Endpoints:** KW + Semantic (tiered, keyword-first for binary framings; semantic-only for spectrum framings).
**Handles:** "movies about grief," "redemption arcs," "man-vs-nature," "coming-of-age about self-acceptance."
**Mechanism:** if the ask is binary — "must be about this theme" — and the theme matches a curated ConceptTag (FOUND_FAMILY, CORRUPTION, REDEMPTION), query KW only and stop. If no tag matches the binary request, fall back to Semantic (plot_analysis primary — elevator_pitch, conflict_type, thematic_concepts, character_arcs; anchor for vibe-framed variants). If the framing is inherently spectrum-y ("kind of about grief," "leans redemptive"), skip KW entirely and go straight to Semantic.
**Why tiered with a spectrum escape hatch:** a canonical theme tag is authoritative for binary "about this" questions; Semantic is the right fallback for themes without a tag. But theme questions frequently arrive with fuzzy framing that no binary tag captures correctly, so the LLM is allowed to skip the tag tier when the ask is a gradient rather than a commitment.

### 22. Viewer experience / feel / tone / cognitive demand
**Endpoints:** VWX + ANC.
**Handles:** feel to watch, tonal aesthetic (dark, whimsical, gritty), cognitive demand (mindless vs cerebral), realism/stylization mode, tension/disturbance intensity.
**Mechanism:** VWX primary (emotional_palette, tension, disturbance, sensory_load, cognitive_complexity, tone_self_seriousness) + ANC for multi-axis vibe matches.
**Why fan out:** VWX is experiential-primary; ANC catches queries where feel blends with identity (e.g. "cozy Sunday vibe" is both feel and identity-level).

### 23. Occasion / self-experience goal / comfort-watch / gateway
**Endpoints:** Semantic + KW (additive combo).
**Handles:**
- Viewing occasion (date night, background, rainy Sunday).
- Self-experience goal ("make me cry," "cheer me up," "challenge me," "something mindless").
- Comfort-watch archetype ("go-to movie," "feel-better movie").
- Gateway / entry-level ("good first anime," "accessible arthouse").

**Mechanism:** all applicable channels fire together and merge additively. Semantic contributes three spaces — watch_context primary (watch_scenarios + self_experience_motivations are literally named for these framings), viewer_experience for the emotional-target side ("make me cry" touches emotional_palette directly), and reception for reviewer commentary ("tearjerker," "comfort-rewatch," "accessible arthouse"). KW contributes canonical tags where present (TEARJERKER, FEEL_GOOD).
**Why additive combo, not tiered:** "make me cry" is the clearest multi-endpoint case — watch_context holds the motivation, viewer_experience holds the emotional match, reception holds the critical "tearjerker" evaluation, and a KW tag adds precision when available. Each channel carries distinct signal; none is authoritative alone.
**Gap absorbed:** gateway / entry-level.

### 24. Craft acclaim (visual / music / dialogue)
**Endpoints:** RCP + PRD + NRT.
**Handles:** "visually stunning," "killer cinematography," "iconic score," "great soundtrack," "quotable dialogue," "technical marvel," "beautifully shot," "naturalistic dialogue," "memorable theme."
**Mechanism:** RCP is the primary across all three craft axes (praised_qualities extraction names the axis); PRD contributes for visual/technical when the praise is about production craft ("IMAX-shot," "practical effects"); NRT contributes for dialogue-as-craft-pattern ("Sorkin-style") and other narrative-craft-flavored acclaim. Semantic endpoint internally routes to the relevant vectors based on the query's craft axis.
**Fallthrough:** if the user names a specific creator (composer, cinematographer), that's Cat 1.
**Why one category:** all three craft-acclaim flavors route purely to semantic and conceptually cover the same thing ("the movie is praised for a specific craft axis"). Per-axis fan-out variance is internal to semantic dispatch, not a category-split signal.

### 25. Reception qualitative + quality superlative
**Endpoints:** RCP + META.
**Handles:** cult / acclaimed / underrated / divisive / overhyped; thematic weight on the acclaim side ("has something to say"); cultural influence ("era-defining"); still-holds-up; cast popularity ("stacked A-list cast"); quality superlatives ("best horror of the 80s," "scariest movie ever," "funniest").
**Mechanism:** RCP primary (reception_summary, praised/criticized tags) + META.reception_score as global-quality baseline. Axis-of-superlative ("of the 80s," "horror") composes in Cat 10 and/or Cat 15.
**Why fan out:** RCP carries the qualitative-reception prose; META anchors to a numeric baseline. Both genuinely compose: "cult" is reception-prose heavy but still wants a reception-score prior, and "best X" needs the numeric baseline plus whatever reception language names the axis.
**Why no AWD here:** if a query mentions formal recognition ("Oscar-winning," "BAFTA-nominated"), the compound split rule decomposes it into Cat 8 alongside Cat 25 rather than folding AWD into Cat 25's default fan-out. Leaving AWD in Cat 25 would duplicate Cat 8 and blur this category's identity (reception prose + numeric baseline, not award records).
**Gaps absorbed:** cultural influence, still-holds-up, thematic weight (acclaim side), cast popularity, below-the-line creator acclaim (via Cat 29 fallthrough).

### 26. Post-viewing resonance
**Endpoints:** KW + Semantic (tiered with short-circuit, additive combo otherwise).
**Handles:** "stays with you," "haunting," "can't stop thinking about it," "gut-punch ending," "forgettable," "happy ending," "twist ending," "ambiguous ending," "downer ending."
**Mechanism:** if the request maps perfectly onto a canonical ending-type tag — "happy ending" → HAPPY_ENDING, "twist ending" → TWIST_ENDING, "downer ending" → SAD_ENDING, "ambiguous ending" → OPEN_ENDING — query KW only and stop. The tag is definitive for these structural ending questions. If the request is more experiential ("stays with you," "haunting," "gut-punch") or has no matching tag, fire both Semantic (viewer_experience.ending_aftertaste for first-viewing resonance + reception for lasting critical/audience resonance) and any partially-applicable KW tags, merging additively.
**Why tiered short-circuit plus additive fallback:** ending-type tags are structurally authoritative — no Semantic near-miss should dilute a clean HAPPY_ENDING hit. But "haunting" is inherently experiential and needs Semantic; no single space carries post-viewing resonance alone, so when Semantic fires it fires across multiple spaces at once.
**Why split from Cat 22:** Cat 22 is during-viewing feel; Cat 26 is post-viewing aftertaste. Different vector space (ending_aftertaste vs. the full viewer_experience surface) and different reception involvement.

### 27. Scale / scope / holistic vibe
**Endpoints:** ANC + P-ANA.
**Handles:** scale ("epic," "intimate," "sprawling," "small and personal"), multi-axis vibe queries, "movies that feel like X" when X is more vibe than specific.
**Mechanism:** ANC (identity-level capsule is the natural home for scope and multi-axis vibe) + P-ANA for thematic-archetype corroboration.
**Why fan out:** ANC is the only plot-light identity space; P-ANA adds thematic context when the vibe has a definable story-kind behind it.

---

## Trick cases (specific enough to handle explicitly rather than fall to the catch-all)

These three categories all route to semantic endpoints but are broken out from Cat 31 (Interpretation-required) because they're specific enough that an unguided LLM would otherwise scatter them across wrong categories. Listing them explicitly gives dispatch a stable hook.

### 28. Curated canon / named list membership
**Endpoints:** Semantic + META (additive combo, interpretation-driven).
**Handles:** Criterion Collection, AFI Top 100, Sight & Sound greatest-films lists, IMDb Top 250, BFI, National Film Registry, "1001 Movies to See Before You Die," film-school canon.
**Mechanism:** all three channels fire and merge additively, with the Semantic side interpretation-guided so the LLM decodes what a named list actually implies. Semantic over reception (reviews reference list membership: "a Criterion pick," "AFI-honored") and anchor (identity-level pitch often reflects canonical stature). META.reception_score contributes as a numeric prior because list-member films are overwhelmingly well-received, so the baseline quality score is an independently useful signal.
**Why additive-with-interpretation, not gate-plus-inclusion:** treating reception_score as a hard floor would reject legitimate list members that score lower than the cutoff (arthouse picks on Sight & Sound, for example). Treating it as an additive prior lets each endpoint contribute what it can — list-citation prose where reviewers name the list, canonical-stature framing where the identity prose reflects it, reception_score as a numeric lift — and interpretation-guided semantic handles the "what does this list *mean*" decoding (e.g. "IMDb Top 250" implies broadly-popular + highly-rated, "Sight & Sound" implies critic-canon + arthouse).
**Why a distinct category (not Cat 31 fallback):** named lists are stable, recognizable referents. Dispatch benefits from knowing "this is a named-list query" to prompt the semantic LLM to search for list-citation language specifically, rather than generic interpretation.

### 29. Below-the-line creator lookup
**Endpoints:** RCP + ANC.
**Handles:** cinematographer, editor, production designer, costume designer, visual-effects supervisor queries — "Roger Deakins movies," "Thelma Schoonmaker-edited," "Sandy Powell costumes," "Colleen Atwood designs."
**Mechanism:** semantic fallback. Reception text routinely names these creators when their work was noted ("Deakins' cinematography," "edited by Schoonmaker"); ANC identity prose sometimes encodes it too.
**Why a distinct category (not Cat 1 fallthrough):** Cat 1 is posting-table-backed and fails silently when the role isn't indexed. Lifting this out makes the non-indexed roles visible as a deliberate pattern, not an afterthought.

### 30. Source-material author / origin-work creator
**Endpoints:** Semantic.
**Handles:** "Stephen King adaptations," "Jane Austen movies," "Tolkien films," "Philip K. Dick stories," "Neil Gaiman works."
**Mechanism:** semantic search across the vectors where an author name would plausibly appear — plot_events (synopsis prose frequently names the author: "based on Stephen King's novel"), reception (reviews cite the author), and anchor if the identity capsule references them. No dedicated keyword channel inside this category.
**Why single-endpoint semantic:** step 2 decomposes "Stephen King adaptations" into the author atom (Cat 30) AND the adaptation-source atom (Cat 5) as separate coverage_evidence entries. Cat 5 carries SourceMaterialType.NOVEL_ADAPTATION and its own score term; adding the same KW filter inside Cat 30 would double-count what Cat 5 already contributes. With the adaptation-class filter handled upstream by Cat 5's score merge, Cat 30 only needs to answer one question — "which movies' prose actually names this author?" — and Semantic is the only channel where that name will appear.
**Why a distinct category:** Cat 1 is film credits — source-material authors aren't film credits. Cat 5 is adaptation flag — can't name the author. Cat 6 is subject-depicted — these movies aren't *about* the author. Without explicit routing, the LLM scatters across the wrong three and underdelivers.

---

## Fallback

### 31. Interpretation-required (hard-to-fit)
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

## Orchestration shapes

Every category runs under one of four shapes, which govern how
many endpoint queries can fire for that category on a given user
request. The shapes fall into two groups: shapes that cap the
category at a single query, and shapes that allow multiple.

### At most one query fires

**Single endpoint** — only one endpoint is ever applicable. No
routing decision to make.
Cats 1, 2, 3, 4, 5, 8, 9, 10, 13, 20, 22, 24, 27, 29, 30, 31.

**Mutually exclusive** — two (or more) endpoints could each
individually answer the question, but they answer *different
versions* of it. The query-generation LLM picks whichever matches
the user's framing and ignores the others. Firing both would mix
answers to different questions rather than reinforce a single one.
Cats 11 (canonical genre → Keyword; qualifier-laden genre →
Semantic), 12 (tradition tag → Keyword; no tag → Metadata fallback).

**Tiered** — an ordered preference list of endpoints. The LLM
fires whichever is the first *genuine fit* for the user's phrasing.
Earlier tiers are authoritative when they apply; later tiers exist
as fallbacks for cases the earlier tiers can't cleanly express
(typically spectrum-framed or long-tail asks outside the canonical
vocabulary).
Cats 6, 7, 14, 15, 16, 21, 26.

### More than one query may fire

**Combo** — multiple endpoints apply to the same request and each
carries distinct, complementary signal that can't be collapsed
into a single call. All applicable endpoints fire in parallel and
their outputs populate the handler's return buckets. Combo is
reserved for categories where forcing a single endpoint would drop
real signal — either because no single endpoint's data shape fully
covers the question, or because the question is inherently
multi-faceted (e.g. seasonal intent spans proxy tags, watch context,
and narrative setting at once).
Cats 17, 18, 19, 23, 25, 28.

---

## Where every data gap lands

| Gap | Category | How handled |
|---|---|---|
| Below-the-line credits (cinematographer, editor, etc.) | Cat 29 | Dedicated category — Semantic |
| Curated canon (Criterion, AFI, etc.) | Cat 28 | Dedicated category — Semantic + META.reception_score additive combo |
| Source-material author (Stephen King adaptations, etc.) | Cat 30 | Dedicated category — Semantic only (relies on Cat 5 co-emission) |
| Gateway / entry-level | Cat 23 | Semantic (watch_context + reception) in additive combo |
| Cultural tradition | Cat 12 | KW tag first, META country/language fallback |
| Seasonal | Cat 19 | KW proxy chains + Semantic additive combo |
| Narrative setting time/place | Cat 20 | Semantic (plot_events prose) |
| Cast popularity ("stacked cast") | Cat 25 | Semantic (reception prose) |
| Thematic weight ("has something to say") | Cat 25 (acclaim framing) + Cat 27 (vibe framing) | Framing-dependent routing |
| Character-vs-plot focus | Cat 16 | Semantic (narrative_techniques) fallback tier |
| Character archetype ("lovable rogue") | Cat 7 | KW tag first, Semantic fallback |
| Plot element / motif (clowns, zombies, sharks) | Cat 6 | KW tag first, Semantic fallback |
| Self-experience goal ("make me cry") | Cat 23 | Semantic + KW additive combo |
| Scale / scope | Cat 27 | Semantic (anchor + plot_analysis) |
| Rewatch value | Cat 23 + Cat 25 | Cat 23 for "comfort rewatch" (watch_context), Cat 25 for "holds up on rewatch" (reception prose) |
| Cultural influence | Cat 25 | Semantic (reception prose) |
| Still-holds-up | Cat 25 | Semantic (reception prose) |
| Live trending | Cat 9 | Dedicated endpoint |

## Composition notes

Categories are composable atoms, not mutually exclusive buckets.
"Tom Hanks comedies from the 90s rated above 8" fires Cat 1 (actor)
+ Cat 11 (genre — drama/comedy) + Cat 10 (release date + numeric
reception). "Best horror of the 80s" fires Cat 25 (superlative) +
Cat 11 (horror) + Cat 10 (date). The dispatcher resolves each
category independently and merges scores.

## Compound split rule

**If a phrase or query seems to fit multiple categories, that is a
signal it should be split into separate atomic requirements.** The
upstream category-identification step is expected to decompose
compound phrases into their constituent category firings rather
than inventing an umbrella category to absorb the compound.

Compound descriptors never warrant their own category — the word
"classic" means older + canonical, and the correct handling is to
fire Cat 10 (release era) + Cat 25 (canonical / acclaimed)
simultaneously. Creating a "Canonical stature" category to hold
"classic" would just duplicate endpoints already covered by
Cats 10 and 25 while hiding the compound nature from dispatch.

Examples:
- "Classic Arnold Schwarzenegger action movies" → Cat 1
  (Schwarzenegger) + Cat 11 (action) + Cat 10 (older era) +
  Cat 25 (canonical stature).
- "Disney classics" → Cat 3 (Disney) + Cat 10 (older era) +
  Cat 25 (canonical).
- "Lone female protagonist" → Cat 7 (female-lead archetype) +
  Cat 16 (single-lead structural form).
- "Modern classic" → Cat 10 (recent era, narrower range) +
  Cat 25 (canonical stature).

The only time a compound stays bound to a single category is when
the category explicitly owns the compound — e.g. a named curated
list ("Criterion Collection") in Cat 28, which *is* the compound
of "canonical recognition + specific named list."

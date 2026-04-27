Expanded Proposed Changes

1. Surface failed handlers (logging + user-facing flag)
* Problem: Q2's streaming filter silently vanished after a Pydantic crash, leaving a hard constraint unenforced and invisible.
* Fix: When a handler crashes or returns empty, log it and attach a "couldn't enforce X" note to the response.
* How: Makes silent degradation observable so it can be diagnosed and users know which constraints weren't honored.
2. Format/runtime floor for movie searches
* Problem: Q5's top results were polluted by ~10-minute Animatrix shorts and DTV animated features that aren't what "best superhero movies" means.
* Fix: Default-exclude content under 60 minutes from movie queries unless the query explicitly asks for shorts.
* How: Removes a class of candidates that no scoring tweak would naturally identify as wrong.
3. Programmatic filter/trait routing for metadata, with soft-falloff for softening
* Problem: Q2's "around 90 minutes" was demoted to a soft trait, letting films way outside the range rank high; current date filters are also hard step functions with no wiggle room.
* Fix: Force concrete numeric/categorical metadata (runtime, date, genre, streaming) to always be filters; soften constraints by widening ranges with a gradient falloff (full credit inside, decaying credit outside, zero past the outer bound), never by demoting to trait.
* How: Removes LLM judgment from a place it consistently fails and gives "around" / "in the 2000s" the natural softness the user expects without abandoning the constraint.
4. Vague quality vocabulary maps to predetermined query modifications
* Problem: Q2 ("substance") and Q5 ("best") both ran with quality=off despite the user explicitly caring about quality.
* Fix: Curated lexicon mapping vague descriptors ("best," "substance," "great," "iconic," "underrated," "mindless fun") to a fixed bundle of prior modifications and pre-canned reception/quality sub-queries — LLM picks the descriptor, system runs the bundle.
* How: Removes the LLM's freedom to under-translate vague signals and guarantees explicit quality language always activates the matching backstop.
5. Tonal negations rewritten as positive opposite searches
* Problem: Q4's "without being depressing" failed because depressing and grief embed near each other, so semantic downranking pulled down the entire grief region.
* Fix: Detect tonal/viewer_experience negations and translate to positive searches for the opposite vocabulary ("not depressing" → "hopeful, cathartic, life-affirming"); never downrank for vibe negations.
* How: Uses the embedding space in the direction it actually has signal instead of trying to push down a region too entangled to separate.
6. User-preference defaults for vague descriptors
* Problem: "Modern" defaults to 5–10 years, which doesn't match your mental model — and there's no slot for personal calibration of similarly vague terms ("recent," "old-school," "long").
* Fix: Maintain a per-user defaults table (modern: last 15 years, short: <100 min, etc.) that the LLM consults when resolving vague descriptors before any handler runs.
* How: Anchors ambiguous terms to user-specific values consistently rather than letting each query re-guess.
7. Trait grouping IDs with within-group normalization, dealbreaker/preference at group level
* Problem: A single user intent that expands into many sub-searches (multi-space, multi-actor, CENTRAL+SUPPORTING) drowns out a sibling intent that resolved to one search.
* Fix: Step 2 produces high-level user-intent groups, each with a unique ID and a dealbreaker/preference designation; every downstream sub-query inherits the parent ID; final scoring normalizes within group before combining across groups.
* How: Preserves the original query structure through retrieval so each stated intent contributes equally regardless of how much expansion happened, and moves the filter/trait choice up to where it belongs.
8. Inter-atom awareness in step 2; combine qualifying atoms before searching
* Problem: Q2's "light + feel-good + substance" was split into three overlapping searches; Q4's "grief without being depressing" was split into two contradicting ones; "modern classics" would split into two atoms that lose the meaning of the conjunction.
* Fix: Step 2 emits each atom with explicit relationship metadata (qualifies, modifies, contradicts) to its siblings, and atoms in qualifying/contradicting relationships get fused into a single composite semantic query.
* How: Lets the embedding query land on the actual textured concept the user described instead of trying to reconstruct it through additive intersection across spaces that don't naturally overlap.
9. Per-query preference weighting via 5 discrete balance states
* Problem: Q5's "best" trait was capped too low to outrank low-quality genre matches; other queries have the opposite issue where one trait shouldn't dominate.
* Fix: At step 2, the LLM picks one of 5 discrete dealbreaker-vs-preference balance states for the query (e.g., dealbreakers_dominant, balanced, preferences_dominant); the state maps to a fixed weight schedule.
* How: Lets the dominant signal in a query actually dominate while staying simpler than continuous weights, and the discrete states are easier to audit and tune than free-form numbers.
10. Multiplicative implicit-prior application
* Problem: Q4's flat 0.25 popularity boost on a max-1.5 filter score is ~17% influence regardless of how well the movie matched the actual constraints, washing out tonality signals.
* Fix: Apply implicit priors multiplicatively (score *= 1 + alpha * factor) so the boost scales with base relevance.
* How: Stops popular-but-irrelevant films from getting the same absolute boost as relevant films, preserving the constraint signal.
11. Rarity-weighted scoring at the trait-group level
* Problem: Q1's Memento ranked #16 because a film matching all 5 generic atoms vaguely outranks the canonical film matching the unique conjunction.
* Fix: Weight each trait group's score contribution inversely with how many candidates match it (log-scaled and capped); apply rarity at the group level (per #7), not at the sub-query level, so expansion doesn't double-count.
* How: Naturally rewards rare/specific signals and conjunctions thereof, without needing special-case intersection logic.
12. Dual-firing keyword + semantic when keyword tag partially captures the concept
* Problem: "Plot twist" keyword doesn't fully cover "twist ending"; "iconic" went to a semantic search when no keyword captures the superlative; "murder mystery" missed the SUSPENSE_MYSTERY tag.
* Fix: When a keyword tag is a partial match for a user concept, fire both the keyword filter and a semantic query for the gap, then merge under one trait_id; also audit and expand the keyword registry where common concepts are missing tags.
* How: Stops partial keyword coverage from silently dropping the residual meaning into the void, and uses semantic to cover what the closed taxonomy can't express.
13. Freeform pre-interpretation pass for vague atoms in step 2
* Problem: Vague atoms ("substance," "iconic," "feel-good with edge") get categorized straight into endpoint buckets without first establishing what they actually mean in this query's context.
* Fix: Insert a per-atom freeform interpretation step after extraction but before category assignment — the LLM writes 1–2 sentences on what the atom means here, and category assignment + endpoint routing reads from that interpretation.
* How: Forces the system to commit to a meaning for the vague term before downstream handlers translate it into queries, reducing the noisy guess-from-thin-air problem.
14. Adaptive thresholds for filter softening based on result distribution
* Problem: A hard filter that returns too few candidates leaves a thin pool; a soft trait producing flat scores doesn't actually rank anything.
* Fix: After initial retrieval, widen filter tolerance if it produced fewer than N candidates; promote a trait toward filter behavior if its score distribution is flat.
* How: Adapts the dealbreaker/preference balance to the actual candidate distribution rather than locking in the LLM's a-priori classification.
15. Precomputed entity-popularity tables (franchises first, extend to actors/directors)
* Problem: "Iconic franchises" or "auteur directors" have no backing data, so the LLM either hallucinates examples or skips the constraint.
* Fix: Build a franchise popularity table (avg or max popularity of member movies, not sum) and similar tables for actor career notability and director auteur status; vague reference classes resolve against these tables before further retrieval.
* How: Replaces LLM guesswork with grounded rankings for the most common vague reference classes, and provides a clean foundation for #17.
16. Context-aware semantic expansion conditioned on the whole query
* Problem: Semantic queries are generated atom-by-atom with thin term sets (Q4's grief query had 3 emotional_palette terms total); the expansion doesn't know what role this atom plays in the broader query.
* Fix: Pass the full query intent summary into each semantic expansion call and let the LLM expand more aggressively (8–12 terms) when the atom is dominant, terser when it's supporting.
* How: Right-sizes term coverage to the atom's role in the overall query so dominant tonal queries get the expansion density they need.
17. Parametric-knowledge handler for meta-reasoning
* Problem: Q3's "comedians who did drama" was classified as Interpretation-required and silently dropped because no handler exists for queries needing world knowledge.
* Fix: New endpoint type where the LLM expands a vague reference class (career profiles, era characteristics, aesthetic schools) into concrete instances, which then route to the normal entity/franchise handlers; pairs with #15 where structured tables exist and falls back to LLM expansion where they don't.
* How: Gives meta-reasoning queries a real path through the pipeline and combines naturally with #7 so the expansion doesn't dominate scoring.
18. Ingestion-aligned semantic query phrasing
* Problem: Vector queries are written as overviews/summaries, but each ingestion vector was generated from a different format (transcript-style for plot events, palette terms for viewer experience) — the query and document distributions don't match.
* Fix: Re-think query phrasing per vector space to mirror the ingestion text format for that space, so query embeddings land in the same neighborhood as document embeddings.
* How: Improves cosine similarity quality across the board by closing the format gap; biggest scope of any item here, biggest potential payoff for semantic precision.

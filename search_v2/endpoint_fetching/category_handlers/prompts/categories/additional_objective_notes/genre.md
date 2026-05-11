# Additional objective notes

## Category Target

Genre identity: top-level genres and named sub-genres. Ask: "What genre
or sub-genre is the call asking for?"

## Genre is mutually exclusive — overrides the bucket default

This category's two endpoints answer DIFFERENT VERSIONS of the same
question, not complementary slices. The bucket's general guidance about
overlap, refinement, and "fire both when one fills the other's weakness"
does NOT apply here. Commit EXACTLY ONE endpoint; the other abstains.

Genre is the rare case where firing both endpoints under MAX combine
adds no recall (a clean keyword hit already scores films at ~1.0) and
imports semantic noise from adjacent films into the result set. The
strict-mutex rule below is the operational consequence.

- **Commit KEYWORD** when the call points at a registry member that is
  DEFINITIONALLY EQUIVALENT to the user's genre — same concept, same
  retrieval target, same set of films. Surface-spelling differences are
  fine ("noir" → `FILM_NOIR`, "scary movies" → `HORROR`, "slasher" →
  `SLASHER_HORROR`). When keyword commits, semantic abstains with
  `commitment-criteria-fail` — its slice is already covered, and adding
  it injects noise from genre-adjacent films.
- **Commit SEMANTIC** when no registry member definitionally covers the
  call: a sub-genre or hybrid the registry doesn't carry ("neo-noir",
  "elevated horror", "cosmic horror"), or a qualified texture without
  a canonical compound. When semantic commits, keyword abstains with
  `commitment-criteria-fail` — stretching to an adjacent registry
  member would tag-match adjacent-but-wrong films at 1.0 while the
  genuinely-relevant films score 0.

### Definitional-equivalence test (for keyword commit)

Ask: would a movie fan, told "this film is `<registry member>`",
consider that THE SAME STATEMENT as "this film is `<user phrase>`"?

- Yes → commit keyword.
- "Kind of" / "close enough" / "the closest one we have" → commit
  SEMANTIC. That hesitation is the stretching failure mode, not a
  reason to commit keyword anyway.

Concrete reads:

- `HORROR` vs "scary" → yes, same statement. Keyword.
- `SPAGHETTI_WESTERN` vs "spaghetti westerns" → yes. Keyword.
- `DARK_COMEDY` vs "dark comedy" → yes, the registry carries the
  compound as a single member. Keyword.
- `FILM_NOIR` vs "neo-noir" → no, different movements with different
  conventions and different films. Semantic.
- `HORROR` vs "elevated horror" → no, "elevated" picks out a specific
  subset that `HORROR` would over-pull. Semantic.
- `WESTERN` vs "weird western" → no, the modifier defines the slice.
  Semantic.

## Semantic body authoring (when semantic commits)

When semantic commits, populate ONLY `plot_analysis.genre_signatures`
with the user's exact phrase as one short term. The genre-specific
authoring rules override the semantic endpoint's default register
guidance:

- ONE term, the user's phrase verbatim. Do not paraphrase, expand into
  synonyms, or split the phrase across multiple terms.
- Do NOT populate other plot_analysis sub-fields (`elevator_pitch`,
  `plot_overview`, `thematic_concepts`, `character_arcs`,
  `conflict_type`). Genre identity is not carried by those fields.
- Do NOT populate other vector spaces. Genre identity is a
  `plot_analysis` concern.

The synonym-density and cross-field repetition rules that work for
emotional/experiential bodies DRIFT the retrieval target here.
`["neo-noir", "noir-influenced", "post-noir"]` lands on films described
as any of those, not on films specifically described as neo-noir. The
user's word is the signal; everything else is noise.

## Boundaries

- Story shapes ("revenge", "underdog", "survival", "post-apocalyptic")
  → Story / thematic archetype.
- Formats ("documentary", "anime", "mockumentary") → Format + visual.
- Pure tone without genre anchor ("dark", "whimsical") → Emotional /
  experiential.

## No-Fire

If the call's intent does not name a genre identity, abstain on both
endpoints. A misrouted call should produce an honest double-abstain,
not a coerced commit on either endpoint.

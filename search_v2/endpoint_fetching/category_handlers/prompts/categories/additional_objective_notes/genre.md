# Additional objective notes

## Category Target

Genre identity OR pseudo-genre description: top-level genres, named
sub-genres, and free-form descriptions of film clusters that don't have
a registry or canonical-sub-genre name. Ask: "Is the call naming an
established genre identity, or describing a film-cluster that isn't a
registry/sub-genre name?"

## Triage — Case A vs Case B

Two response shapes live in this category. Decide the shape first; the
authoring rules below split on that decision.

- **Case A — established genre identity.** The call names a registry
  member ("horror", "slasher", "dark comedy") or a canonical sub-genre
  the registry doesn't carry but is still a recognized identity
  ("neo-noir", "elevated horror", "cosmic horror", "spaghetti western").
  The modifier, when present, RESTRICTS the genre slice — picks out a
  subset of the parent genre.
- **Case B — pseudo-genre description.** The call describes a film
  cluster the user can point at but that isn't a genre or sub-genre
  name. Phrasing is free-form: descriptor + axis ("comic book visual
  style", "noir cinematography"), pure descriptor ("stylized
  animation"), viewer-seeking framing ("scratches the slasher itch"),
  acclaim framing ("celebrated noir voiceover storytelling"),
  cross-genre aesthetic transplant ("anime energy in live action").
  The defining signal is orthogonal to genre identity — visual, aural,
  experiential, structural, or appetite-shaped — not a slice of an
  established genre.

The discriminator is whether the call names a genre identity or
describes a cluster around one. **Default to Case A on ambiguity** —
the Case A authoring rules are stricter and a Case A-classified call
that turns out to be aspect-shaped degrades less than a Case B-classified
call that was actually genre identity.

The two cases have completely different authoring rules — read the
correct section below.

## Case A authoring

### Mutually exclusive endpoints — overrides the bucket default

In Case A the two endpoints answer DIFFERENT VERSIONS of the same
question, not complementary slices. The bucket's general guidance about
overlap, refinement, and "fire both when one fills the other's weakness"
does NOT apply here. Commit EXACTLY ONE endpoint; the other abstains.

Genre identity is the rare case where firing both endpoints under MAX
combine adds no recall (a clean keyword hit already scores films at
~1.0) and imports semantic noise from adjacent films into the result
set. The strict-mutex rule below is the operational consequence.

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

### Semantic body authoring (Case A, when semantic commits)

Populate ONLY `plot_analysis.genre_signatures` with the user's exact
phrase as one short term. The genre-specific authoring rules override
the semantic endpoint's default register guidance:

- ONE term, the user's phrase verbatim. Do not paraphrase, expand into
  synonyms, or split the phrase across multiple terms.
- Do NOT populate other plot_analysis sub-fields (`elevator_pitch`,
  `plot_overview`, `thematic_concepts`, `character_arcs`,
  `conflict_type`). Genre identity is not carried by those fields.
- Do NOT populate other vector spaces. In Case A, genre identity is a
  `plot_analysis` concern and other spaces would import drift.

The synonym-density and cross-field repetition rules that work for
emotional/experiential bodies DRIFT the retrieval target here.
`["neo-noir", "noir-influenced", "post-noir"]` lands on films described
as any of those, not on films specifically described as neo-noir. The
user's word is the signal; everything else is noise.

## Case B authoring

### Endpoint commitment — semantic only

In Case B, **keyword always abstains** with `commitment-criteria-fail`.
Pseudo-genre descriptions don't have registry members that supersets
them — stretching to the nearest registry member (e.g.,
`COMIC_BOOK_ADAPTATION` for "comic book visual style") would tag-match
every member of the registry slice regardless of whether the film fits
the described cluster. The keyword superset test fails by construction.

**Semantic always commits** in Case B. The handler's job is to
interpret the cluster across whichever vector spaces honestly cover it.

### Eligible vector spaces

Authoring follows the standard per-space rules in
[`endpoints/semantic.md`](../../endpoints/semantic.md) — sub-field
density, substitution test, register table, default-populate negations
on `viewer_experience`, canonical labels verbatim on
`narrative_techniques`. Only the menu of eligible spaces differs from
the default.

**Primary homes (one or more almost always fires):**

- `viewer_experience` — experiential, sensory, and tonal facets of the
  cluster. The workhorse space for Case B because most pseudo-genre
  descriptions are pointing at what it feels like to watch the cluster.
  Sub-fields commonly active: `sensory_load`, `tone_self_seriousness`,
  `emotional_palette`, `tension_adrenaline`.
- `production` — concrete craft that defines the cluster.
  `production_techniques` is the typical sub-field; `filming_locations`
  is rarely relevant in Case B.
- `watch_context` — when the call frames the cluster as a draw or as
  a viewer appetite. `key_movie_feature_draws` for noteworthy aspects
  framed as reasons to watch ("great soundtrack", "stylized animation"
  treated as a draw); `self_experience_motivations` for viewer-seeking
  phrasings ("scratches the slasher itch").
- `narrative_techniques` — when the cluster is defined by structural
  craft tied to the pseudo-genre. Canonical craft labels verbatim
  ("voiceover narrator", "unreliable narrator", "non-linear timeline",
  "anthology structure") — do not paraphrase.

**Situational:**

- `reception` — when the user's phrasing names an *acclaimed* aspect
  rather than a descriptive one ("praised for noir cinematography",
  "celebrated kung-fu choreography"). `praised_qualities` is the
  sub-field. If the phrasing is descriptive rather than acclaim-shaped,
  default to `viewer_experience` / `production` instead.

**Excluded — never populate in Case B:**

- `plot_analysis` — the Case A/B split exists to keep Case B out of
  this space. Pasting a pseudo-genre phrase into `genre_signatures`
  embeds against ingest text written for genre labels and retrieves
  noise. The empty `plot_analysis` body is the load-bearing signal of
  a Case B commit.
- `plot_events` — too literal. A pseudo-genre description doesn't name
  on-screen events; populating `plot_events.plot_summary` fabricates
  plot detail the cluster doesn't imply (the failure mode called out
  in `endpoints/semantic.md`'s "no fabrication" section).

### How many spaces to fire

One or two spaces is the typical commit. The cluster usually has a
clear primary axis (visual texture, viewer appetite, structural craft,
acclaim) that picks one space, plus at most one complementary space
that the same cluster genuinely also implicates. Firing all four
primary spaces sprawls the query vector toward films that match any
facet rather than the cluster as a whole; honor the `endpoints/semantic.md`
discipline of populating only sub-fields the call grounds.

## Boundaries

- Story shapes ("revenge", "underdog", "survival", "post-apocalyptic")
  → Story / thematic archetype.
- Formats ("documentary", "anime", "mockumentary") → Format + visual.
- Pure tone without genre anchor ("dark", "whimsical") → Emotional /
  experiential.

These remain the cleanest upstream homes. When Step 3 routes a call
here whose canonical home was elsewhere, the Case B branch recovers it
by authoring across the eligible spaces above — even though a cleaner
fix would have been upstream routing.

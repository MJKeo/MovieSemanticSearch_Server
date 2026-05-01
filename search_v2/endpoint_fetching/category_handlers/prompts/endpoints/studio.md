# Endpoint: Studio (Production Company)

## What this endpoint does

Translates a CategoryCall about production companies — `expressions: list[str]` (one phrase per dimension Step 3 routed here) plus `retrieval_intent` — into a `StudioQuerySpec`. The executor runs each `StudioRef` against either the closed-enum brand registry or a freeform IMDB-token resolver, then combines per-ref scores by `scoring_method`.

## What does NOT belong here

- Streaming-platform availability ("movies AVAILABLE on Netflix") → metadata streaming endpoint. Streamer-vs-producer disambiguation was resolved upstream — when an item reaches this endpoint, treat the entity as a producer. `NETFLIX`, `AMAZON_MGM`, `APPLE_STUDIOS` are producer brands here.
- Named persons → entity endpoint.
- Named franchises / shared universes → franchise endpoint.

## The three commitments

Each commitment reads off the reasoning prose committed before it.

1. **Studio inventory** (`exploration` → `studios`) — which distinct studios is the user naming? Two expressions naming the same studio under different aliases collapse to ONE `StudioRef`; two expressions naming different studios become two `StudioRef`s.
2. **Per-studio routing** (`studio_exploration` → `brand` or `freeform_names`) — for each `StudioRef`, brand (umbrella registry) or freeform (sub-label / long-tail).
3. **Scoring method** (`exploration` → `scoring_method`) — `ANY` (or-case) or `ALL` (all-of), driven by `retrieval_intent`.

`retrieval_intent` carries operational context the bare `expressions` strings cannot — read it for the scoring-method commitment AND for any phrasing that disambiguates which level (parent brand vs. sub-label) the user means.

## Brand vs. freeform

The umbrella-vs-subsidiary call is the most failure-prone decision at this endpoint.

**Brand** is the right path only when the user clearly meant the entire brand's catalog — every subsidiary, every historical sub-label, every era. The brand path auto-handles time-bounded ownership (Lucasfilm joined Disney in 2012, so `disney` does NOT match Star Wars 1977) and rename chains (Twentieth Century Fox → 20th Century Studios). Don't reason about timelines; pick the brand, the data handles it.

**Freeform** is the right path when the user named a sub-label, subsidiary, or long-tail studio — even when it falls under a covered umbrella:

- Sub-labels of a covered brand are NOT the brand. Touchstone ≠ Disney; Fox Searchlight ≠ 20th Century; HBO Documentary Films ≠ HBO; Marvel Television ≠ Marvel Studios.
- A studio with its own registry entry that is also a member of a larger umbrella reads off the user's specificity: "Marvel Studios films" → `marvel-studios` (the narrower reading); "Disney movies" (which sweeps everything Marvel along with the rest) → `disney`.
- Long-tail, niche, and foreign studios absent from the registry → freeform with native surface forms IMDB credits use.

When unsure: if no registry brand on the table below covers the user's phrasing at the right level, route freeform.

## Registry brands

Closed set of registry brand values, each followed by a sample of IMDB `production_companies` surface forms counting as that brand. The form list is a sample, not exhaustive — umbrella brands cover more strings than shown. Use this table to decide (a) which brand to pick when umbrella applies, and (b) whether the query is umbrella-level at all — when no registry brand covers the phrasing, route freeform.

{{BRAND_REGISTRY}}

## Scoring method — reading retrieval_intent

`retrieval_intent` is the disambiguator. Patterns:

- **Set / OR → `ANY`**: we only care if the movie has at least one of the named studios, like an "or" case. Movies score equally high for matching 1+ values. Cues: "like A24, Neon, Mubi"; "any of"; "anything from"; "studios like"; comma-separated alternatives without joining language.
- **Conjunction / AND → `ALL`**: we care how many named studios a given movie matches. Movies score higher depending on how many values they match. Cues: "co-produced by"; "X and Y partnership"; "both worked on"; "joint production"; "made together".

Ambiguous case — `retrieval_intent` lists studios without joining language ("Pixar, DreamWorks, Illumination movies") — default to `ANY`. The user is more likely sweeping a category than requiring every studio to appear on the same film.

## Surface forms for freeform_names

The slots are for variants of ONE studio most likely to appear verbatim in IMDB's `production_companies` credits. A good covering set:

- Condensed / acronym ("MGM", "HBO", "BBC").
- Expanded / full ("Metro-Goldwyn-Mayer", "Home Box Office").
- Alternate well-known variant when a distinct form exists ("Studio Ghibli" alongside "Ghibli").

Pick by what IMDB credits, not by what the user typed:

- "WB" rarely appears standalone in credits → emit "Warner Bros." instead (and prefer `brand=warner-bros` anyway).
- "Mouse House" never appears in credits → don't emit it; use `brand=disney`.
- "Ghibli" expands to "Studio Ghibli" because that is the credited form.
- "Tri-Star" / "TriStar" / "Tristar" all tokenize equivalently — emit the most common form, the executor handles the rest.

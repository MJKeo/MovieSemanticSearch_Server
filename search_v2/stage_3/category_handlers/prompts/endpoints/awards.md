# Endpoint: Award

## Purpose

Translates an award requirement into filter parameters across five independent axes (ceremonies, award names, category tags, outcome, years) plus a scoring shape (mode + calibration mark) that converts a count of matching award rows into a `[0, 1]` score. Execution runs the filter against `movie_awards` (or the `award_ceremony_win_ids` fast path on `movie_card`) and applies the scoring formula.

## Canonical question

"Which award filter axes does this requirement signal, and what count-intensity pattern scores the result?"

## Capabilities

- Filter on any combination of: ceremony (event/festival), specific prize name, category tag (role/discipline), outcome (win vs. nomination), year range.
- Scoring converts a raw count of matching award rows into a score: binary presence (`floor`) or gradient toward saturation (`threshold`).
- Handles both specific requests ("won Best Picture at the Oscars in 2021") and generic intensity requests ("heavily decorated", "award-winning", "most decorated").

## Boundaries (what does NOT belong here)

- "Critically acclaimed" / "well-reviewed" without naming awards → metadata reception.
- "Popular", "underrated", "blockbuster" → metadata popularity / box_office.
- Specific praise or criticism of craft elements ("praised for its cinematography") → semantic reception.
- "Worst movies", "critically panned", "poorly received" → metadata reception (low-pole). Razzie data is only pulled when the user explicitly names Razzies or Golden Raspberries (see Razzie Handling).

## Scoring shape

Every spec has a `scoring_mode` and a `scoring_mark`. Together they convert a raw count of matching rows into `[0, 1]`.

**Count unit:** distinct prize rows. Different ceremony, category, prize name, or year each count separately. "Won 11 Oscars" is 11 rows, not 1 ceremony.

**Two modes:**
- `floor` — Binary. 1.0 if row count ≥ `scoring_mark`, else 0.0. For hard thresholds: a specific ceremony, specific category, explicit count floor, or any filter where presence/absence is the right answer.
- `threshold` — Gradient. `min(row_count, scoring_mark) / scoring_mark` — ramps from 0.0 at zero rows to 1.0 at the mark, holds at 1.0 above. For requirements where more wins should score higher up to saturation: generic "award-winning", superlative language, qualitative-intensity language.

**Key distinction:** A generic award concept with no ceremony / prize-name / category filter calls for `threshold` — more wins should score higher, since one win is less "award-winning" than ten. A specific filter ("Oscar Best Picture winner") calls for `floor` — the user wants binary presence, not a gradient.

**Five canonical patterns.** Classify the full requirement before committing; do not anchor on the first intensity-like word you see.

| Pattern | Language signal | scoring_mode | scoring_mark |
|---|---|---|---|
| generic award-winning | "award-winning", "critically decorated", "won awards" (no specific ceremony / prize / category) | threshold | 3 |
| specific filter, no count | "Oscar-winning", "won the Palme d'Or", "Best Picture winner", "BAFTA-nominated", "won at Cannes" | floor | 1 |
| explicit count: N | "at least 3 wins", "won 5 Oscars", "won twice", "multiple" (= 2) | floor | N |
| superlative | "most decorated", "most award-winning", "has the most Oscars", "best-decorated of all time" | threshold | 15 |
| qualitative plenty | "heavily decorated", "loaded with awards", "swept the ceremony", "multiple award wins" with no explicit number | threshold | 5 |

**Note:** "Oscar-winning" is NOT generic — a specific prize is named, so it is specific-filter-no-count (floor / 1).

## Filter axes

Five independent axes. Populate only the ones the requirement signals; leave others null. Multiple populated axes AND together; list entries within an axis OR.

### Ceremonies

Twelve tracked ceremonies with exact stored string values. A one-character difference in the emitted string produces zero matches. Use ceremonies for event / festival / awards-body wording ("at Cannes", "nominated at Sundance", "Academy Awards ceremony") — not as a proxy for a named prize object. Leave null when no ceremony is named; null means all non-Razzie ceremonies apply (see Razzie Handling).

{{CEREMONY_MAPPINGS}}

### Award names

The specific prize name — the individual award object (the ceremony is the event granting it). Common names: "Oscar", "Palme d'Or", "Golden Globe", "BAFTA Film Award", "Golden Lion", "Golden Bear", "Silver Bear", "Jury Prize".

Emit when the user names the specific prize, even if that prize implies a ceremony. Represent what the user asked for at the most direct level. "Oscar-winning", "won an Oscar", "Palme d'Or winners", "won a Golden Globe" → populate `award_names`. Do NOT automatically add the related ceremony just because the prize belongs to one. Use both axes only when the query explicitly names both levels ("Cannes Palme d'Or winners"). A ceremony alone does not imply a prize name; a prize alone does not imply a ceremony filter.

**Tokenization.** Award names resolve through a shared tokenizer + inverted index. Tokenizer: lowercase, diacritic fold, punctuation strip, apostrophe fold (straight and curly both drop), whitespace collapse, ordinal digit-to-word ("8th" → "eighth"), cardinal 0–99 digit-to-word, whitespace+hyphen split, stopword drop ("award", "awards", "prize", "prizes", "film", "films", "best", common English connectives). The same tokenizer ran at ingest — "Palme d'Or" vs "Palme d'Or", "Critics Week" vs "Critics' Week", "BAFTA" vs "BAFTA Film Award" collapse automatically. Capitalization, punctuation, and apostrophe style do not matter for retrieval.

**Base form.** Emit the OFFICIAL BASE FORM of the prize — "Palme d'Or", "Oscar", "Golden Lion", "BAFTA Film Award", "Grand Jury Prize". The anchor table below guides correct base form per ceremony (e.g. SAG stores the prize as "Actor", not "Actor Award"). It is **guidance, not a closed vocabulary** — when a user names a prize not in the table, use your knowledge of IMDB nomenclature for that ceremony and emit the base form directly. Do NOT pattern-match onto a similar-looking entry when the user named a different specific prize (a user asking for "Cannes Jury Prize" emits "Jury Prize", not "Palme d'Or").

**Specificity.** Emit the BASE prize name for umbrella queries ("Palme d'Or winners" → "Palme d'Or" sweeps every Palme variant). Emit a narrower sub-variant ("Palme d'Or Best Short Film", "Silver Berlin Bear Jury Grand Prix") only when the user explicitly asked for it. Token intersection sweeps sibling variants naturally; do not enumerate them.

**Count.** 1 entry in the common case. 2–3 only when genuinely different canonical names are in common use for the same prize (OR-union for umbrella sweep). Do NOT pad with casing, punctuation, apostrophe, diacritic, hyphenation, or digit-vs-word variants — the tokenizer collapses those on both sides.

{{AWARD_NAME_SURFACE_FORMS}}

### Category tags

Closed enum of concept tags from a 3-level taxonomy (leaf → mid → group). Pick tags at whatever specificity matches the requirement:

- **Leaf level** (e.g. `lead-actor`, `best-picture-drama`, `worst-screenplay`) — a single specific concept. Use when the user names the exact category including its narrow form ("Best Actor", "Best Adapted Screenplay", "Best Animated Short Film").
- **Mid level** (e.g. `lead-acting`, `music`, `short`, `worst-acting`) — intermediate rollup spanning multiple leaves without spanning the whole group. Use when the requirement is deliberately broader than a leaf but narrower than the group (typically gender-, format-, or medium-neutral). "Won Best Actor or Best Actress" → `lead-acting`. "Won any sound award" → `sound-any`. "Any short film award" → `short`.
- **Group level** (`acting`, `directing`, `writing`, `picture`, `craft`, `razzie`, `festival-or-tribute`) — the whole bucket. "Won an acting award" → `acting`. "Recognized for craft work" → `craft`.

Emit multiple tags when concepts don't share an ancestor below group level ("Best Director or Best Screenplay" → `[director, screenplay-any]`). A row matches if it overlaps with ANY supplied tag. The retrieval layer uses stored ancestor lists, so emitting a group-level tag automatically picks up every leaf and mid under it — do NOT enumerate descendants of a tag already emitted.

**Tag selection is ceremony-agnostic.** The same leaf tag (e.g. `lead-actor`) covers "Best Actor" at the Oscars, "Best Performance by an Actor in a Motion Picture — Drama" at the Globes, "Best Leading Actor" at BAFTA, and dozens of other phrasings.

Leave `category_tags` null when the requirement names no category (pure ceremony query like "Cannes winners" → null; rely on ceremony filter).

{{CATEGORY_TAG_TAXONOMY}}

### Outcome

`winner`, `nominee`, or null. Null = both winners and nominees count. Populate only when the user explicitly distinguishes: "won" / "winner" / "winning" → `winner`; "nominated" / "nomination" / "nominee" → `nominee`. "Award-winning" = `winner`. "Award-nominated" = `nominee`. A requirement with no outcome signal (e.g., "recognized at Sundance") → null.

### Years

Year range (or single year). Null = any year. Populate when the user names a specific year, decade, or era ("2023 Oscars", "late-90s Sundance films", "since 2010"). Resolve relative terms against the supplied `today`: "this decade" = current decade start to today; "recently" ≈ last 2–3 years ending today; "this year" = today's calendar year. Calendar years, not award-ceremony season numbers.

## Razzie handling

Razzie Awards celebrate the worst of cinema and are **excluded by default** from all award counts and filters, even when `ceremonies` is null. A generic "award-winning" query means positive-prestige awards only; Razzie wins do not make a film "award-winning" in the user's sense.

Razzies are included ONLY when the user explicitly names them: "Razzie winners", "Razzie-nominated", "won a Razzie", "Golden Raspberry Award", or any "Worst …" category ("Worst Picture", "Worst Actor"). When you see these, emit `"Razzie Awards"` in `ceremonies` AND, if a worst-category was named, the corresponding `worst-*` tag in `category_tags`. Either signal alone is sufficient (the executor recognizes Razzie intent on either axis); emitting both when both are present makes the spec self-documenting. Razzies may be combined with other ceremonies ("Oscar and Razzie winners").

**Never infer Razzie intent.** "The worst movies" / "critically panned films" is NOT asking for Razzie data — that is low-reception content and routes to metadata. Razzie intent requires explicit Razzie / Golden Raspberry naming.

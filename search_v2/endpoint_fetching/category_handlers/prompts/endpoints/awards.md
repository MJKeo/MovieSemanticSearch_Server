# Endpoint: Award

## Purpose

Use this endpoint for formal award records: wins and nominations at tracked ceremonies, named prizes, award categories, award years, and award-count intensity. The endpoint executes one or more structured award searches and then combines their scores.

The central task is not just "fill award fields." First decide whether the input describes one award search or several award searches.

## Canonical question

"What award search or searches should run, and should their results behave like alternatives or like additive partial-credit requirements?"

## What this endpoint can represent

Each search is one executable award-record lookup. Inside a single search, populated filter axes are ANDed together:

- ceremony / festival / awards body
- named prize
- category or discipline tag
- winner vs. nominee
- award year or year range

List values inside one axis are ORed. For example, one search can mean "winner of Best Actor or Best Actress at the Oscars" because those are same-axis category alternatives under one award condition.

Multiple searches are for separate award conditions. For example, "Oscar winners or BAFTA winners" is two searches, not one search with both axes blended together.

## Input priority

Use `retrieval_intent` to decide the shape of the plan:

- whether the expressions are pieces of one structured search or separate searches
- whether separate searches are alternatives or additive partial-credit requirements
- whether the user wants binary presence, explicit count satisfaction, or gradient recognition strength
- whether context from a qualifier/carver framing changes the threshold or scope

Use `expressions` to populate concrete search values:

- ceremony names
- prize names
- category concepts
- outcome language
- year ranges
- count or intensity language

Do not pull in siblings, raw query context, or unstated assumptions. If context matters, it should already be in `retrieval_intent`.

## One search vs. multiple searches

Create ONE search when the expressions are components of the same award condition.

Examples:

- "Oscar-winning Best Picture movies from the 2010s" → one search with award name, category, outcome, and year filters.
- "nominated at Cannes for directing" → one search with ceremony, category, and outcome filters.
- "won at least 3 Oscars" → one search with award name, outcome, and count scoring.

Create MULTIPLE searches when the expressions name separate award conditions that should be evaluated independently.

Examples:

- "Oscar or BAFTA winners" → one Oscar-winner search, one BAFTA-winner search.
- "won acting awards and directing awards" → one acting-award search, one directing-award search.
- "Oscar-winning and Razzie-nominated" → one Oscar-winning search, one Razzie-nominated search.

Do not force independent award conditions into one search. One search ANDs its filter axes against the same award rows, so blending separate conditions can accidentally require an impossible hybrid row.

## Combine mode

Use `any` when the searches are alternatives. A movie's best search score wins.

Good fits:

- "Oscar or BAFTA winner"
- "Cannes, Venice, or Berlin recognition"
- "won an acting or directing award" when the intent means either path is acceptable

Use `average` when the searches are separate desirable checks and partial matches should still get credit. Missing searches count as zero, so movies matching more checks score higher.

Good fits:

- "won acting and directing awards"
- "recognized by both festivals and major ceremonies"
- multi-expression calls where retrieval_intent frames several award facts as jointly desirable rather than strict alternatives

If uncertain, follow retrieval_intent. If it frames the searches as "either / any / one of," use `any`. If it frames them as multiple traits a movie should ideally satisfy, use `average`.

## Per-search scoring

Score each search from the count of matching award rows.

Use `floor` when the search is a yes/no requirement:

- specific ceremony, prize, or category with no intensity language
- explicit minimum count
- named count such as "won 5 Oscars"
- "multiple" as a hard count means mark 2

Use `threshold` when more matching award rows should make the movie more strongly satisfy the search:

- generic "award-winning" with no ceremony / prize / category filter
- "heavily decorated", "loaded with awards", "swept the ceremony"
- "most decorated", "most award-winning", "has the most Oscars"

Count unit: distinct award rows. Different ceremony, prize, category, outcome, or year rows count separately. "Won 11 Oscars" means mark 11, not one ceremony.

Calibration:

- generic award-winning: `threshold` mark 3
- specific filter, no count: `floor` mark 1
- explicit count N: `floor` mark N
- "multiple" as a hard count: `floor` mark 2
- qualitative plenty: `threshold` mark 5
- superlative / "most decorated": `threshold` mark 15

"Oscar-winning" is not generic award-winning. It names a specific prize, so it is a specific filter with `floor` mark 1 unless count/intensity language changes that.

## Filter axes

### Ceremonies

Use ceremonies for the event, festival, or awards body: "at Cannes", "Sundance nominee", "Academy Awards ceremony". Emit only tracked ceremonies from the registry.

Do not add a ceremony just because a prize implies one. "Oscar-winning" should use the named prize. "Cannes Palme d'Or winner" can use both because the input names both.

{{CEREMONY_MAPPINGS}}

### Award names

Use award names for the specific prize object: "Oscar", "Palme d'Or", "Golden Globe", "BAFTA Film Award", "Golden Lion", "Golden Bear", "Silver Bear", "Jury Prize".

Emit the official base prize name. Do not pad with casing, punctuation, apostrophe, diacritic, hyphenation, or digit-vs-word variants; tokenization handles those. Use 2-3 names only when genuinely different canonical names are needed for the same prize concept.

For umbrella prize asks, prefer the base form. For explicitly narrow variants, emit the narrower prize name.

{{AWARD_NAME_SURFACE_FORMS}}

### Category tags

Use category tags for role, discipline, or category concepts. Pick the broadest tag that exactly matches the request:

- leaf for exact categories: "Best Actor", "Best Adapted Screenplay"
- mid-level rollup for deliberate middle breadth: "Best Actor or Best Actress", "any sound award"
- group-level rollup for whole buckets: "acting award", "craft recognition"

Do not enumerate descendants of a selected parent tag. Stored award rows carry ancestor tags, so one parent tag already covers its descendants.

Tag selection is ceremony-agnostic. The same acting/directing/writing/etc. tag can apply across Oscars, Globes, BAFTA, festivals, and other tracked ceremonies.

{{CATEGORY_TAG_TAXONOMY}}

### Outcome

Use `winner` when the input says won / winner / winning / award-winning.

Use `nominee` when the input says nominated / nomination / nominee / up for.

Use null when the wording is recognition-oriented but not win-vs-nomination specific, such as "recognized at Sundance" or bare ceremony/festival references.

### Years

Use years only for award-year constraints, not release-year constraints. If the input says "recent Oscar winners" and retrieval_intent makes clear that recent modifies the awards, use award years. If it means recently released movies that won awards, the release-year part belongs elsewhere.

Only emit concrete years when the input gives a concrete year/range or retrieval_intent has already resolved the relative phrase into one. Do not invent the current year from model knowledge.

## Razzie handling

Razzies are excluded by default from generic award counts. Generic "award-winning" means positive-prestige awards.

Include Razzies only when explicitly named:

- Razzie
- Golden Raspberry
- Razzie-nominated / Razzie winner
- a "Worst ..." award category

When Razzie intent is present, use the Razzie ceremony and the relevant worst-category tag when applicable. Do not infer Razzie intent from general negative-quality language like "worst movies", "critically panned", or "badly reviewed"; those belong to reception/metadata, not award records.

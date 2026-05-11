# Few-shot examples - Award records

The examples below span the full specificity range the endpoint
handles. Read every one — the right output shape for a given input
changes sharply with how specific the request is (category alone,
prize alone, both, neither, count, recognition strength). The
recurring failure mode is over-specifying filters that the input
never named or splitting one award condition into per-ceremony
searches when an unfiltered ceremony field would have covered them
all at once.

<example>
Input:
```xml
<retrieval_intent>Find films that have won an acting award.</retrieval_intent>
<expressions><expression>award-winning acting</expression></expressions>
```
Expected: fire award endpoint with ONE search.

- `category_tags: ["acting"]` (group-level rollup — the request
  names a discipline, not a specific category).
- `outcome: "winner"` ("winning" is explicit).
- NO `ceremonies` filter and NO `award_names` filter — the
  request doesn't name a ceremony or prize, and an absent
  ceremony/prize filter already spans every tracked ceremony for
  free.
- `scoring: floor, mark 1`.

Do NOT split this into per-ceremony searches (one for Oscars, one
for BAFTAs, etc.). The unfiltered ceremony field covers Oscars,
BAFTAs, SAG, Golden Globes, Critics' Choice, Cannes, Venice,
Berlin, Sundance, Spirit, Gotham simultaneously in a single search;
per-ceremony decomposition adds searches without adding coverage.
Do NOT expand `outcome` to include nominees just because the
acting-award universe contains both — the request said "winning".
</example>

<example>
Input:
```xml
<retrieval_intent>Find films that won an Oscar.</retrieval_intent>
<expressions><expression>Oscar winner</expression></expressions>
```
Expected: one search.

- `award_names: ["Oscar"]` (specific prize named).
- `outcome: "winner"`.
- `scoring: floor, mark 1`.
- NO `ceremonies` filter — the prize already pins the source; a
  redundant `ceremonies: ["Academy Awards, USA"]` adds nothing.
- NO `category_tags` — "Oscar winner" without a category means
  any Oscar category counts.
</example>

<example>
Input:
```xml
<retrieval_intent>Find films nominated at BAFTA.</retrieval_intent>
<expressions><expression>BAFTA nominated</expression></expressions>
```
Expected: one search.

- `ceremonies: ["BAFTA Awards"]` (ceremony named without a
  specific prize → ceremony filter, not prize filter).
- `outcome: "nominee"`.
- `scoring: floor, mark 1`.
- NO `category_tags` — the request isn't category-specific.
</example>

<example>
Input:
```xml
<retrieval_intent>Find films that won the Oscar for Best Director.</retrieval_intent>
<expressions><expression>Oscar-winning Best Director</expression></expressions>
```
Expected: one search.

- `award_names: ["Oscar"]`.
- `category_tags: ["director"]` (leaf — the request names the
  specific category, not the discipline-level rollup).
- `outcome: "winner"`.
- `scoring: floor, mark 1`.

The category and the prize narrow the same row together; one
search ANDs them.
</example>

<example>
Input:
```xml
<retrieval_intent>Find films that won either an Oscar or a BAFTA — either path is acceptable.</retrieval_intent>
<expressions><expression>Oscar or BAFTA winner</expression></expressions>
```
Expected: TWO searches, `combine: any`.

- search 1: `award_names: ["Oscar"]`, `outcome: "winner"`, floor 1.
- search 2: `award_names: ["BAFTA Film Award"]`, `outcome: "winner"`, floor 1.

Two searches because the two prizes name independent award
conditions; ANDing them inside one search via `award_names:
["Oscar", "BAFTA Film Award"]` against a single row would still
require ONE row to satisfy both prize-name strings, which never
happens. `combine: any` because retrieval_intent frames them as
alternatives — a single match is sufficient evidence.
</example>

<example>
Input:
```xml
<retrieval_intent>Find films that won both acting awards and directing awards — partial credit applies when only one fires.</retrieval_intent>
<expressions>
  <expression>won acting awards</expression>
  <expression>won directing awards</expression>
</expressions>
```
Expected: TWO searches, `combine: average`.

- search 1: `category_tags: ["acting"]`, `outcome: "winner"`, floor 1.
- search 2: `category_tags: ["directing"]`, `outcome: "winner"`, floor 1.

Two searches because acting-wins and directing-wins are
independent award conditions; one row cannot be both an acting
win and a directing win. `combine: average` because
retrieval_intent frames them as jointly desirable with partial
credit, not as alternatives.
</example>

<example>
Input:
```xml
<retrieval_intent>Find films that won at least three Oscars.</retrieval_intent>
<expressions><expression>won 3 Oscars</expression></expressions>
```
Expected: one search.

- `award_names: ["Oscar"]`.
- `outcome: "winner"`.
- `scoring: floor, mark 3`.

Explicit count → floor at the named number. Each Oscar win is a
distinct row, so mark 3 means "at least 3 matching rows".
</example>

<example>
Input:
```xml
<retrieval_intent>Find films loaded with award recognition; more wins is better.</retrieval_intent>
<expressions><expression>heavily decorated</expression></expressions>
```
Expected: one search.

- `outcome: "winner"`.
- NO `ceremonies` / `award_names` / `category_tags` filters —
  generic recognition strength.
- `scoring: threshold, mark 5`.

Generic recognition strength → threshold scoring so more matching
rows raise the score gradually rather than passing a binary cut.
Mark 5 follows the "qualitative plenty" calibration; superlative
phrasings ("most decorated") go higher (mark 15).
</example>

<example>
Input:
```xml
<retrieval_intent>Find films recognized at Cannes — wins or nominations both count.</retrieval_intent>
<expressions><expression>Cannes recognition</expression></expressions>
```
Expected: one search.

- `ceremonies: ["Cannes Film Festival"]`.
- `outcome: null` (both winners and nominees count).
- `scoring: floor, mark 1`.

Recognition-oriented wording with no win-vs-nomination verb → null
outcome. Same shape applies to bare festival or ceremony
references ("at Sundance", "Venice presence") when retrieval_intent
doesn't pick a side.
</example>

<example>
Input:
```xml
<retrieval_intent>Find films that won either Best Actor or Best Actress.</retrieval_intent>
<expressions><expression>Best Actor or Best Actress winner</expression></expressions>
```
Expected: one search.

- `category_tags: ["lead-acting"]` (mid-level rollup that covers
  both lead-actor and lead-actress leaves).
- `outcome: "winner"`.
- `scoring: floor, mark 1`.

Do NOT emit `category_tags: ["lead-actor", "lead-actress"]`. The
mid-level rollup already covers both descendants, and stored rows
carry ancestor tags so one parent tag suffices — enumerating
descendants of a chosen parent is redundant.
</example>

<example>
Input:
```xml
<retrieval_intent>Find films nominated for a Razzie.</retrieval_intent>
<expressions><expression>Razzie-nominated</expression></expressions>
```
Expected: one search.

- `ceremonies: ["Razzie Awards"]`.
- `outcome: "nominee"`.
- `scoring: floor, mark 1`.

Razzies fire only when explicitly named — this input names them.
Do NOT route generic negative-quality language ("worst movies",
"critically panned") here; those belong to reception scoring, not
award records.
</example>

<example>
Input:
```xml
<retrieval_intent>Find critically acclaimed films.</retrieval_intent>
<expressions><expression>critically acclaimed</expression></expressions>
```
Expected: no-fire. Reception quality, not award records. Critical
acclaim without an award-vocabulary anchor (ceremony, prize,
outcome verb) is the wrong endpoint regardless of how
prestige-laden the language sounds.
</example>

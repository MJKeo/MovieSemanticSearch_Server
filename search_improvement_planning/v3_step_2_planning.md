# v3 Step 2 Planning

How step 2 (query pre-pass / trait identification) is being redesigned
for v3. Captures the responsibilities, decisions, and final output
schema, along with the reasoning behind each choice.

Read alongside:
- `query_categories.md` — the 1-43 category taxonomy step 2 routes to.
- `v3_trait_identification.md` — the broader decomposition story
  (modifier absorption, atomicity, interpretation deferral).
- `v3_reranking_guide.md` — the rescoring layer that consumes step 2's
  per-trait classifications and the followup LLM's per-query states.
- `carving_qualifying_boundaries.md` — the four-cell role taxonomy.

---

## Step 2's responsibilities

Step 2 takes the user's natural-language query and emits everything
step 3 (per-category handlers) needs to write actual queries against
the corpus. The responsibilities are:

1. **Identify and group high-level traits**, including absorbing the
   modifiers (polarity setters, salience hints, role markers) that
   attach to each trait.
2. **Mark each trait as carver vs qualifier.**
3. **Mark each trait as positive vs negative polarity.**
4. **Assign salience (central / supporting) to each qualifier.**
5. **Pick the best-fit category (1-43) for each trait.**
6. **Provide enough context that step 3 can interpret the trait
   without re-deriving step 2's reasoning.**

Per-query state assignments — carver/qualifier balance state and
implicit-prior strength state — are NOT done in this step. They are
emitted by a separate **followup LLM that runs in parallel with step
3** (see decisions below).

---

## Key decisions (with reasoning)

### Per-query state assignments → separate followup LLM, parallel to step 3

The carver/qualifier balance state and the implicit-prior strength
state are whole-query gestalt judgments, not per-trait classifications.
They benefit from seeing the trait list as input. Three reasons to
split them out:

- **Different reasoning shape.** Trait identification is mechanical
  (segmentation + classification). Per-query states are holistic
  (qualifier richness, headline-vs-supporting balance, mode-word
  presence).
- **Step 2's prompt is already heavy.** 43-category taxonomy +
  atomicity test + modifier registry + worked examples will dominate
  the prompt. Adding two whole-query gestalt judgments overloads it.
- **Per-query states don't gate handler dispatch.** They're consumed
  at rescoring time. So they can run in parallel with step 3 (handler
  dispatch). They converge with step 3's output at the merging stage.

The followup LLM takes (original query, trait list) → emits
(balance_state, implicit_prior_strength).

### Polarity is surface-grammar, not intent

If the query contains a polarity-setter ("not", "without", "no",
"avoid") on a trait, polarity = negative. No interpretation pass.
"Not too boring" → trait="boring", polarity=negative.

Resolves a tension in v3_trait_identification.md L47 (which suggested
intent-based polarity) vs L99 (which described surface-grammar
polarity). The surface-grammar rule is mechanical, supports interpretation
deferral, and leaves any rewrite (e.g. tonal-negation rewriting like
"not depressing" → "hopeful, cathartic") to downstream code.

### One trait : one category mapping (compound rule retired)

Earlier versions of the taxonomy let a single trait fan out to
multiple categories ("classic" → Cat 12 + Cat 38). v3 step 2 maps
each trait to exactly one category. Three implications:

- **The compound split rule is dead.** Replaced by an atomicity test
  on the span: split into multiple traits when each piece could
  classify independently AND splitting preserves meaning.
- **Bucket-B compounds (single concept → one category whose handler
  fans out) are fine.** Cat 32 (emotional/experiential) routes to
  VWX+CTX+RCP+KW from one trait — the handler does the fan-out, not
  step 2.
- **Audit confirmed no new categories are needed.** "Classic," "modern
  classic," "blockbuster," "indie," "popcorn movie," "hidden gem" all
  fit existing categories or split cleanly. Cat 38 is the natural home
  for compound reception-prose descriptors ("classic," "era-defining,"
  "stacked cast"); Cat 17 (financial scale, merged budget+box-office)
  owns "blockbuster"/"indie"; Cat 32 absorbs experiential compounds.

### Modifier absorption preserves meaning (atomicity test)

A token is absorbed as a modifier (not promoted to its own trait) if
it can't classify independently OR if splitting it off would damage
searchability. Examples:

- "lone female protagonist" → ONE trait (Cat 9 FEMALE_PROTAGONIST).
  "Lone" alone isn't a meaningful query span — it's a modifier on
  "female protagonist."
- "starring Tom Hanks" → ONE trait. "Starring" is a role-marker
  modifier, not its own trait.
- "not too dark or sad" → "not" applies to BOTH "dark" and "sad" via
  distribution, even though "not" is not adjacent to "sad" in the
  surface form.

### Interpretation deferral

Step 2 does NOT resolve "like Inception," "comedians doing serious
roles," "modern classics," etc. inline. Those stay as traits-with-
context. The category routing (e.g. Cat 41 like-X, Cat 43 catch-all)
sends them to handlers that have full per-handler context to unpack.

---

## Schema design reasoning

### Why a span_analysis pre-pass before the trait list

Small LLMs fail in characteristic ways on this kind of task:

1. **Skipping segmentation.** Going straight to classification while
   reading produces conflated decisions.
2. **Conflating attributes.** Deciding role and category in the same
   gestalt judgment — once one is committed it constrains the other,
   even when the evidence doesn't support both.
3. **Hallucinating modifiers as traits.** Promoting "ideally,"
   "starring," or "lone" to standalone traits.
4. **Picking the wrong category at boundaries.** 43 categories, with
   many semantically adjacent (Cat 3 vs Cat 5, Cat 7 vs Cat 8,
   Cat 12 vs Cat 42).

The pre-pass forces explicit segmentation reasoning *before*
classification. By the time the LLM commits to a final trait list,
each span has been examined for boundaries, modifiers, contextual
dependency, and split necessity.

### Why one trait list, not separate carver/qualifier lists

A single `traits` list with `role` as a per-trait field is preferred
over `carvers: [...]` and `qualifiers: [...]`. Three reasons:

- **No premature commitment.** Two-list shape forces role to be
  decided before placement, which inverts the natural reasoning order
  (category shape is a strong prior on role; role should be derived
  *after* category, not before).
- **Localized decisions.** Small LLMs are stronger at atomic per-trait
  classification than at global "which bucket" framing.
- **Handler dispatch doesn't care.** Step 3 dispatches on category,
  not role. Two lists is denormalization that buys nothing.

### Why structured `category_candidates` over freeform reasoning

Project learning: structured-field reasoning beats freeform strings
for small LLMs. `category_candidates` is a list of `{candidate, fits,
doesnt_fit}` objects so the LLM is forced to articulate the boundary
comparison between genuine contenders rather than emitting a vague
"this is a Cat X because it talks about Y" rationalization.

Rules for the list:
- Cap length (~2-4 entries). Don't enumerate every plausible cat —
  compare the genuine boundary contenders.
- Even single-candidate cases emit one entry. Keeps the schema
  uniform; small LLMs benefit from the consistency.

### Why no `role_cue` / `role_evidence` field

Reasoning fields that sit *after* the decision they justify drift
into post-hoc rationalization rather than pre-decision evidence.
Reasoning fields that sit *before* the decision must be evidence-
shaped, not decision-shaped.

The role decision is informed by:
- `query_context` and `contextual_dependency` from `span_analysis`
  (does this trait stand alone, or is it relativized by another?)
- `purpose_in_query` on the trait itself (carving vs qualifying vs
  reranking)
- Category prior (some categories are inherently carver-shaped,
  others qualifier-shaped)

All of those sit upstream of `role` in the schema. No additional
per-trait justification field is needed — adding one would either
restate upstream evidence or rationalize a committed decision.

### Field order encodes reasoning order

Within each trait:
1. `query_phrase` — what we're classifying
2. `modifiers` — what's been absorbed
3. `purpose_in_query` — what role this trait plays in the query (informs
   role-shape and constrains plausible categories)
4. `category_candidates` — boundary comparison reasoning
5. `best_fit_category` — committed pick, informed by candidates
6. `role` — informed by purpose_in_query
7. `polarity` — mechanical from modifiers
8. `salience` — mechanical from modifiers + position; null if carver

Each field appears once, in the location where it does the most work,
and ahead of any decision it informs.

---

## Final output schema

```python
{
  span_analysis: [
    {
      text: <span as it appears in query>,
      modifiers: [
        # Words inside the span OR elsewhere in the query that apply to
        # it. Includes non-adjacent applicability (e.g. "not" applies to
        # "sad" in "not too dark or sad" via distribution).
        <string>, ...
      ],
      query_context: <how other (non-modifier) parts of the query shape
                      this span's meaning, and/or how this span shapes
                      others. Empty when the span stands alone.>,
      decomposition: {
        possible_splits: [
          {
            isolated_span: <text that would be split out of the
                           parent span>,
            reasoning: <whether this isolated_span can stand as its
                        own query — would a user submit just this
                        piece and have it make sense, does it have
                        a category home, would pulling it out leave
                        a meaningful residual or an orphan>,
            should_split_out: <bool — final commit derived from
                              the reasoning above>
          },
          ...  # empty list when the span has no plausible split
               # points (e.g. "Tom Hanks")
        ]
      }
    },
    ...
  ],

  traits: [
    {
      query_phrase: <the trait text, modifiers excluded>,
      modifiers: [
        {
          text: <modifier word>,
          impact: <how it changes the trait's meaning in the query>
        },
        ...
      ],
      purpose_in_query: <reasoning about what role this trait plays in
                        the context of the query — carving the eligible
                        pool, qualifying within it, or reranking.
                        Grounded in query_context from span_analysis.>,
      category_candidates: [
        {
          candidate: <category enum, 1-43>,
          fits: <why this category could be the right home>,
          doesnt_fit: <why it might not be — boundary concern, missing
                      surface form, etc. Empty if no real concern>
        },
        ...  # cap ~2-4; even unambiguous traits emit one entry
      ],
      best_fit_category: <enum, 1-43>,
      role: <carver | qualifier>,
      polarity: <positive | negative>,
      salience: <central | supporting | null if carver>
    },
    ...
  ]
}
```

### Flow

`span_analysis` produces N entries (one per candidate span).
`traits` produces M entries, where M ≥ N when spans split via
`decomposition`. Each trait's `modifiers` and `purpose_in_query`
inherit from the corresponding span_analysis entry, narrowed to the
trait's subset after any decomposition splits.

---

## What step 2 does NOT emit (explicitly out of scope)

Captured here to prevent scope creep:

- **Per-query carver/qualifier balance state.** Followup LLM,
  parallel to step 3.
- **Per-query implicit-prior strength state.** Followup LLM,
  parallel to step 3.
- **Mode-word detection / scope resolution.** Step 3 / handler
  responsibility.
- **Tonal-negation rewriting.** Downstream code, triggered by
  polarity=negative + category=tonal-semantic.
- **Same-space qualifier fusion.** Programmatic, post-grouping.
- **Inter-trait relationship metadata.** Not in v1.
- **Trait-extraction confidence.** Treated as truth.
- **Interpretation of vague compounds.** ("Like Inception," "comedians
  doing drama," etc. stay as traits-with-context for handler-stage.)

---

## Open items / TODOs

- **Modifier-token registry.** Curated list of polarity setters,
  salience hints, role markers, range/intensity modifiers the prompt
  can reference. Some implicit in current step-2 prompt; consolidate.
- **Atomicity test worked examples.** The prompt needs a tight set of
  examples covering modifier absorption, decomposition splits, and
  Bucket-B compounds (single concept, handler fans out).
- **Salience heuristics.** Concrete language for central vs
  supporting (headline wants, structural prominence, hedge tokens).
- **`category_candidates` length cap and prompt instruction.** Need
  to land on the exact wording that prevents enumeration ("list
  genuine contenders, skip obvious non-fits").
- **Followup LLM design.** Schema, prompt, and exact inputs for the
  per-query state assignments LLM that runs parallel to step 3.

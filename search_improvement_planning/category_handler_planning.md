# Category Handler Planning

Finalized decisions from the category-handler design conversation.
Scope: how each category in [query_categories.md](query_categories.md)
translates into runnable code — what the handler looks like, what it
returns, and how the orchestrator composes handlers into a final
candidate set.

---

## The four handler types

Every category handler uses one of four orchestration shapes. The
shape governs how many endpoint queries can fire for that handler
on a given user request. Shapes fall into two families — shapes
that cap the handler at a single query, and shapes that allow
multiple queries to fire.

### At most one query fires

#### Type 1 — Single endpoint
Only one endpoint is ever applicable to the category. No routing
decision to make at query-generation time. This is the degenerate
case; the runtime codepath is identical to the other handler types
and returns the same four buckets.

Covers: Cats 1, 2, 3, 4, 5, 8, 9, 10, 13, 20, 22, 24, 27, 29, 30, 31.

#### Type 2 — Mutually exclusive
Two (or more) endpoints each individually capable of answering the
question, but they answer *different versions* of it. The
query-generation LLM picks whichever matches the user's framing and
ignores the others. Firing both would mix answers to different
questions rather than reinforce a single one.

Covers: Cats 11, 12.

#### Type 3 — Tiered
An ordered preference list of endpoints. The LLM walks the list at
query-generation time and fires whichever is the first *genuine fit*
for the user's phrasing. Earlier tiers are authoritative when they
apply; later tiers are fallbacks for cases the earlier tiers can't
cleanly express (typically spectrum-framed or long-tail asks outside
the canonical vocabulary).

Covers: Cats 6, 7, 14, 15, 16, 21, 26.

### More than one query may fire

#### Type 4 — Combo
Multiple endpoints apply to the same request and each carries
distinct, complementary signal that can't be collapsed into a single
call. All applicable endpoints fire in parallel; their outputs
populate the handler's return buckets. Reserved for categories where
forcing a single endpoint would drop real signal — either because no
single endpoint's data shape fully covers the question, or because
the question is inherently multi-faceted (e.g. seasonal intent spans
proxy tags, watch context, and narrative setting at once).

Covers: Cats 17, 18, 19, 23, 25, 28.

### Why gate + inclusion isn't a fifth type
An earlier "gate + inclusion" shape (for maturity / sensitive-content
gating, Cats 17 and 18) collapsed into Type 4 (Combo) plus the
`exclusion_ids` return bucket. The gate is not an orchestration
shape — it's a cross-cutting exclusion that applies to the final
candidate pool. Whether an endpoint's output lands in
`inclusion_candidates` or `exclusion_ids` is a bucket-population
choice driven by the query, not a separate runtime pattern.

### Why interpretation-first isn't a fifth type
An earlier "interpretation-first" shape (for Cats 28 and 31)
collapsed too. Interpretation is not a separate LLM stage — it's
extra schema fields the handler's existing LLM fills inline (intent
rationale, confidence, list-meaning decoding). No added latency, no
separate codepath.

---

## Modular handler construction

Handlers should not be hand-designed per category. They should be
assembled from composable pieces keyed off (a) the handler type
bucket and (b) the specific category being handled. Context that is
*inherent to an endpoint* (what keywords exist and what they mean,
what vector spaces are available and what each represents, what
Metadata columns can be predicated on, etc.) doesn't vary from
category to category — what varies is *which* endpoints are in
scope for a given category. So the static unit of reuse is the
endpoint-context chunk, dynamically appended to a system prompt
based on the category's endpoint set.

### The four plug-in pieces

Each handler's LLM prompt is composed of four pluggable pieces:

| Piece | Keyed by | Purpose |
|---|---|---|
| **Endpoint context** | Category (→ endpoint set) | Static per-endpoint reference: vocabulary, schema, parameter shapes, capabilities. Appended once per relevant endpoint. |
| **Output schema** | Bucket + category | What the LLM must produce. Bucket determines *shape* (one-of vs. best-fit vs. fan-out-set); category determines fields (which endpoints the schema includes, what hard-constraint sub-fields may appear). |
| **Core objective** | Bucket | The decision framing — what question the LLM is being asked to answer. Uniform across all categories in the same bucket. |
| **Additional objective notes** | Category | Category-specific nuances layered on top of the core objective: tiering bias orderings, when to emit `exclusion_ids`, interpretation/confidence guidance, etc. |

### Core objective per bucket

The LLM's decision-framing is bucket-level. Category-specific
notes adjust tone and nuances but don't change the core shape.

**Single endpoint.** "Given the user's query, decide whether this
endpoint *should* fire. Dispatch may have been wrong — answering
'no query needed' is a valid outcome and is preferable to inventing
a query that doesn't match the user's intent. If it should fire,
fill in the endpoint parameters so the query best satisfies the
attribute under consideration."

**Mutually exclusive.** "Given the user's query and the candidate
endpoints, decide which one best fits — *or* that none of them
should fire. Dispatch may have been wrong and 'none' is a valid
outcome. Once an endpoint is chosen, fill in its parameters so the
query best satisfies the attribute under consideration."

**Tiered.** "Given the user's query and the candidate endpoints,
decide which single endpoint best fits — *or* that none should
fire. You are provided with a preference bias favoring certain
endpoints when they can adequately satisfy the problem. Treat the
bias as a tiebreaker when multiple endpoints fit roughly equally;
do not treat it as an uphill battle for lower-preference endpoints.
A clearly-better lower-preference endpoint wins (e.g. 'lovable
rogue' has no canonical keyword tag, so Semantic beats Keyword
decisively despite the keyword-first bias). Once an endpoint is
chosen, fill in its parameters."

**Combo.** "Given the user's query and the full set of candidate
endpoints for this category, determine which *combination* of them
best covers the attribute under consideration. Any subset is valid
including the empty set (none fire). For each chosen endpoint, fill
in concrete parameters."

### Why tiered is mutually-exclusive-with-bias, not a walk-the-list

Tiered looks like "try tier 1, fall back to tier 2 if needed" but
in practice a single LLM call can evaluate all candidates jointly
and pick the best fit given a preference bias. That's operationally
identical to mutually-exclusive plus a bias prior on the scoring.
It's also strictly better than a sequential walk: the LLM sees all
options at once and can make a holistic judgment (is tier 1 a
*genuine* fit, or is tier 2 clearly a better match?) rather than
greedily taking the first "good enough" tier.

The bias exists to prevent the failure mode of "Semantic can
plausibly answer anything, so the LLM picks it even when a
canonical tag would be a cleaner hit." The bias puts a finger on
the scale toward the authoritative channel without locking out
the fallback for cases where the fallback is genuinely better.

### Why this modularity matters

Writing 31 bespoke handler prompts would leak category-specific
detail into endpoint-specific context and vice versa. The split
keeps each piece single-sourced:

- Endpoint vocabulary updates once per endpoint, not once per
  category that references it.
- Bucket core-objective changes (e.g. tightening the "none is
  valid" guidance) propagate to every category in the bucket.
- Category-specific quirks live in the "additional objective
  notes" chunk and nowhere else.

---

## Full system-prompt composition

The four plug-in pieces above (endpoint context, output schema, core
objective, additional objective notes) are the *variable* parts —
what changes per category or per bucket. A complete handler system
prompt also needs four supporting pieces that are either shared
across all handlers or mechanically required for the LLM to produce
useful output.

| Piece | Keyed by | Purpose |
|---|---|---|
| **Input spec** | Shared | Describes what the handler receives on each call: the stage-2 fragment, the category label, and any upstream metadata already resolved (normalized entities, IDs, prior-stage decisions). Without this pinned down, each handler drifts in what it assumes is given vs. inferable. |
| **Shared vocabulary block** | Shared | Single-sourced definitions of `action_role`, `polarity`, and `fit_quality`. Must be byte-identical across all four bucket prompts; loaded from one markdown file at build time. Does *not* reference return buckets or other bucket types — the handler doesn't need to know its bucket exists. |
| **Few-shot examples** | Bucket + category | Calibration examples, especially for the no-fire case (`should_run_endpoint: false` / `endpoint_to_run: None`) and close-call tiered bias decisions. These are the failure modes where LLMs drift silently. |
| **Failure-mode guardrails** | Bucket | Explicit guidance for ambiguous fragments, fragments that fit no endpoint, and self-contradictory fragments. The desired behavior is an honest no-fire with reasoning, not hallucinated parameters. |

Combined with the four plug-in pieces, a handler's system prompt is
built from eight composable chunks: four variable (per category /
per bucket) and four supporting (three shared across all handlers,
one tuned per bucket). Few-shot examples and guardrails live at the
bucket level because the decision shape is bucket-level; category
specifics enter only through additional objective notes and the
endpoint context already in scope.

### Why the supporting pieces aren't "just prose"

The temptation is to fold input spec and shared vocabulary into the
core-objective string. Don't. Those pieces update on a different
cadence than the objective (vocabulary stabilizes; input spec
changes when stage 2 changes; objectives change when decision
framing changes) and factoring them separately keeps each chunk
single-sourced with the rest of the codebase it mirrors.

---

## Prompt assembly

Finalized decisions about how the eight chunks above are assembled
into an actual system prompt + user message pair. Grounded in
2026 prompt-engineering consensus for small / instruction-tuned
models (GPT-5.4 mini, Gemini 3 Lite, etc.).

### Chunk ordering

System prompt, top to bottom:

1. **Role** — 1–2 sentences framing the handler's identity.
2. **Shared vocabulary** — terms (`action_role`, `polarity`,
   `fit_quality`, `dealbreaker` vs. `preference`) that later
   sections reference.
3. **Endpoint context** — the handler's "tools," one static chunk
   per endpoint in its set.
4. **Core objective + additional objective notes** — the bucket
   instruction with category nuance (see blending below).
5. **Failure-mode guardrails** — explicit no-fire guidance for
   ambiguous, unfit, or contradictory target entries.
6. **Few-shot examples** — wrapped in `<example>` tags.

User message:

7. **Input payload** — `raw_query`, `overall_query_intention_exploration`,
   `target_entry`, `parent_fragment`, `sibling_fragments`, serialized
   as XML.

Rationale: role first primes mode; vocab before instruction resolves
ambiguity of terms the instruction references; endpoints presented
as "tools" before the objective so the objective can refer to them;
constraints and examples late for recency weighting; input in the
user message keeps per-call content freshest. Output schema does
*not* appear in the prompt — it is enforced by Pydantic via
`.chat.completions.parse()`.

#### Finalized section table

Single reference for what each section is and what drives its
content. Core objective and additional objective notes are rendered
as two adjacent labeled sections (see "Additional objective notes:
labeled section, not blended" below), so they appear as distinct
entries here. Few-shot examples are authored per category only —
there is no bucket-level example bank.

| # | Section | Keyed by |
|---|---|---|
| 1 | Role | Shared |
| 2 | Shared vocabulary | Shared |
| 3 | Endpoint context | Per endpoint (assembled from the category's endpoint set) |
| 4 | Core objective | Per bucket |
| 5 | Additional objective notes | Per category |
| 6 | Failure-mode guardrails | Per bucket |
| 7 | Few-shot examples | Per category |
| 8 | Input payload *(user message)* | Per call |

### Field-level semantics live on the Pydantic model, not in the prompt

Per-field guidance — what each output field represents, how the LLM
should reason through reasoning fields like `requirement_aspects`,
`performance_vs_bias_analysis`, and `overall_endpoint_fits` — lives
as `Field(description=...)` on the Pydantic model. Pydantic
serializes those descriptions into the JSON Schema passed to
`.parse()`, and both OpenAI and Anthropic surface them to the model
as inline per-field instructions.

Why the schema is the right channel, not the prompt:

- **Attention anchoring.** When the model emits `requirement_aspects`,
  the description is co-located with the field — no need to recall
  prompt instructions from elsewhere.
- **Single source of truth.** Prompt-restated schemas drift from the
  real Pydantic model as fields evolve. Descriptions on the model
  can't drift from themselves.
- **Token savings.** No duplication between prompt and schema.
- **Industry convention.** OpenAI's structured-output guidance and
  the Instructor library are both built around this pattern.

Treat the Pydantic model as the *semantic* contract — field
meanings, reasoning guidance, enum-value interpretation — and the
prompt as the *procedural* contract (what task to do, in what
order, with what inputs).

### Input serialization — XML tags

Input payload is rendered as XML:

```
<raw_query>...</raw_query>
<overall_query_intention_exploration>...</overall_query_intention_exploration>
<target_entry>
  <captured_meaning>...</captured_meaning>
  <category_name>...</category_name>
  <fit_quality>...</fit_quality>
  <atomic_rewrite>...</atomic_rewrite>
</target_entry>
<parent_fragment>
  <query_text>...</query_text>
  <description>...</description>
  <modifiers>...</modifiers>
</parent_fragment>
<sibling_fragments>
  <fragment>...</fragment>
  ...
</sibling_fragments>
```

Rationale: (a) works cleanly across providers (Anthropic actively
recommends XML structuring; OpenAI and Gemini parse it reliably);
(b) hierarchical Step 2 data nests naturally; (c) small models show
measurably better parse reliability on tagged content vs. dense
JSON because tag boundaries are attention-salient; (d) the handler
can cite sections literally in its visible reasoning
(`per <target_entry>…`); (e) escaping is trivial.

### Shared vocabulary delivery

Single-source the shared-vocabulary block in one markdown file
(e.g. `handler_prompts/shared_vocabulary.md`) loaded verbatim at
prompt-build time for every handler. Physical include beats
hand-synced copies; drift risk is zero.

Scope of the block: definitions of `action_role`
(`candidate_identification` vs. `candidate_reranking`), `polarity`
(`positive` vs. `negative`), and `fit_quality` (`clean` vs.
`partial` — `no_fit` never reaches the handler).

`fit_quality` defines an **incoming signal from Step 2**, not a
criterion the handler computes. The vocab block should frame it as
context the handler takes into consideration — not as gospel. The
handler can still override Step 2's verdict if its own reasoning
over the endpoints says otherwise.

- `clean` → Step 2 judged this category a full cover for the atom.
  Usually handle the atom fully, unless the handler's own analysis
  suggests otherwise.
- `partial` → Step 2 judged this a partial cover; another entry on
  the same fragment covers the remainder. Guiding signal to scope
  the response to your slice, but not a hard lockout.
- `no_fit` → filtered out in code at dispatch time. No LLM call
  fires for `no_fit` entries, so the handler never sees this value
  and the vocab block does not need to guide its interpretation.

How well specific *endpoints* fit the atom is a separate
handler-internal judgment surfaced through the output schema's
`requirement_aspects` / `endpoint_coverage` fields, not part of the
shared vocab.

The block does *not* mention the four return buckets or the other
bucket types. The handler doesn't need to know its bucket exists —
only its own objective. Cross-bucket awareness just adds distraction.

### Endpoint-context chunks are static per endpoint

One chunk per endpoint, plugged in verbatim for every category that
reaches that endpoint. Endpoint capabilities (vocabulary, parameter
shape, caveats) don't vary by category; category-specific nuance
belongs in additional-objective-notes, never inside endpoint
chunks. Single source prevents drift when endpoint vocabulary
changes.

### Role preamble for small models

Keep it (1–2 sentences). Small instruction-tuned models (GPT-5.4
mini, Gemini 3 Lite) benefit more than frontier models from
explicit role framing because they infer less from task structure
alone. Don't over-invest — the structured output schema does most
of the heavy lifting — but the upside is cheap.

### Few-shot example count + shape

3–5 examples per handler, wrapped in `<example>` tags, with
format-consistent input/output pairs matching the XML input
schema and the bucket's Pydantic output schema exactly. Coverage
requirements:

- Every handler must include the **no-fire case**
  (`should_run_endpoint: false` or `endpoint_to_run: "None"`). This
  is the primary drift failure — LLMs default to fabricating
  parameters when they should abstain.
- Mutex / Tiered: include one example for each common
  `endpoint_to_run` outcome, plus a "None" case.
- Tiered: include one example where a lower-priority endpoint
  clearly wins despite the bias (prevents the bias from hardening
  into a lockout).
- Combo: include a "some fire, some don't" case to exercise the
  enumerated-breakdown shape.

Authoring: hand-write for launch, then swap in captures from real
runs as evaluation accrues them.

### Additional objective notes: labeled section, not blended

The core objective and additional objective notes are rendered as
two adjacent labeled sections, not interpolated into a single
paragraph.

Blending mechanism (rejected as default): core-objective template
holds a `{category_nuance}` placeholder and the category's notes
are inlined mid-sentence at build time, producing one coherent
paragraph. Tradeoff: reads more naturally to humans but
(a) harder to diff when notes change and (b) small models benefit
from explicit section boundaries, which give clearer attention
anchors. Revisit blending only if evaluation shows the labeled
form reads disjointed *to the model*.

### Reasoning strategy — start structured, iterate

Reasoning lives in the structured output (`requirement_aspects`,
`performance_vs_bias_analysis` for Tiered, `overall_endpoint_fits`
for Combo). No separate free-form scratch field for now. If the
structured-only form proves too shallow in practice (LLM skipping
depth when everything is constrained), add a pre-output reasoning
field later.

---

## Handler input data

Every category handler — regardless of bucket — receives the same
input payload on every call. The dispatch unit is the
`coverage_evidence` entry: one call per entry. The handler sees only
*its* target entry — not sibling entries that belong to other
categories — but does see the fragment that entry came from and the
sibling fragments that give cross-fragment context.

Schema source of truth: [schemas/step_2.py](../schemas/step_2.py).
Five fields are passed on every handler call:

### `raw_query` (str)

The user's query exactly as submitted. Preserves word order,
intonation, and typos that the structured Step 2 output flattens.
Not part of `Step2Response` today — added at dispatch time.

### `overall_query_intention_exploration` (str)

Step 2's 2–4-sentence gloss of what the query as a whole is asking
for, including overarching framing (occasion, audience, mood) that
colors specific requirements. Gives the handler the top-level
context its fragment sits inside.

### `target_entry` (CoverageEvidence)

The single `coverage_evidence` atom this handler is responsible for.
Exactly one per call. Fields:

- `captured_meaning` (str) — neutral one-sentence observation of the
  atom, stated before committing to a category label.
- `category_name` (`CategoryName` enum) — the category whose concept
  definition covers the captured meaning. Full enum (31 categories)
  defined in [query_categories.md](query_categories.md).
- `fit_quality` (`FitQuality` enum) — Step 2's verdict on how well
  this category covers the atom. Treat as context to weigh, not as
  gospel — the handler can override if its own reasoning over the
  endpoints diverges from Step 2. In practice the handler only ever
  sees `clean` or `partial`:
  - `clean` — Step 2 judged this a full cover. Usually handle the
    atom fully unless the handler's analysis suggests otherwise.
  - `partial` — Step 2 judged this a partial cover; another entry
    on the same fragment covers the rest. Guiding signal to scope
    the response to your slice, not a hard constraint.
  - `no_fit` — **filtered out in code at dispatch time**. No LLM
    call is made for `no_fit` entries; they are removed from the
    handler pool before any handler runs. Listed here for schema
    completeness only.
- `atomic_rewrite` (str) — the captured meaning expressed as a
  category-grounded request, preserving specifics from the original
  query (no generalization: `brother` must not become `sibling`,
  `1990s` must not become `older`).

### `parent_fragment` (RequirementFragment *without* coverage_evidence)

The fragment that produced `target_entry`. Carries `query_text`,
`description`, and `modifiers` but not the full `coverage_evidence`
list — the target entry is already broken out above, and any sibling
entries on this fragment belong to other handlers.

`modifiers` is the load-bearing field here: polarity and role
markers bound to the fragment drive the handler's `polarity` /
`action_role` decision. Each modifier has:

- `original_text` (str) — verbatim span.
- `effect` (str) — terse note on how the modifier shifts
  interpretation.
- `type` (`LanguageType` enum) — one of:
  - `POLARITY_MODIFIER` — flips or modulates sign/strength (`not`,
    `not too`, `without`, `preferably`, `ideally`).
  - `ROLE_MARKER` — binds the attribute to a role or dimension
    (`starring`, `directed by`, `about`, `set in`, `based on`).

### `sibling_fragments` (List[RequirementFragment *without* coverage_evidence])

Every other fragment in the query, with its `coverage_evidence` list
stripped. Gives the handler cross-fragment context (what else the
query asks for, ruled-out readings, other attributes in scope)
without leaking category-level atoms that are the business of other
handlers.

### Why `coverage_evidence` is scoped to the target only

Passing the full `coverage_evidence` list would invite the handler
to reason about atoms it isn't responsible for and emit decisions
that cross category lines. The dispatch unit is one entry per call
— the payload should reflect that. Sibling fragments still pass
through (their `query_text` / `description` / `modifiers` carry
cross-fragment signal) but without their category-level expansions.

### What is *not* in the input (for now)

No focus pointer. The handler is told, via its system prompt's
endpoint-context chunks, which category it is responsible for; the
`target_entry` field is pre-selected by dispatch so there is no
matching step on the handler's end. Dispatch routes on
`target_entry.category_name` (an enum), and per-category logic can
live on the enum class itself.

---

## Handler LLM output schema per bucket

The schema the category-handler LLM produces. Distinct from — and
upstream of — the handler-return contract below; this is what the
LLM emits, which the handler then translates into the four return
buckets.

Two classification fields appear inside `endpoint_parameters` for
every bucket that emits one:

- **`action_role`** — `candidate_identification` (the query drives
  which movies enter or leave the pool) or `candidate_reranking`
  (the query nudges scores up or down on an already-established
  pool; i.e. a preference).
- **`polarity`** — `positive` or `negative`. For
  `candidate_identification` this distinguishes inclusion from
  exclusion; for `candidate_reranking` it distinguishes adding from
  subtracting score.

These two fields map onto the four return buckets mechanically — see
the return-contract section below.

### Single endpoint

- **`requirement_aspects`** — list of discrete sub-requirements
  extracted from the fragment. Each has:
  - `aspect_description` — what the user is asking for.
  - `relation_to_endpoint` — capabilities of this endpoint that may
    be able to handle this aspect.
  - `coverage_gaps` — parts of this aspect the endpoint can't fully
    cover, or null.
- **`should_run_endpoint`** — whether the endpoint covers the
  requirement adequately enough to warrant executing a search.
- **`endpoint_parameters`** — endpoint-specific parameter shape
  (same format as the current `stage_3/` query-generation files),
  with `action_role` and `polarity` included as first-class fields
  inside this object rather than appended afterward. Left null when
  `should_run_endpoint` is false.

### Mutually exclusive

- **`requirement_aspects`** — list. Each has:
  - `aspect_description`
  - `endpoint_coverage` — how each candidate endpoint could cover
    this aspect.
  - `best_endpoint` — which endpoint best covers this aspect and
    why (brief).
  - `best_endpoint_gaps` — parts this endpoint can't fully cover,
    or null.
- **`endpoint_to_run`** — enum over the category's candidate
  endpoints, plus `None` when the gaps indicate nothing covers the
  requirement well.
- **`endpoint_parameters`** — same shape as Single endpoint
  (discriminated on `endpoint_to_run`). Left null when
  `endpoint_to_run == None`.

### Tiered

- **`requirement_aspects`** — same shape as Mutually exclusive.
- **`performance_vs_bias_analysis`** — short reasoning field that
  looks at each endpoint's fit and identifies whether there's a
  clear winner, or — if it's a close call — how the priority order
  breaks the tie.
- **`endpoint_to_run`** — same shape as Mutually exclusive.
- **`endpoint_parameters`** — same shape as Mutually exclusive.

### Combo

- **`requirement_aspects`** — list. Each has:
  - `aspect_description`
  - `endpoint_coverage` — how each candidate endpoint could cover
    this aspect.
- **`overall_endpoint_fits`** — bigger-picture reasoning: which
  endpoints fit, why, and how they interact to produce the best
  combined result.
- **`per_endpoint_breakdown`** — **not** a freeform list. Every
  candidate endpoint in the category is addressed explicitly. Each
  entry has:
  - `should_run_endpoint` — boolean.
  - `endpoint_parameters` — same shape as previous buckets. Left
    null when `should_run_endpoint` is false.

The enumerated (non-freeform) shape is deliberate: it forces an
explicit decision per endpoint rather than allowing the LLM to
quietly omit one, which is the failure mode when every endpoint is
independently optional.

### Building output schemas dynamically

The four bucket schemas above should not be hand-written per
category. They're assembled programmatically from (a) the category's
bucket and (b) its endpoint set, using Pydantic's `create_model()`
with `Literal` and discriminated `Union` types.

**Prerequisites**

1. **Canonical per-endpoint param models** — every endpoint already
   has (or needs) a single Pydantic model describing its parameter
   shape. These are the atomic building blocks the factories
   compose.
2. **Shared base class for `action_role` + `polarity`** — both
   fields live on a base class that every endpoint param model
   inherits from, guaranteeing they appear inside every
   `endpoint_parameters` without per-endpoint reimplementation.
3. **Category registry** — maps `category_name → (bucket_type,
   endpoint_set)`. The factory looks up both, picks the bucket
   builder, and returns a typed Pydantic class ready to feed into
   `.chat.completions.parse()`.

**One factory per bucket**

- `build_single_schema(endpoint)` — slots the endpoint's param model
  directly into `endpoint_parameters`.
- `build_mutex_schema(endpoints)` — `endpoint_to_run` becomes a
  `Literal` over `endpoint_names + ["None"]`; `endpoint_parameters`
  becomes a discriminated `Union` over the endpoint param models.
- `build_tiered_schema(endpoints)` — identical to mutex plus a
  single extra field (`performance_vs_bias_analysis: str`).
- `build_combo_schema(endpoints)` — each endpoint becomes a *named
  field* on the `per_endpoint_breakdown` sub-model (not a list
  entry), wrapping an `{should_run_endpoint, endpoint_parameters}`
  decision. This is the clearest win: the enumerated-not-freeform
  invariant is enforced by the type system instead of by
  prose-pleading in the prompt.

A top-level `build_handler_output_schema(category)` dispatches on
bucket.

**Watch-outs**

- **OpenAI structured-output JSON Schema constraints.** Discriminated
  unions require the discriminator field present in every variant;
  deeply nested `anyOf` and large schemas can hit size limits.
  Smoke-test the largest Combo category (3+ endpoints) against the
  API before committing.
- **Combo field keys must be valid Python identifiers.** Endpoint
  names used as named fields on the breakdown sub-model need to be
  sanitized once in the category registry — not at factory call
  time.
- **Cache factory output.** Each category's schema is deterministic
  given its bucket and endpoint set. Build once at process start
  and cache by category name; don't rebuild per request.

---

## Handler return contract

Every category handler returns the same four buckets:

| Bucket | Content | Merge op | Notes |
|---|---|---|---|
| `inclusion_candidates` | IDs + score | UNION across handlers, additive score | Drives candidate generation |
| `downrank_candidates` | IDs + score | Add (negative) score in final rerank | Semantic dealbreakers, soft "avoid" |
| `exclusion_ids` | IDs only, no score | Set-subtract from final pool | Hard dealbreakers (above PG-13, pre-1980, etc.) |
| `preference_specs` | Deferred query specs | Run after candidate consolidation | See preference-handling section |

Not every handler emits all four. Cat 10 (structured metadata) emits
`exclusion_ids` when the query implies a hard ceiling ("PG-13 max",
"under 2 hours"). Cat 17/18 emit `exclusion_ids` for maturity
ceilings alongside `inclusion_candidates` for family-friendly
scoring. Most purely-semantic categories emit only
`inclusion_candidates` and `preference_specs`.

### From LLM output to return buckets

The handler's LLM emits `action_role` + `polarity` inside every
`endpoint_parameters` object it fills in. The handler code
mechanically routes the resulting IDs/scores into one of the four
buckets based on that 2×2:

| `action_role` | `polarity` | → bucket |
|---|---|---|
| `candidate_identification` | `positive` | `inclusion_candidates` |
| `candidate_identification` | `negative` | `exclusion_ids` (hard) |
| `candidate_reranking` | `positive` | `preference_specs` (positive) |
| `candidate_reranking` | `negative` | `downrank_candidates` (soft dealbreaker) |

Only three of these four quadrants carry real load. A hypothetical
fifth quadrant — hard *inclusion* with INTERSECT semantics — is not
in the taxonomy today; see the Open Items note on
`hard_inclusion_ids`.

### Scoring conventions

Orchestrator-level decisions that shape how handlers' emissions
combine. These are *not* LLM-emitted.

- **Binary firing.** The LLM either emits an entry (with
  `action_role` + `polarity`) or it does not. No confidence score,
  no magnitude, no weight hint from the LLM — those fields invite
  fabrication. Weight assignment happens at the orchestrator layer
  and is deterministic.
- **Dealbreaker weighting.** Each firing dealbreaker
  (`downrank_candidates` entry or negative-polarity
  `preference_specs` entry) carries a weight in `[0.5, 1.0]`;
  weights sum across all firing dealbreakers.
- **Preference weighting.** All firing positive preferences share
  up to `0.49` of total preference weight, distributed equally.
  Keeps preferences from overriding identification-driven
  inclusion.
- **Inclusion scoring.** `inclusion_candidates` bring their own
  endpoint-computed scores (already normalized to `[0, 1]`) and
  contribute additively without separate weighting.

The `0.5` / `0.49` split is deliberate: dealbreakers outrank
preferences by construction, so in any conflict between them the
dealbreaker wins.

---

## Hard vs soft exclusion: why the split

The distinction is not about where the exclusion comes from — it's
about the merge op.

- **Hard exclusion** (`exclusion_ids`) — absolute remove. "Must be at
  least PG", "released after 2000", "under 3 hours". Applied as set
  subtraction on the final candidate pool.
- **Soft exclusion** (`downrank_candidates`) — downrank with a
  negative score. "Avoid gore" as a semantic ask that returns IDs
  with negative contribution to the final score.

**Correctness point: hard dealbreakers must go through
`exclusion_ids`, not `inclusion_candidates`, regardless of which set
is smaller.** The merge math forces the direction:

- Maturity-as-inclusion: Cat 17 emits {G, PG}; Cat 11 emits horror.
  Union = G/PG ∪ horror. R-rated horror stays in the pool (via Cat
  11) and just misses Cat 17's additive bonus. Violates "PG max".
- Maturity-as-exclusion: Cat 17 emits {R, NC-17} as `exclusion_ids`;
  Cat 11 emits horror. Union-then-subtract = horror ∩ not-R = family
  horror. Correct.

The "pick the narrower direction" framing still applies *within the
exclusion side*: "at least PG" → exclude G only (small set); "PG
max" → exclude R/NC-17 (larger but workable at 100K). The LLM picks
whichever framing yields the smaller ID set, but the answer is
always an `exclusion_ids` emission, never an inclusion flip.

### Post-hoc subtraction quality note

Applying `exclusion_ids` after endpoints return their top-K means
what's left is "the family-safe fraction of the top-500 horror
matches," not "the top-500 family-safe horror matches." At 100K this
is tolerable — bump endpoint fetch size a bit when large exclusions
are active. Optionally, pass `exclusion_ids` into each endpoint's
query at retrieval time so filtering happens during search. Either
works; the latter is slightly better signal for no latency cost.

---

## Preference handling: deferred, per-search, additive

Preferences never run at handler time. Handlers emit
`preference_specs` (target vector space + prose fragment + role +
polarity). The orchestrator consolidates all preferences across
handlers, establishes the final candidate pool, then runs
preferences against it.

### Separate searches, not concatenated prose

For a stack like "horror movies that are dark yet funny and playful",
three separate vector searches (one per axis) with additive score
merge is the right model:

- Separate + additive → **conjunctive** ("strong on all axes wins").
- Concatenated prose → **centroid target** ("strong on one axis can
  match the centroid"). Worse for preferences, which are
  conjunctive by nature.

### Zero-candidate fallback

If inclusion produced no candidates and we need preferences to
generate them, don't jump straight to a mega-query. First try:
**run each preference search against the full corpus, take top-K
from each, union into a working candidate pool (~K·N items), then
additively re-score.** Qdrant is cheap enough to absorb the extra
searches at 100K scale.

Mega-query concatenation is a tier-below-that fallback if even the
per-preference union produces nothing coherent — not the default.

### When would prose concatenation help?

When a single LLM pass with full query context would write better
joint prose than N disconnected per-category prose chunks (e.g.
"gut-punch ending in a dark slow-burn horror" vs three separate
axis-strings). That's a real quality win, but it fights the
conjunctive-merge property above. For preferences, conjunctive
merge wins; keep them separate. If future evaluation shows per-axis
prose quality is the bottleneck, the right fix is to give the
per-handler LLM more query context, not to concatenate prose at
search time.

---

## Within-handler query interaction

No handler needs its inclusion queries to interact at runtime beyond
what the shape's merge rule defines.

- Types 1–3 (single endpoint / mutually exclusive / tiered): only
  one query fires per request — no runtime interaction possible.
- Type 4 (combo): scores sum across endpoints — queries are
  independent by construction.

Endpoint-internal multi-phase logic (Franchise's two-phase token
resolution, Keyword's DF-ceiling) lives inside the endpoint, not
the handler shape.

---

## Orchestrator flow (summary)

```
Step 2 pre-pass
    ↓ (fragments + coverage evidence)
Category dispatch (categorizer LLM)
    ↓ (per-category fragment assignments)
Run all category handlers in parallel
    ↓ (each returns 4 buckets)
Consolidate:
  - Union all inclusion_candidates, additive score
  - Set-subtract exclusion_ids from the pool
  - Keep downrank_candidates as score contributions
    ↓ (final candidate set established)
Run consolidated preference_specs against the candidate set
  - One vector search per preference (separate, not concatenated)
  - Additive merge of preference scores
    ↓
Final rerank:
  inclusion_score + downrank_score + preference_score
    ↓
Top-N → fetch display metadata → return
```

---

## Pre-implementation setup

Structural decisions finalized before writing handler code. Per-
category content (wording of additional objective notes, few-shot
examples, etc.) is a later step.

### `HandlerResult` shape

The 4 return buckets as concrete Python types:

- `inclusion_candidates: dict[tmdb_id, score]` — map of movie ID to
  score contribution.
- `downrank_candidates: dict[tmdb_id, score]` — same shape; scores
  contribute negatively in the final rerank.
- `exclusion_ids: set[tmdb_id]` — no scores attached; pure set
  subtraction at orchestrator consolidation time.
- `preference_specs: list[EndpointParameters]` — the exact
  `endpoint_parameters` objects emitted by the handler LLM (one
  per deferred preference). The orchestrator routes each to the
  correct endpoint later by inspecting which concrete
  `EndpointParameters` subclass it is — no separate routing tag
  needed.

### `EndpointParameters` base class

Every endpoint's parameter payload is wrapped in a common shape
with three attributes:

1. The nested parameters themselves (endpoint-specific sub-model).
2. `action_role` (`candidate_identification` or
   `candidate_reranking`).
3. `polarity` (`positive` or `negative`).

Lives in `schemas/` (new module if no appropriate existing home).
Every endpoint's pure-payload model becomes the nested-parameters
field inside this wrapper. This is what the dynamic schema
factories slot into the output schemas' `endpoint_parameters`
field.

### Endpoint param model refactor — gap + plan

Per-endpoint Pydantic models exist today in
`schemas/*_translation.py` (keyword, metadata, semantic, award,
franchise, studio, entity) but not in a shape that plugs directly
into the new wrapper:

- **Reasoning fields are bundled with executable payload.** Models
  mix per-LLM-call reasoning (`concept_analysis`,
  `candidate_shortlist`, `constraint_phrases`, `value_intent_label`,
  `signal_inventory`, etc.) into the same class as the executable
  bits. **Decision: keep the reasoning fields in place for now.**
  They're tangled enough that a clean extraction is its own
  project, and it's worth confirming the new
  `requirement_aspects`-style reasoning captures everything the
  old fields did before we rip them out. Revisit once v2 handlers
  are running and we can diff behavior.
- **No common base for `action_role` + `polarity`.** The new
  `EndpointParameters` wrapper introduces them for the first time.
- **Semantic endpoint has two top-level shapes.**
  `SemanticDealbreakerSpec` (7 spaces, single, no weight) and
  `SemanticPreferenceSpec` (8 spaces, multi, weighted) are
  structurally different today. Under the new `action_role` /
  `polarity` model, dealbreaker-vs-preference is a runtime
  decision, not a schema-level split. **Resolved by unifying into
  one shape** — see "Unified semantic schema" below.

### Unified semantic schema

Single Pydantic model replaces `SemanticDealbreakerSpec` and
`SemanticPreferenceSpec`. Anchor is skipped in both paths
(dealbreaker and preference), so the space enum narrows to the 7
non-anchor spaces for everyone.

Shape:

```
SemanticSpaceEntry:
  carries_qualifiers: str        # reasoning for this space
  space: Literal[<one of 7 non-anchor spaces>]
  weight: PreferenceSpaceWeight  # central / supporting
  content: <space-specific body>

SemanticEndpointParameters:
  qualifier_inventory: str       # evidence inventory (reasoning)
  space_queries: conlist[SemanticSpaceEntry, min=1, max=7]
  primary_vector: DealbreakerSpace  # retrospective pick from space_queries
```

Field generation order is deliberate:
`qualifier_inventory` → `space_queries` → `primary_vector`.
Evidence inventory primes the list; the populated list anchors the
retrospective pick.

**Why `primary_vector` is placed last.** Generation order = field
order in Pydantic structured output, and each field anchors on
what came before. Placing `primary_vector` *before* `space_queries`
would force the model to commit to "the most effective space" up
front, then feel structurally pressured to keep the list short
(why list three when you already picked one). That's exactly the
bias we're trying to avoid — small models especially collapse
under that framing. Placing `primary_vector` *after* `space_queries`
reframes it as a retrospective summary judgment over an
already-populated inventory: the list gets filled honestly, then
the model picks the single most effective entry from among them.

The `Field(description=...)` on `primary_vector` should reinforce
the retrospective framing: "Among the spaces you populated in
`space_queries` above, identify the single most effective one.
Listing multiple genuinely-applicable spaces above is always
correct when multiple apply — this field collapses to one only
for execution paths that require a single target."

**Downstream execution** (deterministic; no conditional validator
on the LLM output):

- `action_role == candidate_identification`: use only the
  `space_queries` entry whose `space` matches `primary_vector`.
  Ignore `weight`.
- `action_role == candidate_reranking`: use all `space_queries`
  entries with their weights. Ignore `primary_vector`.

Symmetric shape, no runtime schema conditional. Each action role
ignores one field: the dealbreaker path ignores `weight`, the
preference path ignores `primary_vector`. Both fields are always
populated so the schema surface is identical regardless of role.

### Error handling

Every handler call retries **once** on failure (timeout, invalid
structured output, provider error). If the retry also fails, the
handler does not fail the whole request — it returns empty results
for inclusion/downrank/exclusion buckets, and preference specs
from a failed handler contribute 0 to all items (i.e., don't show
up in the preference rerank at all). Soft-fail by design: one
broken handler should not tank the whole query.

### Concurrency — fully parallel, no semaphore

All handler calls for a given query run concurrently. No
semaphore, no rate-limit guard at the dispatcher level. If
provider rate limits become an issue we revisit, but at ~5–10
coverage entries per query this is well within normal usage.

### Caching — none for now

Handler calls are not cached. If latency/cost becomes a concern
after we have real numbers, revisit. The existing per-query
`Step2Response` cache already absorbs the upstream work.

### Model defaults

Gemini by default (aligned with the other pipeline steps). Provider
and model are passed as kwargs into the handler function, so
switching to GPT-5.4 mini (or any other) is a call-site change,
not a code change.

### Category-handlers module layout (finalized)

All handler-execution code and prompt chunks live under
`search_v2/stage_3/category_handlers/`. The per-category output
schema factories and the endpoint-route → wrapper registry moved
into this module too, so every piece of the step-3 handler stack
sits in one place; `schemas/` is reserved for data contracts that
cross the handler boundary (e.g. `EndpointParameters` and its
per-endpoint subclasses).

```
search_v2/stage_3/category_handlers/
  __init__.py
  handler.py              # runs one category handler on one
                          # coverage_evidence entry: build prompt,
                          # call LLM, parse, execute/defer each
                          # endpoint_parameters, return a
                          # HandlerResult. Always scoped to a
                          # single category — fan-out across
                          # coverage_evidence entries lives one
                          # layer up in the orchestrator.
  prompt_builder.py       # assembles the 8-chunk system prompt +
                          # user message from markdown in prompts/
                          # keyed off category, bucket, and the
                          # category's endpoint set.
  handler_result.py       # HandlerResult dataclass — the four
                          # return buckets.
  schema_factories.py     # per-bucket Pydantic output-schema
                          # factories + get_output_schema(category).
                          # (Was schemas/handler_outputs.py.)
  endpoint_registry.py    # EndpointRoute → EndpointParameters
                          # wrapper map used by schema_factories.
  prompts/
    shared/
      role.md
      shared_vocabulary.md
    buckets/
      {single,mutex,tiered,combo}_objective.md
      {single,mutex,tiered,combo}_guardrails.md
    endpoints/
      {keyword,metadata,semantic,award,franchise,studio,entity}.md
    categories/
      cat_NN_<slug>/
        notes.md          # additional objective notes (§5)
        examples.md       # few-shot examples (§7) — always per
                          # category, never per bucket
```

`CategoryName.bucket` and `CategoryName.endpoints` already serve
the "category registry" role on the enum itself — no separate
registry module. The existing `*_query_execution.py` files in
`stage_3/` remain the endpoint-execution layer that `handler.py`
calls once the LLM has emitted parameters.

---

## Open items

- **Handler-level exclusion vs endpoint-level exclusion for
  `exclusion_ids`.** At 100K, post-hoc set subtraction is probably
  fine but produces lower-quality candidate pools when a large
  fraction of endpoint top-K gets kicked out. Revisit with real
  numbers once handlers are implemented.
- **Do we need `hard_inclusion_ids` (INTERSECT semantics)?** Not
  today. No category in the taxonomy requires "only these IDs, no
  others" positive-direction semantics. Every hard constraint we
  have is naturally expressible as an exclusion. Revisit if a new
  category breaks this.
- **Interpretation fields in handler output schemas.** Cat 28 and
  Cat 31 need intent-rationale and confidence fields on their LLM
  output so the orchestrator can lower confidence in the returned
  results. Design those schemas when those handlers are built.

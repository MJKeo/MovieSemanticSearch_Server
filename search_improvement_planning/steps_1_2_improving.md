# Steps 0 and 1 — Finalized Design

Finalized design for the query-understanding front half of the V2
search pipeline. Steps 0 and 1 run in parallel and together produce
the full set of branches that the standard-flow decomposition
pipeline (steps 2–4) will execute.

---

## Pipeline Shape (Renumbered)

The V2 search pipeline is re-shaped into five stages. Query
understanding is split across two parallel pre-processing stages
instead of fused into one:

0. **Flow Routing** — Classify the query into its plausible major
   flows and extract any flow-specific payload. LLM-based.
1. **Standard-Flow Intent Expansion** — Produce up to three
   standard-flow intents for the query (one fully faithful + up to
   two alternates or creative spins). LLM-based.
2. **Query Understanding (Decomposition)** — Per-intent dealbreaker /
   preference / quality-prior decomposition. Runs once per surviving
   standard-flow intent after the step 0 / step 1 merge.
3. **Query Translation & Execution** — Per-endpoint LLMs translate
   each decomposition into concrete retrieval actions and execute
   them.
4. **Assembly & Reranking** — Combine candidate sets, score, and
   return the final ranked list.

Steps 0 and 1 run concurrently. Their outputs are merged by code
into the final branch set fed to step 2.

---

## Step 0: Flow Routing

### Role

A narrow classifier. Step 0 decides which major flows the query
could plausibly belong to — `exact_title`, `similarity`, or
`standard` — and extracts the payload each non-standard flow needs
to execute.

Step 0 does **not** rewrite the query, decompose intent, assess
within-flow ambiguity, or propose alternate interpretations. Those
all live in step 1 or later.

### Output

The minimum data needed for downstream flows to run without any
additional LLM or complex logic. Conceptually:

```
flows: [
  { flow: "exact_title", title: "<extracted title string>" },
  { flow: "similarity",  reference_title: "<extracted title string>" },
  { flow: "standard" }
]
```

Rules for the output:

- Emit **1 to N** flows. Multi-flow emission is used when the query
  is genuinely ambiguous at the flow level (e.g. "scary movie",
  "date night" — each plausibly a literal title and a standard
  search).
- `exact_title` and `similarity` are mutually exclusive in practice
  — a phrase cannot simultaneously be a literal title and a
  "movies like X" request — so the realistic multi-flow shapes are
  `{exact_title, standard}` or `{similarity, standard}`.
- For `exact_title`, the extracted title string is **mandatory**.
  For `similarity`, the reference title string is **mandatory**.
  Step 0 has already reasoned about the phrase's title-ness, so
  the extraction is free here — no downstream LLM should have to
  re-derive it.
- For `standard`, no payload is emitted. Step 1 owns standard-flow
  decomposition, and passing a rewrite from step 0 would either
  duplicate or over-constrain it.

### Errors are symmetric — emit multiple flows on real ambiguity

Both false positives and false negatives are equally bad:

- A false positive (routing a non-title phrase to `exact_title`) ends
  in a dead "we don't have that title" response.
- A false negative (routing a real title phrase to `standard`) feeds
  a title into a pipeline that isn't designed for title lookup,
  producing fuzzy results where an exact match was expected.

Both are failures. The correct response to genuine flow-level
ambiguity is to emit multiple flows and let the user pick from the
resulting branches. The prompt should err toward emitting multiple
flows whenever the phrase could plausibly be read more than one way.

### Why step 0 is its own LLM call (not fused with step 1)

Flow-level ambiguity ("is this phrase a literal title?") and
within-flow ambiguity ("what does `millennial favorites` mean inside
a standard search?") are two different kinds of reasoning. Keeping
them in separate LLM calls lets each prompt stay narrow:

- Step 0 only reasons about a phrase's surface form vs. known-title
  plausibility. Bounded classification.
- Step 1 only reasons about standard-flow interpretation and
  exploration. Open-ended decomposition.

Small LLMs are more reliable on one-kind-of-reasoning prompts than on
multi-task prompts. Splitting here is the lever for making this
whole front half run on small models.

---

## Step 1: Standard-Flow Intent Expansion

### Role

Given the user's raw query, produce up to three **standard-flow
intents**. Step 1 is entirely flow-agnostic — it treats every input
as if it were headed to the standard pipeline, regardless of what
step 0 decides in parallel.

The three intents are ordered. The **first intent is always a
completely faithful rewrite** of the user's query. It preserves the
user's hard constraints, specificity, and wording intent without
inference, proxy traits, or narrowing. This is the guarantee that
every user sees results matching literally what they asked for.

The remaining two intent slots are filled with some combination of:

- **Alternate interpretations** — when the query has within-flow
  ambiguity and multiple readings would genuinely retrieve different
  movies (e.g. "disney millennial favorites" varying only what
  `millennial favorites` could mean, while keeping `disney` fixed).
- **Creative spins** — productive sub-angles or narrowings of the
  primary intent that propose something the user may not have
  thought of (e.g. "80s action classics" → "Schwarzenegger's 80s
  peak"). Spins always preserve the primary's hard constraints.

### Always three branches when standard flow executes

Step 1 always outputs exactly three ordered standard-flow intents.
The total branch budget across the whole pipeline is also three, so
when step 0 routes one or more branches to non-standard flows, code
trims step 1's list from the end to fit the remaining budget. See
"Branch Budget Merge" below for the exact trimming rule.

The "always 3" rule exists so the branch shape is predictable:
whenever the standard flow runs at all, the user sees a faithful
result plus up to two complementary explorations. This collapses
gracefully to fewer total branches only when the step 0 classifier
eats into the standard-flow slots.

### Branch allocation by ambiguity tier

The mix of alternates vs. spins in slots 2–3 is driven by how much
within-flow branching pressure the query carries:

- **Clear** — one dominant reading, no real within-flow ambiguity.
  Slots 2–3 are **creative spins** narrowing the primary into
  interesting sub-angles.
  *Example:* "Tom Cruise 80s movies" → faithful primary + "Tom
  Cruise 80s thrillers" + "Tom Cruise 80s action peak".

- **Moderate** — one dominant reading plus at least one useful
  adjacent interpretation. Slot 2 is an **alternate
  interpretation**, slot 3 may be another alternate or a spin.
  *Example:* "80s action classics" → faithful primary +
  "Schwarzenegger's 80s action peak" (alt interpretation of
  `classics`) + "Stallone's 80s run" (spin).

- **High** — multiple strong readings, or the query is vague enough
  that several distinct searches are clearly better than forcing one
  interpretation. Slots 2–3 are **alternate interpretations**; no
  spin headroom.
  *Example:* "I need to feel something" → faithful primary + one
  catharsis-oriented interpretation + one adrenaline-oriented
  interpretation.

The tier is not a confidence score — it's a summary of how much of
the budget is spent resolving ambiguity vs. generating adjacent
exploration. The primary is faithful in every tier.

### Flow-agnostic operation and raw-query caching

Step 1 runs on the user's raw query every time, with no input from
step 0. Its output is identical for identical raw queries, so it
can be cached on a raw-query hash regardless of step 0's routing
decision.

Step 1's intents are discarded for that query only if step 0 emits
zero standard-flow slots (e.g. an unambiguous `exact_title` like
"Interstellar"). In practice this is a minority of queries, the
wasted step 1 call is on a small model, and the cache hit rate on
re-queries absorbs the rest. This is an explicit, accepted cost
paid in exchange for parallel-execution latency (see next section).

---

## Parallel Execution

Step 0 and step 1 run **in parallel**, both operating on the user's
raw query. The merge happens in code after both return.

### No semantic dependency justifies serializing them

The only conceivable runtime dependency between step 0 and step 1
is budget-based trimming of step 1's output. That's fully handled
post-hoc by code (see "Branch Budget Merge") and does not require
either LLM to see the other's result.

Step 1 does not need step 0's flow classification to produce valid
standard-flow intents: its job is to treat the raw query as if the
standard flow will execute, and the intents it produces are
meaningful regardless of whether standard actually runs.

### Latency vs. wasted-call tradeoff

Running in parallel saves ~1–2 seconds on every query (one LLM
round trip instead of two). The cost is that step 1 runs even when
step 0 ends up emitting no `standard` slot, making step 1's output
entirely discarded on those queries.

The tradeoff is accepted:

- Pure-title and pure-similarity queries are a minority of overall
  traffic.
- Step 1 is a small-model call, so the wasted-call cost is low in
  absolute terms.
- The latency win applies to every query, not just the minority.

Expected value is clearly positive, and the latency hit matters for
user-perceived responsiveness on every search, not just the
minority that would save a call.

### Merge semantics

Step 0's flow classification is authoritative for **which flows
execute**. Step 1's output only fills `standard`-flow slots in the
final branch set. Step 1 cannot override step 0's decision to run
or not run the standard flow.

---

## Branch Budget Merge

Total branch budget across the whole pipeline is **three**.

After step 0 and step 1 return, code performs the merge:

1. Start with the flows emitted by step 0. Each non-standard flow
   (`exact_title` or `similarity`) occupies one branch slot and
   executes its own dedicated downstream path.
2. Count the remaining budget: `remaining = 3 - non_standard_count`.
3. If step 0 emitted `standard` as one of its flows, take the
   **first `remaining` intents** from step 1's ordered output. The
   first slot is always step 1's faithful primary — it is never
   trimmed away as long as standard runs at all.
4. If step 0 did **not** emit `standard`, discard step 1's output
   entirely and run only the non-standard flows.

Concrete cases:

| step 0 output                        | standard slots | final branches                                  |
|--------------------------------------|----------------|-------------------------------------------------|
| `{standard}`                         | 3              | step 1 primary + alt/spin #2 + alt/spin #3      |
| `{exact_title, standard}`            | 2              | exact_title lookup + step 1 primary + #2        |
| `{similarity, standard}`             | 2              | similarity search + step 1 primary + #2         |
| `{exact_title}`                      | 0              | exact_title lookup only (step 1 output dropped) |
| `{similarity}`                       | 0              | similarity search only (step 1 output dropped)  |

The "faithful primary first" ordering inside step 1 is the
guarantee that trims from the tail never remove the literal reading
of the query.

---

## Why This Shape

The redesign is driven by three constraints:

1. **Faithfulness floor.** Every user must see a result that matches
   exactly what they asked for, on every query. The faithful
   primary is the structural lever that enforces this.
2. **Browsing value.** Beyond the faithful result, the pipeline
   should present up to two more branches — either alternate
   interpretations (for ambiguous queries) or creative spins (for
   clear ones) — so browsing feels alive instead of one-shot.
3. **Small-LLM reliability.** Each LLM call in the pipeline should
   do one kind of reasoning. Splitting flow routing from intent
   expansion keeps both prompts narrow enough to run on small
   models without multi-task drift.

Running step 0 and step 1 in parallel preserves all three while
paying the smallest latency bill possible.

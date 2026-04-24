# Query Categorization Step — Planning

This is the interpretation step that sits between the raw user
query and endpoint dispatch. It takes a natural-language query
and emits a list of atomic, categorized requirements. Downstream
steps (endpoint-level query generation) then decide *how* each
requirement translates into concrete endpoint calls.

The category vocabulary is defined in `query_categories.md`
(30 categories + fallback). This doc covers the *structure of
the output* and the decisions that shaped it.

---

## What this step must do

For each user query, produce a list of **atomic requirements**.
Each requirement carries four fields:

1. **Category** — one of the 30 categories from `query_categories.md`,
   or Cat 31 (Interpretation-required) for true edge cases.
2. **Intent** — `dealbreaker` (candidate generation / hard filter)
   vs `preference` (preference sorting / soft scoring).
3. **Polarity** — `include` vs `exclude`. The requirement itself is
   always phrased as positive inclusion; this flag records whether
   the original query framed the trait positively or negatively.
4. **Hint** — a brief note explaining the reasoning behind the
   category choice, passed along as context to the endpoint-level
   LLMs that handle the requirement downstream.

---

## Decisions and why

### Compound splitting
Any phrase that seems to fit multiple categories is a signal to
split. The categorization step decomposes compound phrasing into
atomic requirements before categorizing. Compound descriptors
(e.g. "classic," "modern classic," "Disney classic") never get
their own category — they fan out into the categories that
already exist (see compound-split rule in `query_categories.md`).

### No bound payload at this stage
This step outputs the *category* and a brief *hint*, not a fully
bound value ({role: actor, name: "Tom Hanks"}). Payload binding
is the next step's job: dispatch code sends the requirement to
every endpoint associated with the category, and each endpoint's
LLM decides whether to fire and what to search for. The hint is
the only context the next step gets from this step's reasoning.

**Why:** keeps this step focused on classification. Endpoint-level
generation already has to reason about endpoint-specific schema;
double-binding here would duplicate work and constrain downstream
flexibility.

### Preferences are unweighted
All preferences are treated as equally weighted. No `strength`
axis.

**Why:** real queries rarely include explicit strength cues
("must have" vs "would be nice"). Users list traits; we infer.
Adding a strength axis would be speculative and rarely load-bearing.
Revisit if scoring shows preferences systematically over- or
under-dominating.

### Prominence / centrality is out of scope
Whether "Wolverine movie" means starring-role vs cameo is a
concern for the Cat 2 flow, not this step.

**Why:** prominence is category-specific (only meaningful for
entity-like categories) and the endpoint flows already handle it
natively. Pushing it up would bloat the schema with fields that
are null for most categories.

### OR is treated as AND
"Tom Hanks or Meg Ryan" produces two independent atomic
requirements, both preferences. Movies with both score highest,
movies with either score middle, movies with neither score
lowest — the ranking naturally expresses the OR semantics.

**Why:** the user wouldn't object to seeing a movie with both.
Modeling explicit OR would require logical-operator tracking
across requirements for no practical benefit; additive scoring
gets the same user outcome with a simpler schema.

### No interpretation confidence flag
The catch-all (Cat 31) already handles uncertainty with its
explicit "confidence-lowered phrasing" instruction. Per-requirement
confidence is skipped for now as unjustified complexity.

**Why:** adds a field that would have to be threshold-tuned
downstream for unclear benefit. Worth revisiting if we see
systematic over-confident misclassification in practice.

### Gradient negation collapses into polarity + preference
"Not too bloody" is `{category: 17, polarity: exclude, intent: preference}`.
"No gore" is `{category: 17, polarity: exclude, intent: dealbreaker}`.
Same polarity axis, different intent. Requirements are always
phrased in positive-inclusion form internally; the polarity flag
records the original sign.

**Why:** keeps the schema binary (no gradient-negation type) while
still letting downstream differentiate hard excludes from soft
tilts via the intent field.

---

## Output shape (conceptual)

```
[
  {
    category: int,            # 1–30
    intent: dealbreaker | preference,
    polarity: include | exclude,
    hint: str,                # brief reasoning trace for downstream
  },
  ...
]
```

Example — "Classic Tom Hanks comedies from the 90s, not too sappy":

```
[
  {category: 1,  intent: dealbreaker, polarity: include,
   hint: "actor credit: Tom Hanks"},
  {category: 11, intent: dealbreaker, polarity: include,
   hint: "top-level genre: comedy"},
  {category: 10, intent: dealbreaker, polarity: include,
   hint: "release era: 1990s"},
  {category: 25, intent: preference,  polarity: include,
   hint: "'classic' compound → canonical/acclaimed stature"},
  {category: 22, intent: preference,  polarity: exclude,
   hint: "'not too sappy' — tilt away from sentimental tone"},
]
```

Note "classic" got split: the era part lands in Cat 10 via the
"90s" atom already, and the acclaim part becomes its own Cat 25
preference. Compound descriptors never warrant their own category.

---

## What happens next (out of scope here)

Each requirement is dispatched to every endpoint associated with
its category (per `query_categories.md` fan-outs). The endpoint-
level LLM receives the original query, the requirement, and the
hint, and decides (a) whether this endpoint should fire for this
requirement, and (b) what concrete query to run. That's the next
planning doc.

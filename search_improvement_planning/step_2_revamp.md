# Step 2 Revamp

Working draft capturing the current redesign direction for query understanding.
This is a brainstorming-to-spec bridge, not a finalized ADR.

---

## Goal

Redesign Step 2 so it can represent one user concept through multiple retrieval
expressions without forcing that concept into a single endpoint route.

The current Step 2 shape is too rigid because it assumes:

- one extracted item maps to one route
- each route instance behaves like an independent scoring unit
- multi-source concepts are narrow enough to defer

Real queries exposed that this is not true in practice:

- `"Spider-Man movies"` should capture franchise membership and character-based
  membership
- `"Indiana Jones movie where he runs from the boulder"` can miss the right
  movie if we only express the concept through character lookup
- `"Christmas movies"` often needs broad recall plus stronger ranking for
  conventional, central Christmas stories
- terms like `"classic"` are being misrouted as keywords even when they should
  stay semantic

---

## Finalized Decisions

### 1. Keep the high-level intent model, but move it down to the expression level

We still want the broad search notions of:

- dealbreakers
- preferences
- exclusions

But these do **not** live at the concept level anymore.

A single concept may need multiple expressions with different roles.

Example:

- `"Christmas movie"` can legitimately produce:
  - a keyword **dealbreaker**
  - a semantic **preference**

So Step 2A should not try to classify concepts as one fixed role. That
classification belongs to Step 2B's expressions.

### 2. Step 2 should reason in concepts, not route-bound items

A concept is one user intent such as:

- `"Spider-Man movie"`
- `"Christmas movie"`
- `"dark and gritty"`
- `"animated movie"` in an exclusion context

Each concept may need multiple retrieval expressions.

### 3. Retrieval expressions are unioned, not intersected

If something seems to require intersection across multiple sources, that is
usually a sign it should actually be represented as multiple concepts.

This preserves the desired ranking behavior:

- top-tier matches rise first
- broader near-matches remain available as the user scrolls

### 4. Multiple expressions under one concept do NOT become multiple full hits

A concept contributes one conceptual signal, not one full score contribution
per expression.

This remains a **code-level invariant**, not an LLM output field.

### 5. Step 2 should still choose exact route families

Step 2B should continue choosing route families directly:

- `entity`
- `studio`
- `metadata`
- `awards`
- `franchise_structure`
- `keyword`
- `semantic`
- `trending`

The `"classic"` problem is not an argument for looser routing. It is an
argument for better boundary discipline so Step 2B routes it to `semantic`
instead of incorrectly forcing it into `keyword`.

### 6. Step 2 should NOT commit to closed-vocabulary values

Step 2B should decide:

- which expressions exist for the concept
- which route handles each expression
- whether each expression is a dealbreaker or preference
- whether a dealbreaker expression is include vs exclude
- whether a preference expression is core vs supporting

It should **not** decide exact step-3 grounded values such as:

- keyword enum member
- metadata field value
- franchise canonical surface form

That grounding remains Step 3's job.

### 7. Retrieval expressions must always be written in positive-presence form

When Step 2B routes a dealbreaker toward Step 3, the expression description
must always describe the **presence** of an attribute, never its absence.

Examples:

- `"Disney movies not animated"` should produce a keyword dealbreaker
  expression described as `"movie is animated"` with exclusion handled by
  `dealbreaker_mode="exclude"`
- `"movies without Adam Sandler"` should produce an entity dealbreaker
  expression described as `"movie includes Adam Sandler"` with exclusion
  handled separately

This keeps endpoint translation direction-agnostic:

- description answers **what attribute are we testing for**
- inclusion / exclusion answers **how should that attribute be used**

`not`-wording should never appear inside dealbreaker descriptions.

Preferences are different: if the user phrases a preference negatively, Step 2B
must rewrite it into a positive ranking target before it reaches Step 3.

Example:

- `"not too long"` becomes a preference like
  `"prefer movies below a certain runtime"`

### 8. Exclusions are dealbreakers with exclusion polarity

We should think of exclusions as dealbreakers that remove or downrank rather
than as a separate kind of abstract concept.

At the Step 2B schema level, that means:

- `kind="dealbreaker"`
- `dealbreaker_mode="exclude"`

### 9. Step 1 stays interpretation-level; concept extraction moves to Step 2A

Do **not** move concept extraction into Step 1.

Step 1 should stay focused on:

- flow routing
- multi-interpretation branching
- standard-flow interpretation rewrites

It should not also be responsible for deciding concept boundaries.

Instead, split the current Step 2 into two substeps:

1. `Step 2A: Concept Extraction`
   One LLM sees the full standard-flow interpretation and emits the concept
   inventory.
2. `Step 2B: Expression Planning`
   One parallel call per concept turns that concept into the retrieval
   expressions to union.

### 10. Step 2A should be as simple as possible

Current preferred shape:

- one top-level `concept_inventory_analysis`
- `concepts: list[str]`

The analysis field is where the model should think through:

- what candidate concepts exist in the interpretation
- which ones are truly separate
- which ones should remain unified because they are alternate expressions of
  one user intent

Important note:

- if later evaluation shows that Step 2B is redoing too much concept-boundary
  work, the first thing to revisit is adding a lightweight per-concept boundary
  note
- current preference is to **not** add that yet

### 11. Step 2B should do expression planning, not system-internals planning

Step 2B should work in language the model is good at:

- concepts
- attributes
- routes
- dealbreakers
- preferences
- exclusions

It should **not** use system-internals language like:

- generates candidates
- scores only
- candidate pool logic

Those belong in code, not in the LLM contract.

### 12. Preference strength should be explicit: `core` vs `supporting`

Preference expressions should carry an explicit strength label from the LLM:

- `core`
- `supporting`

This should **not** be inferred from a simplistic structural rule like:

- "if a preference appears alongside a dealbreaker, it is automatically
  supporting"

That rule breaks on queries like:

- `"scariest zombie movies"`
- `"funniest Christmas movies"`
- `"best revenge movies"`

Those all contain a dealbreaker and still have a clearly dominant preference.

### 13. Boundaries for `core` vs `supporting`

`core` means the preference is the dominant ranking instruction for the query
or for that concept.

Examples:

- `"scariest movies ever"`
- `"funniest rom-coms"`
- `"darkest Batman movies"`
- `"most emotionally devastating breakup movies"`

`supporting` means the preference matters, but is a secondary refinement.

Examples:

- `"Spider-Man movies, preferably funny"`
- `"Disney movies not animated, preferably from the 90s"`
- `"Christmas movies that feel especially Christmasy"`

Rule of thumb:

- if one preference clearly dominates, it should be `core`
- if multiple preferences exist and none clearly dominates, default to
  `supporting`

### 14. No generic `boost` field in Step 2B

We considered a generic `boost` mechanism for sibling expressions inside a
concept and rejected it.

Reason:

- the meaningful use cases were not general concept-planning problems
- they were endpoint-specific notions of prominence

### 15. Endpoint-specific prominence belongs in Step 3, not Step 2B

Instead of a generic `boost` field, handle prominence inside the relevant
endpoint logic.

Current examples:

- character entity lookup should support character prominence preferences,
  analogous to actor prominence
- franchise lookup should support cases where lineage matches should be
  preferred over broader universe matches

This keeps:

- Step 2B focused on expression planning
- Step 4 free of new generic boost logic
- prominence close to the data that can actually judge it

### 16. Multiple expressions are still necessary

Removing generic `boost` does **not** mean one concept collapses to one
expression.

Some concepts still legitimately need multiple expressions.

Example:

- `"Spider-Man movies"` may still want:
  - franchise expression(s)
  - character expression(s)

Then Step 3 handles endpoint-native nuance:

- lineage vs universe within franchise
- prominent vs incidental presence within entity

### 17. Franchise remains one route family

Do not split franchise into multiple top-level endpoints.

The route remains `franchise_structure`.

Lineage-vs-universe preference is an endpoint concern, not a new Step 2B field.

### 18. "Mainline" is not a first-class stored concept today

Current stored franchise data includes:

- `lineage`
- `shared_universe`
- `recognized_subgroups`
- `launched_subgroup`
- `lineage_position`
- `is_spinoff`
- `is_crossover`
- `launched_franchise`

But there is no explicit `mainline` / `core trunk` field.

That means `"mainline"` cannot be treated as a clean deterministic filter yet.
It must currently be approximated through explicit query logic and endpoint
behavior, not hidden heuristic assumptions.

### 19. Spinoff / crossover handling should be explicit

Do not silently apply default demotions for franchise queries.

If the user explicitly implies non-spinoff intent, Step 2B can emit
dealbreaker exclusions or preferences that reflect that.

### 20. Quality / notability priors should be removed from the revamped Step 2

Current priors are considered flawed and should not be carried forward into the
new Step 2 design.

If priors return later, they should come back as a separate mechanism.

---

## Step 2A Responsibilities

Step 2A should:

- read the full interpretation rewrite from Step 1
- identify separable concepts
- preserve concept boundaries
- keep genuinely distinct concepts separate
- keep one concept unified even if it may later need multiple expressions

Step 2A should not:

- decide routes
- decide dealbreaker vs preference labels
- decide include vs exclude polarity
- decide exact step-3 grounded values
- use search-system-internals language

### Step 2A Output Principles

- Concepts are the stable unit of user intent
- Step 2A owns concept-boundary decisions
- Step 2B should not casually split one Step 2A concept into many

### Proposed Step 2A Schema

```python
class Step2AResponse(BaseModel):
    concept_inventory_analysis: str
    concepts: list[str]
```

### Step 2A Field Notes

#### `concept_inventory_analysis`

This is the main thinking field for Step 2A.

It should explicitly think through:

- what candidate concepts are present
- which concepts should remain separate
- which things should stay unified as one concept even if Step 2B later emits
  multiple expressions for them

This field is especially important for cases like:

- `"Spider-Man movie"` staying one concept despite multiple retrieval
  expressions
- `"Christmas movie"` staying one concept even if it later emits both a
  keyword dealbreaker and a semantic preference
- grouped tonal concepts like `"dark and gritty"` staying together when that
  better reflects user intent

#### `concepts`

Plain list of concept strings.

Examples:

- `"Spider-Man movie"`
- `"Christmas movie"`
- `"heartwarming family tone"`
- `"animated movie"`

---

## Step 2B Responsibilities

Step 2B should:

- take exactly one concept from Step 2A
- decide which expressions should exist for that concept
- choose the route family for each expression
- label each expression as a dealbreaker or preference
- mark dealbreakers as include vs exclude
- mark preferences as core vs supporting

Step 2B should not:

- emit exact step-3 grounded values
- phrase dealbreaker descriptions negatively
- emit code-level aggregation rules
- encode endpoint-specific prominence logic as generic fields

### Step 2B Inputs

Each Step 2B call should receive:

- full interpretation rewrite from Step 1
- the current concept string
- compact concept inventory from Step 2A

This preserves whole-query awareness while still narrowing the planning task.

### Proposed Step 2B Schema

```python
class EndpointRoute(str, Enum):
    ENTITY = "entity"
    STUDIO = "studio"
    METADATA = "metadata"
    AWARDS = "awards"
    FRANCHISE_STRUCTURE = "franchise_structure"
    KEYWORD = "keyword"
    SEMANTIC = "semantic"
    TRENDING = "trending"


class ExpressionKind(str, Enum):
    DEALBREAKER = "dealbreaker"
    PREFERENCE = "preference"


class DealbreakerMode(str, Enum):
    INCLUDE = "include"
    EXCLUDE = "exclude"


class PreferenceStrength(str, Enum):
    CORE = "core"
    SUPPORTING = "supporting"


class RetrievalExpression(BaseModel):
    routing_rationale: str
    route: EndpointRoute
    kind: ExpressionKind
    description: str
    dealbreaker_mode: DealbreakerMode | None = None
    preference_strength: PreferenceStrength | None = None


class Step2BResponse(BaseModel):
    concept: str
    expression_plan_analysis: str
    expressions: list[RetrievalExpression]
```

### Validation Rules

- `dealbreaker_mode` is required iff `kind == "dealbreaker"`
- `preference_strength` is required iff `kind == "preference"`
- `description` must always be written in positive-presence form

### Step 2B Field Notes

#### `expression_plan_analysis`

Top-level planning field for the concept.

This should think through:

- whether the concept needs one expression or several
- whether the concept needs both dealbreaker and preference expressions
- whether some expressions are broad recall while others are refinement

This is where relative planning across sibling expressions belongs.

#### `routing_rationale`

This should appear **before** the route and other expression decisions so it
acts as a pre-generation scaffold rather than a post-hoc justification.

It should answer:

- what kind of expression is this?
- why is this route appropriate?

Examples:

- `"broad deterministic holiday membership"`
- `"semantic refinement for centrality of the same concept"`
- `"named character expression for the same concept"`

#### `description`

Natural-language expression statement for the downstream Step 3 translator.

Dealbreaker descriptions must always describe the positive presence of the
attribute being tested.

Examples:

- `"movie is a Christmas / holiday movie"`
- `"Christmas is central to the story"`
- `"movie includes Spider-Man as a named character"`
- `"movie is animated"` for an exclusion dealbreaker

#### `kind`

Whether the expression functions as:

- a `dealbreaker`
- a `preference`

This lives at the expression level, not the concept level.

#### `dealbreaker_mode`

Only used for dealbreakers.

- `include`
- `exclude`

This is how exclusions are represented without creating a separate abstract
expression kind.

#### `preference_strength`

Only used for preferences.

- `core`
- `supporting`

This lets the LLM explicitly state whether a preference is the dominant ranking
instruction or a smaller refinement.

---

## Complex Query Checks

These examples are here to pressure-test the schema rather than define exact
prompt wording.

### Example 1: `"Heartwarming Christmas movies for the family, not animated, where Christmas is central to the story"`

Likely Step 2A concepts:

- `"Christmas movie"`
- `"heartwarming family tone"`
- `"animated movie"`

Likely Step 2B plan for `"Christmas movie"`:

- keyword dealbreaker include:
  `"movie is a Christmas / holiday movie"`
- semantic preference supporting:
  `"Christmas is central to the story"`

This is an important proof case because one concept contains both a dealbreaker
and a preference.

### Example 2: `"Spider-Man movies, preferably mainline, not animated"`

Likely Step 2A concepts:

- `"Spider-Man movie"`
- `"mainline Spider-Man"`
- `"animated movie"`

Likely Step 2B plan for `"Spider-Man movie"`:

- franchise dealbreaker include
- entity dealbreaker include

Lineage-vs-universe preference should **not** be modeled as a generic Step 2B
boost flag. That should live inside Step 3 franchise handling.

### Example 3: `"Disney movies not animated, not too long, preferably from the 90s"`

Likely Step 2A concepts:

- `"Disney movie"`
- `"animated movie"`
- `"shorter runtime"`
- `"1990s release period"`

Likely Step 2B plan:

- studio dealbreaker include
- keyword dealbreaker exclude
- metadata preference
- metadata preference

This is a clean proof case for positive-presence phrasing:

- `"animated movie"` with exclusion polarity
- `"prefer movies below a certain runtime"` rather than `"not too long"`

### Example 4: `"scariest zombie movies"`

Likely Step 2A concept(s):

- `"zombie movie"`
- `"scary intensity"`

or one unified concept if evaluation shows that structure performs better.

The key pressure-test here is that the presence of a dealbreaker does **not**
force the scare preference to become merely supporting. This is exactly the kind
of query where the preference may still be `core`.

---

## Implementation Notes For Later

These are not part of the LLM contract, but they are important context for
later implementation.

### Concept-level aggregation remains code, not schema

If one concept emits multiple expressions, code should aggregate them so the
concept does not behave like multiple independent full matches.

This logic belongs in search execution / reranking code, not in the LLM output.

### Step 3 will need endpoint upgrades

Current expected follow-on work:

- entity endpoint:
  add character prominence handling, analogous to actor prominence
- franchise endpoint:
  support cases where lineage matches should be preferred over broader universe
  matches

These are endpoint-native improvements, not reasons to expand the Step 2B
schema.

### If Step 2A proves too lossy, revisit only one addition first

If evaluation later shows that `concepts: list[str]` makes Step 2B redo too
much concept-boundary reasoning, the first thing to revisit should be adding a
lightweight per-concept boundary note.

Do **not** reintroduce:

- concept-level dealbreaker/preference labels
- candidate-generation hints
- generic boost fields
- endpoint-specific scope fields

unless later evidence proves they are necessary.

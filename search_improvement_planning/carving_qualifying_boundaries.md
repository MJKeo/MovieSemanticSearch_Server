# Carving vs Qualifying Trait Boundaries

The search pipeline classifies every high-level trait in a user query
as either **carving** or **qualifying**. The distinction is
role-based: it depends on what the trait does in the query, not what
kind of word the trait is. The same trait can play different roles in
different queries.

This document defines the two categories, gives the classification
rule, and walks through the edge cases that pin down the boundary.

---

## Definitions

### Carving trait

A trait that defines what kinds of movies belong in the result set at
all. The natural test is *"does this movie have X?"* — a yes/no
answer, possibly via a threshold. A movie that fails the test is
irrelevant to the request, not just ranked lower. The natural
operation is filtering.

When multiple carving traits appear in a query, they intersect: the
result is the AND of every carving trait's qualifying set.

Maps to: dealbreaker, filter, hard requirement.

### Qualifying trait

A trait that orders movies within a pool that other traits have
already carved. The natural test is *"how much X does this movie
have?"* — a continuous spectrum that every movie sits somewhere on.
A movie low on X is still a valid result, just ranked below higher-X
candidates. The natural operation is ranking.

When multiple qualifying traits appear in a query, they stack as
parallel ranking axes, each contributing a normalized score to the
final rerank.

Maps to: preference, soft requirement, reranking signal.

---

## Polarity: the four roles

Carving and qualifying are orthogonal to **polarity** (does the user
want this trait, or want to avoid it). Crossing the two axes gives
four roles, and every trait sits in exactly one cell.

### Positive carver

Determines whether a movie is worth including in the candidate set
and scores how strongly it matches. The score has a downstream
consumer — rerank — so the gradient tail past the elbow is kept:
weak-but-plausible matches enter the pool with a low score and let
rerank arbitrate. Over-fetch by design.

Maps to: `inclusion_candidates` in `HandlerResult`.

### Negative carver

Determines whether a movie should be excluded from the results.
Binary by nature — exclusion has no downstream consumer (a movie is
either out or in; exclusion does not compose with anything), so a
graded exclusion score would be dead weight. Each endpoint defines
its own confidence cutoff for emitting an exclusion. Movies that do
not get considered at all simply do not get excluded.

The cost calculus differs from positive carving: a false-positive
exclusion is expensive (movie is gone forever, user never sees it),
so the cutoff sits conservatively past the elbow. Better to under-
exclude than over-exclude.

Maps to: `exclusion_ids` in `HandlerResult`.

### Positive qualifier

Scores candidate movies on how well they match a desired trait and
boosts them in the rerank. Every movie in the pool gets a score; no
movie is dropped on the basis of a qualifier. Multiple positive
qualifiers stack as parallel ranking axes.

Maps to: `preference_specs` in `HandlerResult`.

### Negative qualifier

Scores candidate movies on how well they match an undesired trait
and penalizes them in the rerank. Same shape as positive qualifier
with the polarity flipped: a strong match contributes a downward
push rather than removal from the pool. Necessary for "not too
violent" / "doesn't feel formulaic" — gradient axes where hard
exclusion would over-filter.

Maps to: `downrank_candidates` in `HandlerResult`.

### Why the carver asymmetry is principled

Positive carvers carry gradient scores; negative carvers do not.
That falls out of asking what the downstream stage does with the
score:

- **Inclusion scores compose with rerank.** A weak inclusion at 0.3
  still combines with other signals and may or may not surface;
  removing the tail throws away information rerank could have used.
- **Exclusion scores have no consumer.** A movie is either cut or
  kept; there is no half-excluded state. The only meaningful
  question is "should this be cut, yes or no," and given the cost
  asymmetry the only safe answer is "yes, only when the match is
  confident."

Symmetric hard-cutoff-both-sides would be the worse design: it
treats the elbow as a hard truth boundary on the inclusion side
when it is really a confidence-of-match curve, and sacrifices
recall on borderline-but-correct matches that rerank could have
surfaced.

Rule of thumb: **gradient where there is a downstream consumer for
the gradient; hard threshold where there is not.**

### Open considerations / TODOs

- **Per-endpoint negative-carver logic.** Each endpoint needs an
  explicit, documented rule for what crosses the exclusion
  threshold. For categorical endpoints (entity, franchise, awards,
  studio, keyword, most metadata) this is mostly trivial — a binary
  fact match. For the semantic endpoint it is the live question:
  what similarity score to a negation query counts as a confident
  match? Tune conservatively; the elbow is a floor, not a target.
- **Two-knob threshold tuning for semantic carvers.** The
  inclusion-tail extent and the exclusion-elbow cutoff are
  independent knobs and do not have to move together. Surface them
  separately in whatever config layer ends up holding them rather
  than assuming a single "elbow" setting per vector space.
- **Endpoint-level emission constraints.** Some endpoints can
  legitimately emit only a subset of the four roles (e.g. a
  hard-categorical endpoint will rarely produce qualifier output).
  The handler prompts and schemas should make clear which cells
  each endpoint is allowed to populate so the LLM does not invent
  cross-bucket findings.

---

## The classification rule

Walk the high-level traits in the query. Apply in order:

1. **Categorical traits → all carving.** A categorical trait is a
   verifiable fact about the movie: a named entity, setting, location,
   event, structural device, format, source material, tag, or
   threshold-bounded numeric. They intersect with each other
   regardless of relative specificity.
2. **If at least one categorical trait exists → all gradient traits
   become qualifying.** A gradient trait is an experiential or
   evaluative quality whose natural form is a continuous spectrum:
   mood, tone, intensity, vibe, praise level, popularity, scope,
   resonance.
3. **If zero categorical traits exist → gradient traits become
   carving** via their categorical sibling (funny↔COMEDY,
   scary↔HORROR, feel-good↔FEEL_GOOD) or via a soft descriptiveness
   threshold (popular = above the popularity bar; unsettling = above
   the disturbance bar). The trait has to take on the defining role
   when nothing else does.

The role decision is made at the **trait level**, not the atom level.
A high-level trait like "like Eternal Sunshine" decomposes into
multiple atoms internally, but the trait itself gets one role
classification, and that role applies to the trait's aggregated
normalized score.

---

## Why role, not trait identity

An earlier draft of this rule classified traits by their intrinsic
nature: "funny" was always one type, "horror" always the other. That
fails on queries where the same trait functions differently:

- "funny movies" — funny defines what the user wants. Carving.
- "funny horror movies" — horror defines the kind of movie; funny
  ranks horror by humor. Qualifying.

Specificity drives role *only across the categorical/gradient
boundary*. Two categoricals at different specificity levels
(e.g. "90s comedies") do not qualify each other — they intersect.
A categorical and a gradient (e.g. "funny horror") do qualify: the
gradient drops to ranking signal because the categorical is already
carving the pool.

The "broad qualifies the niche" intuition is the same rule from the
gradient side: gradients are by nature broader (every movie has some
amount), categoricals by nature narrower (yes/no fact). When both
appear, the broad one qualifies the narrow one.

---

## Edge cases

### 1. Same trait, different role via companions

**"a feel-good film"** — feel-good is the only trait. It steps up to
carving via its FEEL_GOOD categorical sibling. The result is
heartwarming/uplifting movies as a defined pool.

**"a feel-good comedy"** — comedy is categorical and already carves.
Feel-good drops to qualifying: rank comedies by feel-good intensity.

The exact same word switches role based on what else is in the query.
Funny↔COMEDY, scary↔HORROR, made-me-cry↔TEARJERKER follow the same
pattern. These are gradient-with-categorical-sibling traits, the
only true role-flippers.

### 2. Modifier-binding promotes a gradient to a categorical

**"a quiet movie"** — "quiet" attached to the whole movie reads as
gradient quality (low-energy, restrained). Steps up to carving as
the only trait.

**"a movie with a quiet score"** — "quiet" attached to "score" pins
it to a structural property of a specific element of the movie.
Categorical fact: does the score read as quiet, yes/no. Carving.

The categorical/gradient decision isn't about the modifier word — it's
about what surface the modifier is attached to.

### 3. Same root word, different surface

**"ambiguous ending"** — ambiguous attached to a structural element
(the ending). Categorical → carving (clean OPEN_ENDING tag).

**"ambiguous tone"** — ambiguous attached to a quality dimension
(tone). Gradient → qualifying when other traits carve, carving when
alone.

**"morally ambiguous protagonist"** — ambiguous attached to a
character archetype. Categorical (ANTI_HERO-adjacent tag) → carving.

The pre-pass output already captures the binding via the noun the
modifier attaches to, which is the right level for the rule to
operate on.

### 4. All-gradient figurative query

**"a movie that feels like a warm hug"** — fully figurative phrase,
no categorical traits anywhere. Decomposes into multiple gradient
atoms (comforting tone, warm emotional palette, comfort-watch
occasion) — none with a clean categorical sibling.

The whole conglomerate steps up to carving via descriptiveness
threshold: pool becomes "movies above the warm-hug threshold."

This tests the rule's edge: even when every component is gradient,
something has to do the carving. Step-up via threshold is always
available.

### 5. Specificity gap with both traits categorical

**"a horror movie set on a submarine"** — horror is a top-level
genre with thousands of qualifying movies. "Set on a submarine" is
a narrow setting fact with maybe a few dozen. The specificity gap is
huge.

But both are categorical. They stack via intersection: horror AND
submarine-set, both carving. The user wants both to be true; the
narrower one doesn't get treated as the "real" carving trait while
the broader one drops to ranking. The result pool may be small, but
that's a recall concern, not a role concern.

Specificity-driven role-flips happen only when the broad trait is
gradient.

### 6. Superlative ordinal on a gradient axis

**"Tarantino's least violent film"** — Tarantino is categorical
(entity) → carving. "Least violent" is a gradient quality (violence
intensity) plus an ordinal sort directive (ascending).

The ordinal doesn't promote it to carving — it's still ranking
within the carved pool, just with a position-selection on top. Same
shape as "the most recent Scorsese": entity carves, the gradient
sorts.

### 7. Negation as carving-by-absence

**"a non-violent crime thriller"** — "crime thriller" is categorical.
"Non-violent" is the negation of a gradient trait, but the user is
defining the pool *by absence* of violence. The set "crime thrillers
without violence" is a hard intersection, not a ranking direction.

Negation flips role to carving. Compare with #8.

### 8. Negation as polarity inside a qualifying trait

**"a romantic comedy that doesn't feel formulaic"** — "romantic
comedy" is categorical → carving. "Doesn't feel formulaic" is a
gradient quality (formulaic-ness) with negative polarity. Every
rom-com sits somewhere on the formulaic spectrum; the user wants
the anti-formulaic end.

Negation here only flips polarity within the qualifying role. The
result is rom-coms ranked ascending by formulaic-ness; mildly
formulaic rom-coms aren't excluded, just down-ranked.

The distinction between #7 and #8: in #7 the user is defining a hard
sub-pool (crime thrillers minus violent ones). In #8 the user is
expressing a ranking direction over a continuous trait. Look at
whether the negated trait, if it appeared positively, would have
been carving (then negation = carving-by-absence) or qualifying
(then negation = polarity inversion within qualifying).

### 9. Parametric reference with hybrid constraint

**"a movie like Eternal Sunshine but set in space"** — two high-level
traits:

- "like Eternal Sunshine" — similarity request. Gradient by nature
  (rank by how similar). Decomposes internally into multiple atoms,
  some of which look categorical (non-linear timeline, romance,
  memory themes) — but those atoms stay scoped inside the parametric
  trait. They don't escape and gate independently. Qualifying.
- "set in space" — concrete setting fact. Categorical → carving.

Result: space-set movies, ranked by Eternal-Sunshine-likeness.

If the internal categorical atoms escaped to gate the pool, the user
would get nothing — sci-fi space-set non-linear romances about memory
loss is a vanishing intersection, and that's not what they asked for.
Trait-level role classification keeps the user's intent intact:
similarity stays a ranking signal, not a fan-out of hard filters.

This is the core argument for trait-level (not atom-level) scoring:
atoms within a trait collapse into one normalized similarity score;
the trait's role decides how that score is used.

### 10. Pure gradient alone, no categorical sibling

**"popular movies"** — popular is a gradient scalar with no clean
categorical sibling (unlike funny→COMEDY). Only trait, so no other
carve exists.

The trait still has to define the pool somehow. Step-up via
descriptiveness threshold: pool becomes "movies above the popularity
threshold," ranked descending. The user's expectation is a ranked
list of well-known movies — that IS the carving output of the rule
when the trait is alone.

Compare to #1 where "feel-good" alone steps up via FEEL_GOOD
categorical sibling. Both end up carving when alone; the mechanism
differs (categorical sibling vs threshold) but the role is the same.

---

## Practical implications

### Trait-level scoring

The unit of scoring and role classification is the trait, not the
atom. Atoms within a trait route to whichever endpoints they need
(semantic, keyword, metadata) and produce raw scores; those raw
scores collapse into one **per-trait score** normalized to [0, 1].
The trait's role decides whether that normalized score gates
candidate inclusion (carving) or contributes to the rerank
(qualifying).

This prevents three failure modes:

- **Atom-count dominance.** A parametric reference with six atoms
  doesn't get six times the ranking weight of a single-atom trait.
- **Internal-categoricals-escaping.** Categorical atoms nested inside
  a qualifying parametric trait don't independently gate the pool.
- **Hybrid pure-similarity queries falling through cracks.** "Movies
  like X but Y" goes through the standard flow once trait-grouping
  is in place; pure "movies like X" is just the degenerate case of
  one qualifying trait with no carving traits, which steps up via
  the rule.

### Where the role decision lives

Role classification happens at the pre-pass / fragment level. The
pre-pass already groups atoms under fragments; the shift is making
the fragment the load-bearing unit for:

- Role classification — one decision per fragment, applied to the
  trait conglomerate.
- Score normalization — atoms aggregate within fragment, then
  fragment scores normalize against each other.
- Reranking weights — each fragment is one weighted contributor to
  the final score.

When the role of a multi-atom fragment seems atom-specific, the
fragment's role wins. The atoms are the implementation detail of how
the fragment computes its score, not independent voices in the
routing decision.

### Most decisions are deterministic

Of the 32 query categories, 23 align with a single role unconditionally
(17 always-carving, 6 always-qualifying). 8 are mixed and require the
rule to be applied at query time, almost always inside the semantic
endpoint where the gradient/categorical decision is genuinely live. 1
(Interpretation-required) is unsure by design.

Handlers for the always-carving and always-qualifying categories can
hardcode the role and skip the LLM-side reasoning. The mixed
categories are where the rule earns its keep.

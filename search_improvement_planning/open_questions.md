# Open Questions

Unresolved conversation topics, untested theories, and questions that need answers.

---

## Architecture Questions

### Can the LLM reliably classify deal-breakers vs preferences?

This classification is **load-bearing for the entire pipeline.** If the LLM
misclassifies "marvel" as a preference instead of a deal-breaker, you get the same
results as today. How do we make this robust?

Possible mitigations:
- Few-shot examples with edge cases in the prompt
- Rule-based overrides for common patterns (named entities always deal-breakers?)
- Confidence scores that trigger fallback to the current broad search when uncertain
- User feedback loop where clicking "this doesn't match" trains the classification

The classification boundary is fuzzy and query-dependent. "Twist ending" in "iconic
twist ending" is a deal-breaker, but in "thriller with a twist ending" it might be a
preference. The same word changes structural role based on what else is in the query.
How do we handle this reliably?

### Does this replace the channel weight system or layer on top?

The current channel weight system (vector_relevance, lexical_relevance,
metadata_relevance) was solving a different version of the same problem — "how much
should each signal matter?" The dependency hierarchy gives that information more
structure. Options:

- **Replace:** New query understanding output completely replaces channel weights
  with the deal-breaker/preference/implicit structure
- **Layer:** Keep channel weights for Phase 4 (exploratory extension) and use the
  new structure only for Phases 1-3
- **Evolve:** Channel weights become an implementation detail within each phase
  rather than a top-level concept

### How does the anchor vector fit in the new system?

Currently anchor always runs and gets 0.8x the mean of active non-anchor weights.
In the new system, what role does it play?

- Phase 1 deal-breaker retrieval: probably not useful (too broad)
- Phase 2 preference scoring: maybe useful as a "general relevance" signal
- Phase 4 exploratory extension: likely valuable for broad similarity
- Or does it become a safety net that ensures movies with strong overall relevance
  don't get completely excluded by narrow deal-breaker retrieval?

---

## Retrieval Questions

### The Top-N Retrieval Gap

Concern: If we don't explicitly fetch for movies with a given attribute, we risk
those movies not being included in the top N of any individual vector search despite
being a decent enough match for each one.

**Example scenario:** A movie is 600th in narrative_techniques for "twist" AND 600th
in reception for "iconic." It genuinely belongs at the intersection but never enters
the candidate pool because it's not top-500 in either space.

**Preliminary assessment:** This is likely less of a problem than feared because:
1. Deal-breaker retrieval should use high recall (500-1000 candidates per space)
2. For binary-ish attributes like "has a twist ending," relevant movies cluster
   tightly in embedding space — there aren't 600+ twist movies scoring higher than
   Fight Club
3. The anchor vector provides a second retrieval path that's naturally intersection-
   aware because it combines many attributes

**Where the gap IS real:** Obscure movies with rare attribute combinations. These
should be handled by the exploratory tier (Phase 4).

**Still needs validation:** Run actual queries and check where known-relevant movies
rank in raw cosine similarity within each space. If Fight Club is truly 300th for
"twist ending" in narrative_techniques, that's a retrieval issue. If it's 50th, the
top-N gap isn't the problem.

### How do we handle intersection when deal-breakers span different retrieval channels?

"Dark gritty marvel movies":
- "Marvel" → lexical search
- "Dark and gritty" → vector search (viewer_experience)

The candidate set is the intersection of lexical and vector results. But lexical
might return 200 marvel movies, and vector might return 500 "dark/gritty" movies with
only 30 in common. Is 30 enough? Do we need to expand either pool?

### What candidate limits should we use for deal-breaker retrieval?

Current system pulls top-500 per vector space. For deal-breaker retrieval:
- Higher limits = better recall but more noise and latency
- Lower limits = faster but might miss borderline candidates
- Should limits differ by attribute type? Entity deal-breakers might need fewer
  candidates than semantic deal-breakers

---

## Scoring Questions

### How to set the deal-breaker threshold?

Three options proposed, none evaluated:

1. **Score-distribution-based:** Find natural gap/elbow in similarity scores
   - Pro: adapts to each query's score distribution
   - Con: assumes a gap exists (what if scores are uniformly distributed?)

2. **Relative to top:** e.g., above 70% of top candidate's score
   - Pro: simple, predictable
   - Con: arbitrary percentage, doesn't account for score distribution shape

3. **LLM-adjudicated at the boundary:** Ask LLM "is this a christmas movie?" for
   borderline candidates
   - Pro: most accurate for fuzzy attributes
   - Con: latency cost, need to define the "boundary zone"

Need to test all three on real query distributions to see which produces the most
natural candidate sets.

### Where is the trim point between primary and exploratory results?

Phase 3 trims off "clearly less useful candidates" before the exploratory tier. How?

Options:
- Natural score gap in the preference-ranked list
- Fixed depth (top 25? top 50?)
- Minimum absolute score threshold
- Some combination (e.g., top 25 OR until score drops below 50% of the top result)

### How to handle the "spectrum deal-breaker" problem?

Deal-breakers that are spectrums create a classification problem. "Twist ending"
exists on a spectrum — does Shutter Island have a "twist ending" or just "a reveal"?
Users will disagree.

The threshold + flatten approach handles this mechanically, but the threshold
position determines where on the spectrum you draw the line. Too strict = missing
valid results (Shutter Island excluded). Too loose = including movies that don't
really qualify (any movie with any surprise at all).

---

## Theories That Need Testing

### Embedding density theory (partially confirmed)

**Theory:** Movies whose primary identity revolves around a single attribute have
denser embedding signal for that attribute, leading to higher cosine similarity
even when other movies clearly possess the attribute too.

**Evidence found:** Wild Things has "twist" appearing ~5 times across metadata vs
~2 times for Fight Club and The Sixth Sense. Wild Things' identity IS twists; Fight
Club's identity spans consumerism, identity crisis, anarchy, etc.

**What's untested:** Whether this density effect is the dominant factor in the ranking
gap, or whether the additive scoring architecture would still suppress iconic movies
even with perfectly equal density. The metadata comparison suggests the density effect
is real but secondary to the architectural issue.

### Subquery quality theory (untested)

**Theory:** The subqueries and weights generated for "iconic twist ending" might not
be routing to the right vector spaces or generating effective search terms.

**What we know:** The weight prompts explicitly list "twist ending" as a high-
relevance signal for narrative_techniques. The subquery prompts show how to translate
it. But we never actually ran the query through the pipeline to see what subqueries
and weights were generated.

**Needs:** Run "iconic twist ending" through the notebook and inspect:
- Which spaces got what weights
- What subqueries were generated for each space
- Per-space scores for Fight Club, Sixth Sense, Wild Things
- Whether the right spaces are even being queried

### Flattening impact theory (untested)

**Theory:** If we threshold + flatten deal-breaker scores, the preference layer
(reception/popularity/iconic status) would naturally surface Fight Club and The Sixth
Sense to the top because they're more "iconic" than Wild Things.

**Needs:** Simulate this by:
1. Taking the narrative_techniques results for "twist ending"
2. Flattening all candidates above some threshold to 1.0
3. Re-ranking by reception/popularity scores
4. See if the expected iconic movies rise to the top

---

## Presentation Questions

### Append vs weave for exploratory results?

**Append:** Clearly separated "You might also like" section. Safer, more honest, but
creates a hard tier boundary.

**Weave:** Interleave exploratory with primary results. More discovery-friendly but
risks confusing users who expect everything to match their query.

**Not yet discussed:** Could the choice be query-dependent? Pure vibe queries might
benefit from weaving (everything is approximate anyway). Multi-constraint queries
might benefit from appending (the primary results clearly match, the exploratory
clearly don't).

### How to signal which tier a result belongs to?

If using tiered results, should the UI communicate WHY something is in the
exploratory tier? "This doesn't have a twist ending but is iconic and similar in
tone" is useful context. But generating explanations adds cost and complexity.

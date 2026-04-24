# Specific subject / element / motif — additional notes

This category is about whether a specific **subject, element, or motif is IN the story** — a real-world subject (JFK, Watergate, the Vietnam War, Princess Diana) or a fictional element / motif (clowns, zombies, sharks, robots, vampires). The canonical question is binary: does this thing feature in the film? Keyword is the authoritative tier when the subject has a canonical registry member; Semantic plot_events is the fallback for long-tail subjects the registry does not cover and for gradient "how-much-of-X" framings.

## How the tier bias actually applies here

- **Canonical registry member exists and the framing is binary → Keyword wins.** "Zombie movies", "movies with vampires", "biopics" — the registry carries the target concept as a closed-enum tag, so Keyword is both more precise and more recall-complete than Semantic over the plot_events space.
- **No registry member covers the subject → Semantic wins decisively.** Specific historical persons or events without a canonical tag (most named real-world subjects: "Princess Diana", "JFK assassination", "the Watergate scandal", "the moon landing") land cleanly in the plot_events space, which holds the concrete "what happens / who is in it" text. The keyword-first bias is a tiebreaker, not a veto: a clearly-better Semantic pick beats a missing-concept Keyword stretch.
- **Gradient framing flips the tier to Semantic even when a canonical tag exists.** Binary membership ("movies with zombies") → Keyword. Spectrum / degree framings ("subtly zombie-themed", "movies with loose allegorical zombie undertones", "has faint vampire motifs without being a vampire movie") → Semantic. A concept tag treats membership as absolute; spectrum phrasing needs the graded matching that vector similarity provides.

State this reasoning explicitly in `performance_vs_bias_analysis`. Name whether the Keyword registry does or does not cover the subject, and whether the framing is binary or graded.

## Semantic routing — plot_events is the primary target

When Semantic fires for this category, plot_events is the single most-effective space for subject-in-story matching. The space holds the concrete prose summary of what happens and who is in it — exactly where a real-world subject or named motif would surface on the ingest side. Additional spaces are populated only when the requirement genuinely spans dimensions (e.g. a thematic allegory framing may also land in plot_analysis). Do not pad additional spaces to look thorough; an honest single-space Semantic call with plot_events as `primary_vector` is the right shape for most of this category's Semantic fires.

## Boundaries with nearby categories

- **Named character (Cat 2).** A specific credited fictional persona — Batman, Wolverine, James Bond — is Cat 2. A motif / element / subject is Cat 6. "Batman movies" → Cat 2. "Movies with vigilante-themed characters" → Cat 6 if it is describing a motif, not a named persona. The discriminator: does the reference name a single identifiable persona the cast list would credit, or a type of thing that appears in the story?
- **Character archetype (Cat 7).** An archetype is about character TYPE — lovable rogue, femme fatale, anti-hero. That is Cat 7. Cat 6 is about a thing that appears IN the story, not a pattern the protagonist embodies. "Movies with a femme fatale" → Cat 7. "Movies with clowns" → Cat 6.
- **Adaptation source flag (Cat 5).** "Biopic" / "true story" can be adaptation-source signals (Cat 5, keyword on BIOGRAPHY / TRUE_STORY) OR subject-presence signals (Cat 6) depending on framing. "Biographical films" as a category ask → Cat 5. "Movies about JFK" → Cat 6 (the specific person is the subject). When both surface, they decompose into separate atoms upstream; handle only the Cat 6 slice.
- **Plot events + narrative setting (Cat 19).** Full plot premises and narrative settings ("a heist that unravels", "set in 1940s Berlin") are Cat 19 — Semantic-only. Cat 6 is narrower: a single subject / element / motif being present, not a whole plot scenario. If the requirement is "movies where [subject] is central", Cat 6; if it is "movies where [full plot premise with multiple elements]", Cat 19.

## When to no-fire

Return `endpoint_to_run: "None"` when:

- The requirement is too vague to point at a concrete subject or motif. "Movies about interesting things", "films with meaningful elements" — neither Keyword nor Semantic can translate this into a target.
- The upstream dispatch was wrong — the phrase is actually a named character (Cat 2), an archetype (Cat 7), a genre (Cat 12), or a full plot premise (Cat 19), and treating it as a subject/motif would misroute. Name the mismatch in `best_endpoint_gaps` or `performance_vs_bias_analysis`.
- The framing is self-contradictory under the available modifiers in a way neither endpoint can express.

No-fire is always better than forcing a Keyword pick on a registry member that does not actually cover the requested subject or authoring a Semantic plot_events body from invented story facts.

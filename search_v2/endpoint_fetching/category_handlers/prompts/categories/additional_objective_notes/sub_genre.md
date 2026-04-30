# Sub-genre + story archetype — additional notes

This category is about **identifiable story patterns that name a sub-form** — either a sub-genre (body horror, cozy mystery, neo-noir, slasher, giallo, space opera, slow-burn thriller) or a story archetype (revenge, underdog, heist, chase, survival, fish-out-of-water, post-apocalyptic). The requirement names a recognizable label for the whole story shape, not a character type and not a broad top-level genre.

## How the two endpoints split the work

- **Keyword** is tier 1 and the authoritative channel when the sub-genre or archetype maps to a registry member whose definition covers the same concept. The registry holds direct members for many common sub-forms (e.g. `BODY_HORROR`, `SLASHER_HORROR`, `COZY_MYSTERY`, `FOLK_HORROR`, `HEIST`, `SURVIVAL`, `REVENGE`, `UNDERDOG`, `POST_APOCALYPTIC`). When the definition of one of those members names the sub-form the user asked for, Keyword wins cleanly.
- **Semantic** is tier 2 and the right channel when the sub-form has no matching registry member. `plot_analysis.genre_signatures`, `plot_analysis.conflict_type`, and `plot_analysis.character_arcs` are the purpose-built sub-fields for sub-genre labels, conflict archetypes, and arc-pattern labels that the closed vocabulary does not absorb.

**The registry has known holes that matter for this category.** A parent genre may have a tag while a specific sub-form does not: `FILM_NOIR` exists but `NEO_NOIR` does not; `REVENGE` exists but compound patterns like "slow-burn revenge thriller" span multiple dimensions no single tag carries. When the sub-form is named specifically but the registry only covers the parent (or an adjacent sibling), picking the parent tag silently broadens the result set past the intent — Semantic wins decisively in that case.

## Boundaries with nearby categories

- **Top-level genre (Cat 11).** A bare coarse genre label ("horror", "action", "thriller", "comedy") is Cat 11's territory. Cat 15 is for a named sub-form that sits inside a parent genre ("body horror", "neo-noir", "slasher"). The discriminator is whether the label names a recognized sub-pattern or just the parent family. If upstream routed a bare genre here, no-fire — it belongs to Cat 11.
- **Character archetype (Cat 7).** Cat 7 covers static character TYPES (anti-hero, femme fatale, lovable rogue). Cat 15 covers STORY patterns (revenge, underdog, heist). The discriminator is framing: "a revenge story" / "underdog movies" frames the whole plot and is Cat 15; "an anti-hero protagonist" frames the character and is Cat 7. A label like "underdog" can read either way — use it here only when the query frames the story shape, not the character.
- **Kind of story / thematic (Cat 21).** Cat 21 covers thematic concepts and arc trajectories as abstract ideas ("grief", "redemption", "a story about forgiveness"). Cat 15 covers named story-pattern labels ("a revenge story", "a heist movie"). The discriminator is whether the phrase names a recognized pattern label or an abstract theme. "Revenge story" → Cat 15; "story about forgiveness" → Cat 21.

## Picking within Semantic

When Semantic wins, `plot_analysis` is almost always the load-bearing space. Populate the sub-field whose native vocabulary best holds the sub-form:

- **Sub-genre labels** (neo-noir, space opera, folk-horror-adjacent) → `genre_signatures`.
- **Conflict archetypes** (man vs nature, man vs self, chase, survival) → `conflict_type`.
- **Arc-pattern labels** (fall from grace, underdog rise, coming-of-age when framed as story shape) → `character_arcs`.

`plot_events` can carry supporting signal when the archetype is expressible as a concrete situational pattern (a heist, a chase, a survival scenario) — add it only when the plot-summary vocabulary genuinely captures something `plot_analysis` does not.

## When to no-fire

Return `endpoint_to_run: "None"` when the requirement is not actually a sub-genre or story-archetype ask:

- The phrase names a bare top-level genre with no sub-form — that is Cat 11's territory.
- The phrase names a character type rather than a story pattern — that is Cat 7's territory.
- The phrase names an abstract theme rather than a recognized story-pattern label — that is Cat 21's territory.
- The phrase is too vague to point at either a specific registry member or a defensible `plot_analysis` body ("cool story pattern", "interesting genre").

## The one principle

Fire **Keyword** only when a registry member's own definition names the sub-form specifically. Fire **Semantic** when no member does but the sub-form is a genuine story-pattern label describable in `plot_analysis`'s native vocabulary. The tier-1 bias never forces picking a parent-genre tag as a stand-in for a named sub-form it does not cover — silently broadening the lookup is a worse outcome than delegating to Semantic.

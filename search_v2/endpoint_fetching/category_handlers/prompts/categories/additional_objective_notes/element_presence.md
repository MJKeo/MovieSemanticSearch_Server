# Additional objective notes

## Category Target

A concrete thing present in the story — something you can perceive on
screen: see, hear, or touch. Creatures (zombies, sharks, witches),
objects (cars, swords), animals (horses, dogs), activities (heists,
weddings, sword fights), professions (lawyers, hitmen), or recurring
visible/audible motifs. "Has X" framing where X is observable.

**Concrete-perception test.** Could two viewers agree they saw, heard,
or touched the element on screen? If yes, it is in scope. If
identifying it requires inference or interpretation — themes
(grief, redemption), story shapes (underdog arc, fall-from-grace),
character archetypes (anti-hero, femme fatale), narrative devices
(twist ending, unreliable narrator), writing techniques (non-linear
structure, POV tricks) — it is abstract and routes to its dedicated
category, not here.

## Endpoint policy

**Semantic always fires.** plot_events is the native home for
element-presence: the motif-syntax body ("the clown. is a clown.
encounters the clown. the clown returns.") retrieves films where
the element actually appears on screen, distinct from films merely
categorized as containing it. Author the body to name the element
exactly. Do NOT fabricate plot around the motif — restate only the
element the user named (the failure mode noted in semantic.md's
plot_events authoring rules). Abstain on semantic only when no space
genuinely carries the element (rare — plot_events covers most
concrete elements via motif syntax).

**Keyword fires only when a single registry member NAMES the
element (entails its presence) — not merely correlates with it.**
The keyword endpoint is a coarse signal; it earns its place when one
tag aligns with the element closely enough to be useful on its own.
Do NOT stitch multiple tags together to fake coverage. If no single
member is a tight, entailing fit, abstain on keyword and let the
plot_events motif body carry the call.

- A single tag that definitionally entails the element commits.
  `ZOMBIE_HORROR` for "zombies" — every ZOMBIE_HORROR film contains
  zombies. `HEIST` for "heist movies" — every HEIST film contains a
  heist. `WITCH_HORROR` for "witches" — every WITCH_HORROR film
  contains witches.
- A genre tag that merely *correlates* with the element fails the
  test. `WESTERN` for "horses" — westerns often feature horses, but
  a western set in a town can lack prominent horses, and a
  horse-prominent contemporary drama is not a western. Firing
  `WESTERN` here tag-matches westerns at 1.0 while genuinely
  horse-prominent non-westerns score 0. This is stretching → abstain.
- A tag that entails the element but only over a narrow sub-form
  fails the test. `WITCH_HORROR` alone for "witches" excludes witch
  comedies (*Hocus Pocus*), witch romances (*Practical Magic*), and
  witch dramas — large slices of element-satisfying films get zeroed.
  Narrow-only entailment with gaps → abstain. Do not stitch a
  broader-but-non-entailing genre (`DARK_FANTASY`,
  `SUPERNATURAL_FANTASY`) onto it to plug the gap — that puts a
  correlation tag into the commit and reintroduces the stretching
  failure. plot_events motif syntax carries the broader slice cleanly.
- The over-pull allowance applies only after the entailment gate is
  passed. `ZOMBIE_HORROR` retrieving comedy and serious zombie films
  alike is acceptable over-pull (semantic refines tone).

**Semantic still fires alongside a perfect-cover keyword.** When the
registry has a direct entailing tag (`ZOMBIE_HORROR`, `HEIST`),
keyword handles the category-level signal and semantic adds the
literal on-screen presence check via plot_events motif syntax. The
two endpoints layer — semantic does not become redundant once keyword
has a clean fit.

**Cross-family keyword borrowing is rare for ELEMENT_PRESENCE.**
Most observable elements live in the GENRE / SUB-GENRE families.
Borrowing from SOURCE_MATERIAL or CONCEPT_TAG only when those tags
genuinely entail the element (uncommon).

## Boundaries

Sibling categories that own adjacent slices — upstream routing
sends each ask to its own home, so these boundaries are
expectations about what this handler will and will not receive,
not a fallback path. They affect how you read overlapping or
ambiguous siblings in the `<sibling_categories>` block.

- Centrality / "about X" framing → CENTRAL_TOPIC.
- Static character TYPES (anti-hero, femme fatale, manic pixie) → CHARACTER_ARCHETYPE.
- Abstract themes, story shapes, character trajectories → STORY_THEMATIC_ARCHETYPE.
- Time/place setting ("set in 1980s NYC") → NARRATIVE_SETTING.
- Multi-clause plot event premise ("a heist that unravels when …") → PLOT_EVENTS.
- Structural / craft devices (twist ending, unreliable narrator, found-footage) → NARRATIVE_DEVICES.

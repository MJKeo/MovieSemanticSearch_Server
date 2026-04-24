# Post-viewing resonance — additional notes

This category covers **how the movie lingers after the credits** — either a structural ending type (happy / sad / bittersweet / twist / open / cliffhanger ending) or the experiential aftertaste it leaves (haunting, gut-punch, stays with you, forgettable). The two sub-kinds route to different endpoints; the split is the central decision you make.

## The structural-vs-experiential split

The user's framing points at one of two sub-kinds, and only one endpoint can fit each:

- **Structural ending type.** The user names the *shape* the ending takes — "happy ending", "sad ending", "bittersweet ending", "twist ending", "ambiguous / open ending", "cliffhanger ending", "downer ending". These are definitional and the closed vocabulary carries them directly through the ending-tag family (HAPPY_ENDING, SAD_ENDING, BITTERSWEET_ENDING) and the structural-tag family (OPEN_ENDING for ambiguous, CLIFFHANGER_ENDING, PLOT_TWIST for twist endings). **Keyword is authoritative** — pick the registry member whose definition names the structural ending the user framed, and fire Keyword alone. A Semantic near-miss on `ending_aftertaste` does not earn a pick when a definitional tag covers the ask.

- **Experiential aftertaste.** The user names the *residue* the film leaves — "haunting", "stays with you for days", "sticks with you", "lingers", "unforgettable", "a film you keep thinking about", "forgettable". No structural tag encodes these — a film can be haunting with any ending shape. **Semantic wins decisively.** The primary sub-space is `viewer_experience.ending_aftertaste`, which exists to carry exactly this dimension. Picking a structural ending tag here would fabricate a shape the user did not name.

## The bias is strong for structural, weak for experiential

The tier-1 Keyword bias only earns its weight when the ask maps cleanly to a canonical ending tag. When the framing is experiential, the bias does not force Keyword — no structural tag covers aftertaste, so Semantic fits on its own merits and the bias has nothing to break a tie on. Say this explicitly in `performance_vs_bias_analysis` whenever Semantic wins: no definitional tag covers experiential aftertaste, so the bias is inapplicable, not overridden.

## Gut-punch and other straddle framings

Some phrasings partially touch both sub-kinds. "Gut-punch ending" leans toward SAD_ENDING structurally but the dominant signal is the *blow it lands on the viewer* — an experiential aftertaste. When the framing emphasizes the aftertaste ("gut-punch", "devastating", "shattering"), prefer Semantic on `ending_aftertaste` — the structural SAD_ENDING tag captures "the protagonists lose" but not "the viewer is left reeling". When the framing is bare structural ("sad ending", "downer ending"), Keyword wins. Read which half the user actually emphasized.

## Secondary Semantic sub-space

When the aftertaste framing is specifically about **lasting critical or audience resonance** — "films still talked about", "movies that haunt the culture", "cult resonance that keeps growing" — `reception.reception_summary` is the right secondary home alongside `ending_aftertaste`. Most aftertaste asks do not need it; reserve it for framings that explicitly invoke how the film has kept lingering in critical or audience conversation, not just in the individual viewer.

## Boundaries with nearby categories

- **Viewer experience / during-viewing feel (Cat 22).** Cat 22 owns how the movie *feels to watch while you are watching it* — `tone_self_seriousness`, `emotional_palette`, `cognitive_complexity`, `tension_adrenaline`. Cat 26's primary is `ending_aftertaste` specifically. Bare during-viewing asks — "dark movies", "cerebral films", "intense thrillers" — belong to Cat 22, not here. Fire Cat 26 only when the framing is clearly about the ending or about the post-viewing residue, not the ongoing experience.
- **Reception quality (Cat 25).** Cat 25 is general acclaim — cult, acclaimed, underrated, divisive, era-defining. Cat 26 is specifically about resonance after the film ends. "Critically acclaimed" alone is Cat 25. "A film that stays with you" shades into Cat 26 because it names lasting audience resonance as the quality, not general acclaim.
- **Narrative devices (Cat 16).** A plot twist framed as a mid-story structural device belongs to Cat 16. A twist framed specifically as the ending's defining shape belongs here (PLOT_TWIST is still the right tag; the difference is which category's handler fires). If the atom routed here names the ending explicitly, handle it.

## When to no-fire

Return `endpoint_to_run: "None"` when the atom routed here is not actually about post-viewing resonance:

- The framing is a during-viewing tone or mood — that is Cat 22's territory.
- The framing is general acclaim with no resonance angle — that is Cat 25's territory.
- The phrasing is too vague to point at either a specific ending type or a concrete aftertaste descriptor ("good ending", "nice conclusion") — no candidate can express the ask cleanly, and inventing `ending_aftertaste` terms from nothing dilutes the query.

## The one principle

Fire **Keyword** when the user names a specific structural ending type and the registry has a tag that definitionally covers it. Fire **Semantic** when the ask is about experiential aftertaste with no structural anchor, or when a straddle framing puts the viewer's lingering reaction ahead of the ending's literal shape. Otherwise, no-fire — forcing a structural tag on an experiential ask or padding `ending_aftertaste` with invented terms harms the result set more than omission does.

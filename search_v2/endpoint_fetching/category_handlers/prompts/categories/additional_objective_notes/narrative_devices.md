# Narrative devices + structural form + craft — additional notes

This category covers **how the story is told** — structural devices (plot twist, nonlinear timeline, unreliable narrator, single-location, time loop, fourth-wall breaks), cast configuration (ensemble, two-hander), craft-level pacing (slow burn, frenetic, methodical), dialogue style ("Sorkin-style", "Tarantino-style"), POV mechanics, and character-vs-plot focus. The three sub-axes — devices, structural form, how-told craft — share the same tiered routing because the question-kind is always "how is this told?" regardless of which sub-axis the user framed it on.

## How the two endpoints split the work

- **Keyword** is tier 1 and the authoritative channel when the device or structural form maps to a canonical registry member whose definition names it — the NarrativeStructureTag family (PLOT_TWIST, NONLINEAR_TIMELINE, UNRELIABLE_NARRATOR, SINGLE_LOCATION, TIME_LOOP, BREAKING_FOURTH_WALL, CLIFFHANGER_ENDING, TWIST_VILLAIN, OPEN_ENDING) and ENSEMBLE_CAST for the cast-configuration side. When a tag's definition names the device, Keyword delivers the authoritative posting-list hit and Semantic near-misses should not dilute it.
- **Semantic** is tier 2 and the right channel when no canonical tag covers the craft. `narrative_techniques` is the primary home — its sub-fields (`narrative_archetype`, `narrative_delivery`, `pov_perspective`, `characterization_methods`, `information_control`, `conflict_stakes_design`, `additional_narrative_devices`) exist precisely to carry craft descriptors the closed vocabulary does not. `viewer_experience` is the right space when the ask is pacing-as-feel — "slow burn", "frenetic" — because its `tension_adrenaline` and `sensory_load` sub-fields carry pacing as the viewer experiences it.

The bias is a tiebreaker only. Dialogue-style asks ("Sorkin-style", "Tarantino-style") and POV mechanics have no canonical tag; Semantic wins decisively in those cases. Picking a weakly-adjacent tag to honor the bias would misclassify the movie set.

## Boundaries with nearby categories

- **Post-viewing resonance (Cat 26).** Cat 26 owns ending types specifically — HAPPY_ENDING, SAD_ENDING, BITTERSWEET_ENDING as aftertaste, plus "gut-punch ending" / "stays with you" framings. A plot twist framed as a mid-story or general structural device belongs here; a twist framed as the ending's defining aftertaste belongs to Cat 26. When the query explicitly frames the ending as the load-bearing element, defer — no-fire here.
- **Format (Cat 14).** Format is physical presentation (black-and-white, 16mm, IMAX, found-footage, mockumentary). Narrative device is a storytelling choice (nonlinear timeline, unreliable narrator). Mockumentary sits on the boundary — prefer the tag Cat 14 owns when the phrasing is about the format itself. If the query frames it as a storytelling device rather than a visual/presentation format, handle here.
- **Viewer experience (Cat 22).** Cat 22 owns how the movie feels to watch in general (dark, whimsical, cozy, cerebral vs mindless). Cat 16 owns pacing-as-craft — the choice of how the story is unfolded. "Slow burn" sits on both axes; handle the craft framing here and use `viewer_experience` as the Semantic sub-space when the feel is what the user is actually reaching for.

## When to no-fire

Return `endpoint_to_run: "None"` when no candidate genuinely fits:

- The phrase is an ending type framed around aftertaste rather than structure — that is Cat 26's territory.
- The phrase names a physical format — that is Cat 14's territory.
- The phrase is too vague to point at either a canonical tag or a defensible narrative_techniques / viewer_experience body ("well-told", "interesting storytelling", "good writing") — no candidate can express the ask cleanly.

## Picking within Semantic

When Semantic wins, match the sub-space to the craft axis:

- **Dialogue style, characterization approach, POV mechanics, information control, character-vs-plot focus** → `narrative_techniques`. This is the default Semantic home for uncanonized craft.
- **Pacing-as-feel ("slow burn", "frenetic", "methodical")** → `viewer_experience`. `tension_adrenaline` and `sensory_load` carry the pacing the viewer experiences; `narrative_techniques` covers narrative pacing as a craft choice, but the phrasing of "slow burn" / "frenetic" is experiential and lands more strongly on `viewer_experience`.
- **Pacing that is clearly a structural craft choice** (e.g. "deliberately withholds information for the first act") can populate `narrative_techniques.information_control` or `narrative_delivery` instead.

`primary_vector` follows from which sub-space carries the heaviest load-bearing signal.

## The one principle

Fire **Keyword** only when a NarrativeStructureTag or ENSEMBLE_CAST definition clearly names the device. Fire **Semantic** when no tag does but the ask is a real craft description — match the sub-space to the craft axis. Otherwise, no-fire — fabricating a near-miss tag or padding a narrative_techniques body with invented terms degrades the result set more than omission does.

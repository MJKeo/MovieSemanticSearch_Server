# Additional objective notes - Plot events

Search concrete on-screen story content: actions, situations, event chains,
premise details, AND specific motifs / elements that recur on screen
(clowns, phone booths, amnesia, masks). Use semantic
`plot_events.plot_summary`.

Write what happens, not what the story means. Keep the body close to the
given event facts. **Do not invent motives, outcomes, twists, character
names, settings, or side-events absent from retrieval_intent and
expressions** — fabricated detail shifts the retrieval target and is the
primary failure mode for this category.

## Two valid body shapes

The `plot_summary` body has two shapes depending on what the user asked for:

**Specific-event query** ("a heist that falls apart due to crew betrayal"):
1–3 dense sentences in past-tense third-person restating only the events
the user named. Generic agents are fine ("a heist crew", "the protagonist");
specific names, settings, or side-events you fabricated are not. Output
should read like a compact movie synopsis, not a one-line log-line.

Example for the heist query:
> "a heist crew's plan unravels when one member betrays the others. the
> betrayal exposes the operation and forces the remaining crew into a
> fallout."

**Motif / element query** ("clowns as a recurring motif"): short fragments
naming the motif in the contexts a real synopsis would mention it,
joined by periods. The fragments mirror the actual phrasings that appear
inside real movie plot summaries — they retrieve films that contain the
motif WITHOUT inventing a plot around it.

Example for clowns:
> "the clown. is a clown. and then the clown. encounters a clown. the
> clown returns."

Example for amnesia as a motif:
> "wakes up with no memory. has amnesia. cannot remember. memory loss."

**Counter-example (do NOT emit):** "a clown chases a woman through a
carnival as her boyfriend tries to save her" — that's a fabricated plot,
not a motif retrieval target. It moves the embedded query away from
generic motif-occurrence films and toward a single fictional plot.

## Boundaries

This category is not for narrative setting alone; that has its own template.
It is also not for theme, genre label, tone, release era, filming location,
or viewing occasion.

## No-fire

No-fire when the phrase is only thematic ("about grief"), tonal
("unsettling"), vague ("wild plot"), or production geography. Concrete
motifs and elements DO fire here — emit them as fragments per the rule
above.


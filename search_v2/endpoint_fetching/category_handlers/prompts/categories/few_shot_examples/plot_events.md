# Few-shot examples - Plot events

<example>
Input:
```xml
<retrieval_intent>Find films where a heist falls apart after one crew member betrays the others.</retrieval_intent>
<expressions><expression>heist unravels after crew betrayal</expression></expressions>
```
Expected: fire semantic plot_events. The body is 2-3 sentences of past-tense third-person synopsis prose restating only the events the user named — generic agents, no fabricated names / settings / side-events.

Expected `plot_summary`:
> "a heist crew's plan unravels when one member betrays the others. the betrayal exposes the operation and forces the remaining crew into a fallout."
</example>

<example>
Input:
```xml
<retrieval_intent>Find films about strangers trapped together who discover one of them is dangerous.</retrieval_intent>
<expressions><expression>strangers trapped together with hidden threat</expression></expressions>
```
Expected: fire semantic plot_events. 2-3 dense sentences naming only the situation the user supplied; do NOT invent the setting (cabin, elevator, island, plane), the killer's identity, motive, or outcome.

Expected `plot_summary`:
> "a group of strangers find themselves trapped together. as tension rises, they realize one among them is dangerous. their attempts to identify the threat unravel the group's fragile trust."
</example>

<example>
Input:
```xml
<retrieval_intent>Find films featuring clowns as a recurring on-screen presence.</retrieval_intent>
<expressions><expression>clowns as a recurring motif</expression></expressions>
```
Expected: fire semantic plot_events. The user named a motif, not a specific event — the body is short fragments naming the motif in synopsis-style contexts, joined by periods. **Do NOT fabricate a plot around the motif.**

Expected `plot_summary`:
> "the clown. is a clown. and then the clown. encounters a clown. the clown returns. another clown appears."

COUNTER-EXAMPLE — do NOT emit:
> "a clown chases a woman through a carnival as her boyfriend tries to save her."

That fabricates a specific plot the user did not name and shifts retrieval toward one fictional film instead of toward the broad set of films where clowns recur on screen.
</example>

<example>
Input:
```xml
<retrieval_intent>Find films where the protagonist wakes up with no memory.</retrieval_intent>
<expressions><expression>amnesia premise</expression></expressions>
```
Expected: fire semantic plot_events. Amnesia is a premise / motif — emit short synopsis-style fragments naming it, do not invent the protagonist's identity or quest.

Expected `plot_summary`:
> "wakes up with no memory. has amnesia. cannot remember. memory loss. tries to piece together their identity."
</example>

<example>
Input:
```xml
<retrieval_intent>Find films about grief and reconciliation.</retrieval_intent>
<expressions><expression>grief and reconciliation</expression></expressions>
```
Expected: no-fire. Thematic archetype, not concrete on-screen event. Belongs to STORY_THEMATIC_ARCHETYPE → semantic plot_analysis.
</example>

<example>
Input:
```xml
<retrieval_intent>Find films with a tense, unsettling mood.</retrieval_intent>
<expressions><expression>tense unsettling</expression></expressions>
```
Expected: no-fire. Tonal feel, not a plot event. Belongs to EMOTIONAL_EXPERIENTIAL → semantic viewer_experience.
</example>

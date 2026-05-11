# Few-shot examples

<example>
Input:
```xml
<retrieval_intent>Find films whose focal subject is World War II.</retrieval_intent>
<expressions><expression>movies about WW2</expression></expressions>
```
Expected: both endpoints commit. Keyword walk surfaces `WAR` (broad
superset — every WW2 film is a war film) and `WAR_EPIC` (narrow
sub-form — covers epic-scale combat films like *Saving Private Ryan*
but excludes WW2 dramas like *Schindler's List*). Commit `WAR` alone:
single-member cover is enough, and the over-pull (other war eras) is
acceptable because WW2 is a major slice of war films and semantic
recovers the WW2-specificity. Do NOT add `WAR_EPIC` to the commit —
the new policy avoids stitching tags together; `WAR` already covers
the subject. Semantic commits on WW2 specifically — plot_events motif
text naming WW2 settings, units, operations, and plot_analysis
thematic anchors on the WW2 axis.
</example>

<example>
Input:
```xml
<retrieval_intent>Find biographical films as a subject class.</retrieval_intent>
<expressions><expression>biopics</expression></expressions>
```
Expected: both endpoints commit. Keyword walk surfaces `BIOGRAPHY` —
perfect cover for the subject class (the subject *is* the biographical
class). Commit `BIOGRAPHY` alone. Semantic STILL fires — the registry
covers the class label, semantic adds the framing dimension a tag
cannot carry: plot_analysis anchors on "real-person life story" /
"biographical structure", plot_events motif text grounded in
life-event coverage. Do NOT abstain on semantic because keyword has
perfect coverage; this category routes semantic by default because
the two endpoints layer.
</example>

<example>
Input:
```xml
<retrieval_intent>Find films whose focal subject is running.</retrieval_intent>
<expressions><expression>movies about running</expression></expressions>
```
Expected: both endpoints commit. Keyword walk surfaces `SPORT` —
running is a meaningful slice of sport films, so SPORT is a slightly
broader single-member superset (not "a tiny needle in a haystack").
Commit `SPORT` alone. The over-pull `SPORT` introduces (football,
basketball, hockey) is acceptable because semantic recovers the
running-specificity on the same call. No running-specific tag exists
and sub-forms like `EXTREME_SPORT` have under-coverage (running is
mostly not extreme) — abstain on those, do not stitch. Semantic
commits on running as central plot focus — training arcs, race
climaxes, the runner's psychology.
</example>

<example>
Input:
```xml
<retrieval_intent>Find films whose focal subject is Princess Diana.</retrieval_intent>
<expressions><expression>about Princess Diana</expression></expressions>
```
Expected: semantic-only commit; keyword abstains with
`commitment-criteria-fail`. Keyword walk surfaces `BIOGRAPHY` — every
Diana film is biographical, but BIOGRAPHY covers every biopic of every
subject (scientists, politicians, athletes, musicians). Diana-focused
films are a tiny fraction of that universe, so firing BIOGRAPHY alone
tag-matches every unrelated biopic at 1.0 while adding effectively no
signal about the Diana-specific subject — that is the
too-broad-to-be-useful pattern, not acceptable over-pull. No tighter
registry member names Diana or the royal family at a useful
granularity. Abstain on keyword. Semantic carries the call —
plot_events motif text naming Diana, royal-family context, and
public-life events grounded in the request, scored against per-movie
text where Diana-focused films will surface uniquely.
</example>

<example>
Input:
```xml
<retrieval_intent>Find films whose focal subject is chess.</retrieval_intent>
<expressions><expression>movies about chess</expression></expressions>
```
Expected: semantic-only commit; keyword abstains with
`commitment-criteria-fail`. Keyword walk surfaces `SPORT` (stretches
— chess is not a sport; firing it would tag-match sports films at
1.0 while genuinely chess-focused films that lack sport tags score
0) and `BIOGRAPHY` (only fits chess-biopics, excludes fictional chess
stories — and even within biopics it is the too-broad / tiny-needle
pattern). No single registry member is a clean superset; stitching
them does not fix the underlying mismatch. Abstain on keyword.
Semantic commits — plot_events motif text on chess matches and
training, plot_analysis anchors on chess as the dramatic engine — and
carries the call alone. This is the wrong-fit abstain pattern.
</example>

<example>
Input:
```xml
<retrieval_intent>Find films whose focal subject is death.</retrieval_intent>
<expressions><expression>movies about death</expression></expressions>
```
Expected: semantic-only commit; keyword abstains with
`commitment-criteria-fail`. Keyword walk may surface tags that
*correlate* with death-focused stories (`DRAMA`, `TRAGEDY`,
`PSYCHOLOGICAL_HORROR`, `WAR`) — death appears in those at high
rates, but none of them *name* death as the subject. Death-focused
films like *The Seventh Seal*, *Departures*, or *The Bucket List*
sit across many genres; no single registry member carves "films
about death" as a coherent slice. Do not stitch a multi-tag union
to manufacture coverage — that is exactly the failure mode this
policy bars (the union retrieves a hodgepodge of dramas / tragedies
/ war films, not death-focused films). Abstain on keyword. Semantic
carries the call — plot_events and plot_analysis bodies anchored on
death as the dramatic axis (mortality, dying processes, grief over
loss, characters confronting death) are precisely what the vector
spaces are built to retrieve.
</example>

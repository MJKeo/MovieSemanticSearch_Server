# Few-shot examples

<example>
Input:
```xml
<retrieval_intent>Find films where zombies are present on screen as a story element.</retrieval_intent>
<expressions><expression>zombie movies</expression></expressions>
```
Expected: both endpoints commit. Keyword walk surfaces `ZOMBIE_HORROR`
— definitionally entails zombie presence, passes the superset test
cleanly. Commit `ZOMBIE_HORROR` alone (over-pull across zombie-comedy
and zombie-drama is acceptable because the tag still entails the
element). Semantic commits with plot_events motif syntax — "the
zombie. is a zombie. encounters a zombie. the zombies return." — to
catch films where zombies appear on screen but may not carry the
ZOMBIE_HORROR tag (e.g., genre-bent zombie films tagged primarily as
comedy or drama). The two layer: keyword sweeps tagged matches,
semantic catches the long tail.
</example>

<example>
Input:
```xml
<retrieval_intent>Find films featuring a heist as a present story element.</retrieval_intent>
<expressions><expression>heist movies</expression></expressions>
```
Expected: both endpoints commit. Keyword walk surfaces `HEIST` — the
activity is directly named, every HEIST film by definition contains
a heist. Commit `HEIST`. Semantic commits with plot_events motif body
— "the heist. a heist. the crew plans the heist. the heist unfolds."
— retrieving films where a heist is on screen as a concrete event,
including those without the HEIST tag. Activity-as-element follows
the same pattern as creature-as-element.
</example>

<example>
Input:
```xml
<retrieval_intent>Find films featuring horses as a present story element.</retrieval_intent>
<expressions><expression>movies with horses</expression></expressions>
```
Expected: semantic-only commit; keyword abstains with
`commitment-criteria-fail`. Keyword walk surfaces `WESTERN` — westerns
correlate with horse presence, but the correlation is not entailment:
a western set in a town can lack prominent horses, and a
horse-prominent contemporary drama (Black Beauty, Hidalgo, War Horse)
is not a western. Firing `WESTERN` would tag-match horseless westerns
at 1.0 while genuinely horse-prominent non-westerns score 0 — that is
stretching, not over-pull. No subset of walked members passes the
entailment gate. Abstain on keyword. Semantic commits with plot_events
motif body — "the horse. a horse. the horses. horses." — and carries
the call alone. This is the correlation-not-entailment abstain
pattern: a genre that *often features* the element is not the same
as a tag that *names* it.
</example>

<example>
Input:
```xml
<retrieval_intent>Find films featuring witches as a present story element.</retrieval_intent>
<expressions><expression>movies with witches</expression></expressions>
```
Expected: semantic-only commit; keyword abstains with
`commitment-criteria-fail`. Keyword walk surfaces `WITCH_HORROR`
(entails witches but is a narrow horror sub-form — excludes witch
comedies like *Hocus Pocus*, witch romances like *Practical Magic*,
and witch period dramas), `DARK_FANTASY` and `SUPERNATURAL_FANTASY`
(adjacent genres that often contain witches but do not entail them —
firing them is stretching, not over-pull). `WITCH_HORROR` alone has
real gaps for non-horror witch films; no broader entailing member
exists in the registry. Abstain on keyword rather than fire a
narrow-only commit that would bias the score toward horror. Semantic
commits with plot_events motif body — "the witch. a witch. the
witches. coven. spellcasting." — and carries the call. This is the
narrow-only-entailment abstain pattern.
</example>


# Few-shot examples - Target audience

<example>
Input:
```xml
<retrieval_intent>Find a movie packaged for family viewing.</retrieval_intent>
<expressions><expression>family movie</expression></expressions>
```
Reason first:
- Audience packaging is explicit.
- Family fit implies a maturity ceiling.
- No watch partner or occasion is named.

Expected: fire metadata for the maturity ceiling and keyword for the family
packaging member. Do not fire semantic; a bare audience label is not a
watch_context scenario.
</example>

<example>
Input:
```xml
<retrieval_intent>Find something appropriate to watch with my grandparents on Sunday.</retrieval_intent>
<expressions><expression>watch with my grandparents on Sunday</expression></expressions>
```
Reason first:
- The target is situational audience fit.
- Cross-generational viewing implies a gentle maturity ceiling.
- No registry member directly names "watch with grandparents."

Expected: fire metadata for the ceiling and semantic watch_context for the
companion / Sunday scenario. Do not force a keyword packaging tag.
</example>

<example>
Input:
```xml
<retrieval_intent>Find animation packaged for adult viewers.</retrieval_intent>
<expressions><expression>adult animated movies</expression></expressions>
```
Reason first:
- Audience packaging is explicit.
- Adult packaging does not imply a ceiling.
- The relevant signal is a registry packaging member.

Expected: fire keyword only. Do not fire metadata as a positive "adult rating"
selector. Do not fire semantic without a viewing scenario.
</example>

<example>
Input:
```xml
<retrieval_intent>Find a movie to put on with the kids tonight.</retrieval_intent>
<expressions><expression>with the kids tonight</expression></expressions>
```
Reason first:
- Watch partner and occasion are explicit.
- Kid-inclusive viewing implies a tighter maturity ceiling.
- The phrase is a scenario, not necessarily a packaging label.

Expected: fire metadata for the ceiling and semantic watch_context for the
with-kids tonight scenario. Fire keyword only if the target also explicitly
commits to family packaging.
</example>

<example>
Input:
```xml
<retrieval_intent>Find a coming-of-age story.</retrieval_intent>
<expressions><expression>coming-of-age story</expression></expressions>
```
Reason first:
- This names a story arc, not who the movie is packaged for.
- No maturity ceiling or watch scenario is grounded.

Expected: no-fire for this category. Story / thematic archetype owns the arc.
</example>

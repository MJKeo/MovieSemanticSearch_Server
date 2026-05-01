# Few-shot examples

<example>
Input:
```xml
<retrieval_intent>Find films with an anti-hero protagonist.</retrieval_intent>
<expressions><expression>anti-hero protagonist</expression></expressions>
```
Expected: preferred-only when the registry directly defines anti-hero.
Do not add Semantic just because characterization prose could also say it.
</example>

<example>
Input:
```xml
<retrieval_intent>Find films featuring a charming outlaw character type.</retrieval_intent>
<expressions><expression>lovable rogue</expression></expressions>
```
Expected: fallback-only if no direct archetype tag exists. Do not use
a broad adjacent tag that misses the charming-outlaw flavor.
</example>

<example>
Input:
```xml
<retrieval_intent>Find films with a quirky life-changing muse archetype.</retrieval_intent>
<expressions><expression>manic pixie dream girl</expression></expressions>
```
Expected: fallback-only to Semantic narrative_techniques unless a
direct tag exists.
</example>

<example>
Input:
```xml
<retrieval_intent>Find films about a redemption trajectory.</retrieval_intent>
<expressions><expression>redemption arc</expression></expressions>
```
Expected: no-fire. Arc trajectory belongs to Story / thematic archetype.
</example>

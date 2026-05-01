# Few-shot examples

<example>
Input:
```xml
<retrieval_intent>Find films praised for a specific craft or effect aspect.</retrieval_intent>
<expressions><expression>praised for tension</expression></expressions>
```
Expected: preferred-only Semantic reception. The target is praise for
an aspect, not mere presence of tension.
</example>

<example>
Input:
```xml
<retrieval_intent>Find films criticized for a pacing flaw.</retrieval_intent>
<expressions><expression>criticized as plodding</expression></expressions>
```
Expected: preferred-only Semantic reception with criticism language.
Do not route to runtime or pacing tags.
</example>

<example>
Input:
```xml
<retrieval_intent>Find films praised for performances, with an optional crisp aspect tag if one directly applies.</retrieval_intent>
<expressions><expression>praised for performances</expression></expressions>
```
Expected: Semantic reception. Keyword only if a direct aspect tag
captures performances without changing praise into presence.
</example>

<example>
Input:
```xml
<retrieval_intent>Find films with broad whole-work reputation.</retrieval_intent>
<expressions><expression>cult classic</expression></expressions>
```
Expected: no-fire. Whole-work cultural status is not aspect-level
praise or criticism.
</example>

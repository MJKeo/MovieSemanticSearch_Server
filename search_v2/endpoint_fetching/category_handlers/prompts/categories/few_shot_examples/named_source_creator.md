# Few-shot examples - Named source creator

<example>
Input:
```xml
<retrieval_intent>Find films adapted from Stephen King novels; the novel flag is handled separately.</retrieval_intent>
<expressions><expression>Stephen King</expression></expressions>
```
Expected: fire semantic; source-creator name in plot/reception source
context. Do not emit the novel flag here.
</example>

<example>
Input:
```xml
<retrieval_intent>Find movies based on Shakespeare plays; the stage/source flag is handled separately.</retrieval_intent>
<expressions><expression>Shakespeare</expression></expressions>
```
Expected: fire semantic; source creator William Shakespeare.
</example>

<example>
Input:
```xml
<retrieval_intent>Find Jane Austen adaptations.</retrieval_intent>
<expressions><expression>Jane Austen</expression></expressions>
```
Expected: fire semantic; named source creator. Medium flag belongs elsewhere
if present.
</example>

<example>
Input:
```xml
<retrieval_intent>Find Star Wars novelizations.</retrieval_intent>
<expressions><expression>Star Wars</expression></expressions>
```
Expected: no-fire; named franchise source, not source-material creator.
</example>


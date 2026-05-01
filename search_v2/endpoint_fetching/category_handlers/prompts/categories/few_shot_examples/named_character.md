# Few-shot examples - Named character

<example>
Input:
```xml
<retrieval_intent>Find films centered on the character Batman.</retrieval_intent>
<expressions><expression>Batman</expression></expressions>
```
Expected: fire character lookup; central prominence; include plausible
credited aliases such as Bruce Wayne when useful.
</example>

<example>
Input:
```xml
<retrieval_intent>Find movies where Yoda appears, including minor appearances.</retrieval_intent>
<expressions><expression>Yoda</expression></expressions>
```
Expected: fire character lookup; default prominence.
</example>

<example>
Input:
```xml
<retrieval_intent>Find films featuring Hermione Granger.</retrieval_intent>
<expressions><expression>Hermione Granger</expression></expressions>
```
Expected: fire character lookup; default prominence; keep the credited
character name narrow.
</example>

<example>
Input:
```xml
<retrieval_intent>Find films with a lovable rogue protagonist.</retrieval_intent>
<expressions><expression>lovable rogue</expression></expressions>
```
Expected: no-fire; character archetype, not a named persona.
</example>


# Few-shot examples - Character-franchise

<example>
Input:
```xml
<retrieval_intent>Find Batman movies.</retrieval_intent>
<expressions><expression>Batman</expression></expressions>
```
Expected: fire both paths. One shared referent: Batman. Character forms use
credited Batman/Bruce Wayne-style persona names when supported; franchise
forms use Batman-series names. Central character framing.
</example>

<example>
Input:
```xml
<retrieval_intent>Find James Bond films, especially the main series.</retrieval_intent>
<expressions><expression>James Bond</expression></expressions>
```
Expected: fire both paths. Character and franchise forms both target Bond,
with main-series intent handled on the franchise side. Do not add actor
names.
</example>

<example>
Input:
```xml
<retrieval_intent>Find Sherlock Holmes adaptations from books.</retrieval_intent>
<expressions><expression>Sherlock Holmes</expression></expressions>
```
Expected: fire both paths for Sherlock Holmes. Keep "books" out of this
payload; adaptation-source handles that as a separate call.
</example>

<example>
Input:
```xml
<retrieval_intent>Find movies where Hermione Granger appears.</retrieval_intent>
<expressions><expression>Hermione Granger</expression></expressions>
```
Expected: no-fire. Hermione is a named character, but not the franchise
anchor.
</example>

<example>
Input:
```xml
<retrieval_intent>Find films in the Star Wars universe.</retrieval_intent>
<expressions><expression>Star Wars</expression></expressions>
```
Expected: no-fire. Star Wars is franchise-lineage, not a named character
that anchors its own character-franchise fan-out.
</example>

# Few-shot examples - Adaptation source flag

<example>
Input:
```xml
<retrieval_intent>Find films based on true events.</retrieval_intent>
<expressions><expression>true story</expression></expressions>
```
Expected: fire keyword endpoint; source-material type TRUE_STORY.
</example>

<example>
Input:
```xml
<retrieval_intent>Find movies adapted from novels.</retrieval_intent>
<expressions><expression>novel adaptation</expression></expressions>
```
Expected: fire keyword endpoint; source-material type NOVEL_ADAPTATION.
</example>

<example>
Input:
```xml
<retrieval_intent>Find films adapted from Stephen King novels; the author name is handled by a separate source-creator trait.</retrieval_intent>
<expressions><expression>novels</expression></expressions>
```
Expected: fire keyword endpoint for NOVEL_ADAPTATION only.
</example>

<example>
Input:
```xml
<retrieval_intent>Find Star Wars movies.</retrieval_intent>
<expressions><expression>Star Wars</expression></expressions>
```
Expected: no-fire; bare franchise identity is not an adaptation-source flag.
</example>


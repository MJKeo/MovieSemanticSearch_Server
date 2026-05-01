# Few-shot examples - Sensitive content

<example>
Input:
```xml
<retrieval_intent>Avoid movies where an animal dies.</retrieval_intent>
<expressions><expression>nothing where the dog dies</expression></expressions>
```
Reason first:
- Concrete content axis: animal death.
- Binary hard exclusion.
- Registry flag directly covers the axis.

Expected: fire keyword for the animal-death flag only. Do not fire metadata;
rating is not the issue. Do not soften this into semantic intensity.
</example>

<example>
Input:
```xml
<retrieval_intent>Find a sad movie where the dog actually dies.</retrieval_intent>
<expressions><expression>where the dog actually dies</expression></expressions>
```
Reason first:
- Concrete content axis: animal death.
- Binary hard inclusion.
- The emotional effect is sibling intent, not this category's payload.

Expected: fire keyword for the animal-death flag only, with parent polarity
positive. Do not fire semantic just because the result may be sad.
</example>

<example>
Input:
```xml
<retrieval_intent>Prefer horror movies that are not too bloody.</retrieval_intent>
<expressions><expression>not too bloody</expression></expressions>
```
Reason first:
- Concrete content axis: blood / gore.
- "Not too" makes it a gradient, not a hard ban.
- No clean binary registry flag is available from the phrasing alone.

Expected: fire semantic viewer_experience.disturbance_profile for blood/gore
intensity. Do not force a genre tag as a content flag. Do not use metadata for
a specific axis.
</example>

<example>
Input:
```xml
<retrieval_intent>Find an action movie with family-friendly intensity.</retrieval_intent>
<expressions><expression>family-friendly intensity</expression></expressions>
```
Reason first:
- The target is global content intensity, not audience packaging.
- The surface implies a rating ceiling.
- No specific content axis is named.

Expected: fire metadata for the maturity ceiling only. Keyword FAMILY belongs
to Target audience, not this content-intensity slice.
</example>

<example>
Input:
```xml
<retrieval_intent>Find something fun, nothing heavy.</retrieval_intent>
<expressions><expression>nothing heavy</expression></expressions>
```
Reason first:
- "Heavy" is vague mood/weight.
- No concrete content axis, rating ceiling, or disturbance gradient is named.

Expected: no-fire for this category. Experiential tone may own the lightness
preference.
</example>

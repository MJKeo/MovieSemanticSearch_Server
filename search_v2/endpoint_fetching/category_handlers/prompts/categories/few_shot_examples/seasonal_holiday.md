# Seasonal / holiday - few-shot examples

<example>
Input:
```xml
<retrieval_intent>Find films framed as Christmas movies.</retrieval_intent>
<expressions><expression>Christmas movies</expression></expressions>
```
Expected: fire semantic on `watch_context` (Christmas viewing
occasion) and `plot_events` (Christmas-set stories). The phrase
covers both viewing-time and story-setting framings; populate both
spaces.
</example>

<example>
Input:
```xml
<retrieval_intent>Find a movie for Halloween-night viewing.</retrieval_intent>
<expressions><expression>something for Halloween night</expression></expressions>
```
Expected: fire semantic on `watch_context` for the Halloween
viewing-occasion fit. Add `plot_events` only if a Halloween-night
narrative setting is genuinely part of the meaning.
</example>

<example>
Input:
```xml
<retrieval_intent>Find documentaries about Valentine's Day as a subject.</retrieval_intent>
<expressions><expression>Valentine's Day documentaries</expression></expressions>
```
Expected: no-fire. Holiday-as-subject is owned by Central topic,
not seasonal framing.
</example>

<example>
Input:
```xml
<retrieval_intent>Find films with a winter-snowed setting.</retrieval_intent>
<expressions><expression>winter setting with snow</expression></expressions>
```
Expected: no-fire. Pure narrative time/place setting without a
seasonal-viewing framing belongs to Narrative setting.
</example>

# Few-shot examples - Audio language

<example>
Input:
```xml
<retrieval_intent>Find thrillers with Korean audio.</retrieval_intent>
<expressions><expression>Korean-language</expression></expressions>
```
Expected: fire metadata; audio language Korean.
</example>

<example>
Input:
```xml
<retrieval_intent>Find Spanish-language dramas.</retrieval_intent>
<expressions><expression>Spanish-language</expression></expressions>
```
Expected: fire metadata; audio language Spanish.
</example>

<example>
Input:
```xml
<retrieval_intent>Find Japanese movies with subtitles.</retrieval_intent>
<expressions><expression>in Japanese with subtitles</expression></expressions>
```
Expected: fire metadata; audio language Japanese; subtitles confirm
non-English audio intent.
</example>

<example>
Input:
```xml
<retrieval_intent>Find Korean cinema.</retrieval_intent>
<expressions><expression>Korean cinema</expression></expressions>
```
Expected: no-fire; this is cultural tradition, not audio language.
</example>


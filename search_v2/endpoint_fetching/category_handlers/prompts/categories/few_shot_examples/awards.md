# Few-shot examples - Award records

<example>
Input:
```xml
<retrieval_intent>Find films that won an Oscar; release recency is handled separately.</retrieval_intent>
<expressions><expression>Oscar winner</expression></expressions>
```
Expected: fire award endpoint; Oscar prize, winner outcome, floor at 1.
</example>

<example>
Input:
```xml
<retrieval_intent>Find films nominated at BAFTA.</retrieval_intent>
<expressions><expression>BAFTA nominated</expression></expressions>
```
Expected: fire award endpoint; BAFTA ceremony/prize as appropriate,
nominee outcome, floor at 1.
</example>

<example>
Input:
```xml
<retrieval_intent>Find heavily decorated films.</retrieval_intent>
<expressions><expression>heavily decorated</expression></expressions>
```
Expected: fire award endpoint; generic positive award recognition,
threshold scoring.
</example>

<example>
Input:
```xml
<retrieval_intent>Find critically acclaimed films.</retrieval_intent>
<expressions><expression>critically acclaimed</expression></expressions>
```
Expected: no-fire; reception quality, not award records.
</example>


# Few-shot examples - Franchise / universe lineage

<example>
Input:
```xml
<retrieval_intent>Find films in the Marvel Cinematic Universe.</retrieval_intent>
<expressions><expression>MCU</expression></expressions>
```
Expected: fire franchise endpoint; shared-universe umbrella; no studio axis.
</example>

<example>
Input:
```xml
<retrieval_intent>Find Star Wars prequel films.</retrieval_intent>
<expressions><expression>Star Wars</expression><expression>prequel</expression></expressions>
```
Expected: fire franchise endpoint; franchise anchor plus prequel lineage
position/subgroup as supported.
</example>

<example>
Input:
```xml
<retrieval_intent>Find the original Scarface, not the remake.</retrieval_intent>
<expressions><expression>Scarface</expression><expression>original</expression></expressions>
```
Expected: fire franchise endpoint; named lineage with original/remake
positioning. Do not use adaptation-source remake flag.
</example>

<example>
Input:
```xml
<retrieval_intent>Find good remakes.</retrieval_intent>
<expressions><expression>remake</expression></expressions>
```
Expected: no-fire; no named franchise anchor.
</example>


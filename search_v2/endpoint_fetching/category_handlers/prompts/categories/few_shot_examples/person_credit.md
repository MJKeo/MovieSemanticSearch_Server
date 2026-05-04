# Few-shot examples - Person credit

Use these for shape calibration, not as a closed list. Note how
`person_exploration` is filled in as a literal template, not prose.

<example>
Input:
```xml
<retrieval_intent>Find films starring Tom Hanks.</retrieval_intent>
<expressions><expression>Tom Hanks</expression></expressions>
```
person_exploration:
  Films: Forrest Gump (1994), Cast Away (2000), Saving Private Ryan (1998)
  Credit per film:
    - Forrest Gump: Tom Hanks
    - Cast Away: Tom Hanks
    - Saving Private Ryan: Tom Hanks
  Distinct forms: Tom Hanks
  Predominant role: actor

forms: ["Tom Hanks"]. person_category: actor. prominence_mode: lead
(starring language present). Single consistent credit across films →
single form; no over-generation.
</example>

<example>
Input:
```xml
<retrieval_intent>Find films directed by Kathryn Bigelow.</retrieval_intent>
<expressions><expression>Kathryn Bigelow</expression></expressions>
```
person_exploration:
  Films: The Hurt Locker (2008), Zero Dark Thirty (2012), Detroit (2017)
  Credit per film:
    - The Hurt Locker: Kathryn Bigelow
    - Zero Dark Thirty: Kathryn Bigelow
    - Detroit: Kathryn Bigelow
  Distinct forms: Kathryn Bigelow
  Predominant role: director

forms: ["Kathryn Bigelow"]. person_category: director.
</example>

<example>
Input:
```xml
<retrieval_intent>Find films with scores composed by John Williams.</retrieval_intent>
<expressions><expression>John Williams</expression></expressions>
```
person_exploration:
  Films: Star Wars (1977), Jurassic Park (1993), Schindler's List (1993)
  Credit per film:
    - Star Wars: John Williams
    - Jurassic Park: John Williams
    - Schindler's List: John Williams
  Distinct forms: John Williams
  Predominant role: composer

forms: ["John Williams"]. person_category: composer.
</example>

<example>
Input:
```xml
<retrieval_intent>Find films starring 50 Cent.</retrieval_intent>
<expressions><expression>50 Cent</expression></expressions>
```
person_exploration:
  Films: Get Rich or Die Tryin' (2005), Twelve (2010), Den of Thieves (2018)
  Credit per film:
    - Get Rich or Die Tryin': 50 Cent
    - Twelve: Curtis Jackson
    - Den of Thieves: Curtis Jackson, 50 Cent
  Distinct forms: 50 Cent, Curtis Jackson
  Predominant role: actor

forms: ["50 Cent", "Curtis Jackson"]. person_category: actor. The
literal Den of Thieves credit ("Curtis '50 Cent' Jackson") bundles a
stage name into a legal-name credit; the per-film line splits it into
two atomic entries so each can match the credit dictionary exactly.
</example>

<example>
Input:
```xml
<retrieval_intent>Find films shot by Roger Deakins.</retrieval_intent>
<expressions><expression>Roger Deakins</expression></expressions>
```
Expected: no-fire; cinematographer is not an indexed role for this endpoint.
</example>

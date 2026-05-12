# Few-shot examples - Named source creator

Use these for shape calibration. The only decision that affects
retrieval is the `forms` list — these examples focus there.

<example>
Input:
```xml
<retrieval_intent>Find films adapted from Stephen King novels; the novel flag is handled separately.</retrieval_intent>
<expressions><expression>Stephen King</expression></expressions>
```
person_exploration:
  Films: The Shawshank Redemption (1994), The Shining (1980), It (2017), Misery (1990), The Mist (2007)
  Credit per film:
    - The Shawshank Redemption: Stephen King
    - The Shining: Stephen King
    - It: Stephen King
    - Misery: Stephen King
    - The Mist: Stephen King
  Distinct forms: Stephen King

forms: ["Stephen King"]. Single consistent credit across films.
</example>

<example>
Input:
```xml
<retrieval_intent>Find Tolkien adaptations.</retrieval_intent>
<expressions><expression>Tolkien</expression></expressions>
```
person_exploration:
  Films: The Lord of the Rings: The Fellowship of the Ring (2001), The Two Towers (2002), The Hobbit: An Unexpected Journey (2012)
  Credit per film:
    - The Fellowship of the Ring: J.R.R. Tolkien
    - The Two Towers: J.R.R. Tolkien
    - An Unexpected Journey: J.R.R. Tolkien
  Distinct forms: J.R.R. Tolkien, John Ronald Reuel Tolkien

forms: ["J.R.R. Tolkien", "John Ronald Reuel Tolkien"]. Two atomic
forms — some credit blocks expand the full given name; including
both avoids silently dropping films credited under the long form.
</example>

<example>
Input:
```xml
<retrieval_intent>Find movies based on Shakespeare plays; the stage/source flag is handled separately.</retrieval_intent>
<expressions><expression>Shakespeare</expression></expressions>
```
person_exploration:
  Films: Romeo + Juliet (1996), Hamlet (1996), Macbeth (2015), Much Ado About Nothing (1993)
  Credit per film:
    - Romeo + Juliet: William Shakespeare
    - Hamlet: William Shakespeare
    - Macbeth: William Shakespeare
    - Much Ado About Nothing: William Shakespeare
  Distinct forms: William Shakespeare

forms: ["William Shakespeare"].
</example>

<example>
Input:
```xml
<retrieval_intent>Find Philip K. Dick adaptations.</retrieval_intent>
<expressions><expression>Philip K. Dick</expression></expressions>
```
person_exploration:
  Films: Blade Runner (1982), Minority Report (2002), Total Recall (1990), The Adjustment Bureau (2011), A Scanner Darkly (2006)
  Credit per film:
    - Blade Runner: Philip K. Dick
    - Minority Report: Philip K. Dick
    - Total Recall: Philip K. Dick
    - The Adjustment Bureau: Philip K. Dick
    - A Scanner Darkly: Philip K. Dick
  Distinct forms: Philip K. Dick, Philip Kindred Dick

forms: ["Philip K. Dick", "Philip Kindred Dick"]. The expanded
middle-name form shows up on a minority of credit blocks; include
it so films that use it are not silently dropped.
</example>

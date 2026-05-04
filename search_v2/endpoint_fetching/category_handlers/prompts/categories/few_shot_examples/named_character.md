# Few-shot examples - Named character

Note how `character_exploration` is filled in as a literal template,
not prose.

<example>
Input:
```xml
<retrieval_intent>Find movies where Yoda appears, including minor appearances.</retrieval_intent>
<expressions><expression>Yoda</expression></expressions>
```
character_exploration:
  Films: The Empire Strikes Back (1980), Return of the Jedi (1983), Attack of the Clones (2002)
  Credit per film:
    - The Empire Strikes Back: Yoda
    - Return of the Jedi: Yoda
    - Attack of the Clones: Yoda
  Distinct forms: Yoda

forms: ["Yoda"]. prominence_mode: default. Single consistent credit
across films: single form; no over-generation.
</example>

<example>
Input:
```xml
<retrieval_intent>Find films featuring Hermione Granger.</retrieval_intent>
<expressions><expression>Hermione Granger</expression></expressions>
```
character_exploration:
  Films: Sorcerer's Stone (2001), Prisoner of Azkaban (2004), Deathly Hallows (2011)
  Credit per film:
    - Sorcerer's Stone: Hermione Granger
    - Prisoner of Azkaban: Hermione Granger
    - Deathly Hallows: Hermione Granger
  Distinct forms: Hermione Granger

forms: ["Hermione Granger"]. prominence_mode: default. Keep the
credited character name narrow.
</example>

<example>
Input:
```xml
<retrieval_intent>Find films where Voldemort appears.</retrieval_intent>
<expressions><expression>Voldemort</expression></expressions>
```
character_exploration:
  Films: Sorcerer's Stone (2001), Goblet of Fire (2005), Deathly Hallows (2011)
  Credit per film:
    - Sorcerer's Stone: Voldemort
    - Goblet of Fire: Lord Voldemort
    - Deathly Hallows: Lord Voldemort, Tom Riddle
  Distinct forms: Lord Voldemort, Voldemort, Tom Riddle

forms: ["Lord Voldemort", "Voldemort", "Tom Riddle"]. prominence_mode:
default. The walk surfaces the civilian-name credit that the queried
short form alone would miss; the multi-credit film line splits the
slash-combined credit into atomic entries.
</example>

<example>
Input:
```xml
<retrieval_intent>Find films with a lovable rogue protagonist.</retrieval_intent>
<expressions><expression>lovable rogue</expression></expressions>
```
Expected: no-fire; character archetype, not a named persona.
</example>

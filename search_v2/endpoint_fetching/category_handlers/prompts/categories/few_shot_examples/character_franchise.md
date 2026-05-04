# Few-shot examples - Character-franchise

Note how `character_form_exploration` and `franchise_form_exploration`
are each filled in as a literal template, not prose. The two walks
are independent — one is per-film cast credits, the other is series /
umbrella / subgroup titles. Do not produce a single referent
description and reuse it for both.

<example>
Input:
```xml
<retrieval_intent>Find James Bond films, especially the main series.</retrieval_intent>
<expressions><expression>James Bond</expression></expressions>
```
character_form_exploration:
  Films: Goldfinger (1964), Casino Royale (2006), Skyfall (2012)
  Credit per film:
    - Goldfinger: James Bond
    - Casino Royale: James Bond
    - Skyfall: James Bond
  Distinct forms: James Bond

franchise_form_exploration:
  Series: James Bond, 007
  Umbrella: none
  Subgroups: Daniel Craig era, Sean Connery era
  Distinct forms: James Bond, 007

character_forms: ["James Bond"]
franchise_forms: ["James Bond", "007"]
Fire both paths. Single consistent character credit; franchise is
known by both the character name and the codename.
</example>

<example>
Input:
```xml
<retrieval_intent>Find Indiana Jones movies.</retrieval_intent>
<expressions><expression>Indiana Jones</expression></expressions>
```
character_form_exploration:
  Films: Raiders of the Lost Ark (1981), Last Crusade (1989), Crystal Skull (2008)
  Credit per film:
    - Raiders of the Lost Ark: Indy
    - Last Crusade: Indiana Jones, Henry Jones Jr.
    - Crystal Skull: Indiana Jones
  Distinct forms: Indiana Jones, Indy, Henry Jones Jr.

franchise_form_exploration:
  Series: Indiana Jones
  Umbrella: none
  Subgroups: none
  Distinct forms: Indiana Jones

character_forms: ["Indiana Jones", "Indy", "Henry Jones, Jr."]
franchise_forms: ["Indiana Jones"]
Fire both paths. The character walk surfaces credit-string drift
across films ("Indy" in Raiders, "Henry Jones, Jr." in Last Crusade)
that abstract aliasing would miss.
</example>

<example>
Input:
```xml
<retrieval_intent>Find Iron Man movies.</retrieval_intent>
<expressions><expression>Iron Man</expression></expressions>
```
character_form_exploration:
  Films: Iron Man (2008), The Avengers (2012), Avengers: Endgame (2019)
  Credit per film:
    - Iron Man: Tony Stark, Iron Man
    - The Avengers: Tony Stark
    - Avengers: Endgame: Tony Stark
  Distinct forms: Tony Stark, Iron Man

franchise_form_exploration:
  Series: Iron Man
  Umbrella: Marvel Cinematic Universe
  Subgroups: Infinity Saga, Phase 1
  Distinct forms: Iron Man, Marvel Cinematic Universe

character_forms: ["Tony Stark", "Iron Man"]
franchise_forms: ["Iron Man", "Marvel Cinematic Universe"]
Fire both paths. Tony Stark is the dominant cast-list credit across
the MCU; emitting only the queried name "Iron Man" would drop nearly
every appearance. The franchise side names both the standalone Iron
Man series and the MCU umbrella.
</example>

<example>
Input:
```xml
<retrieval_intent>Find Sherlock Holmes adaptations from books.</retrieval_intent>
<expressions><expression>Sherlock Holmes</expression></expressions>
```
character_form_exploration:
  Films: Sherlock Holmes (2009), Mr. Holmes (2015), Enola Holmes (2020)
  Credit per film:
    - Sherlock Holmes: Sherlock Holmes
    - Mr. Holmes: Sherlock Holmes
    - Enola Holmes: Sherlock Holmes
  Distinct forms: Sherlock Holmes

franchise_form_exploration:
  Series: Sherlock Holmes
  Umbrella: none
  Subgroups: none
  Distinct forms: Sherlock Holmes

character_forms: ["Sherlock Holmes"]
franchise_forms: ["Sherlock Holmes"]
Fire both paths. Keep "books" out of either payload; adaptation-source
handles that as a separate call.
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

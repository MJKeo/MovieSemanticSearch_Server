## Cross-cutting patterns (how the embedded text is written)

These are consistent across the samples:

* **Everything is “search-text”**: either a dense paragraph, or **comma-separated phrase lists** meant to embed well (lots of short descriptors, some redundancy).
* **Lowercase bias** in plot summaries and lists is common (especially plot_events + viewer_experience).
* **Redundancy is intentionally tolerated** (e.g., repeating “not cynical”, repeating “hand-drawn animation” twice, repeating “epilogue time jump / epilogue”). 
* Many vectors **do not include justification text** even if justification exists in schema (explicitly noted in the schema). 
* Clear “template-y” headers appear in **Production** and **Reception** (“# Production”, “# Cast and Characters”, “Praises: … / Complaints: …”).  

---

## 1) Plot Events Vector

### What it represents

A **literal, spoiler-friendly recounting** of what happens, plus **in-universe setting** and **major characters with motivations** (as shown on screen). The schema explicitly defines: detailed plot summary, setting phrase, and major characters (name/description/motivations). 

### What data belongs here

* Chronological plot beats (including ending)
* In-universe time/place (“1936; south america, nepal, cairo/tanis…”)
* Major characters: who they are + what they want (grounded in plot)

### How it’s worded in the samples

* **Long, chronological paragraph(s)** with lots of concrete events and proper nouns.
* Then a **single-line setting**.
* Then **character blocks**: `name: description … motivations: …` 

### Examples of “correct fit”

* *Raiders*: trap temple → Tanis dig → Ark opening → warehouse storage, plus character motivations. 
* *Terrifier 3*: explicit kill chain + supernatural rebirth + portal cliffhanger, plus motivations. 

---

## 2) Plot Analysis Vector

### What it represents

A **generalized “what it’s about”** vector: high-level arc + core concept + genre signatures + conflict scale + **generic character-arc labels** + themes + lessons + keywords. (Character names/descriptions are intentionally *not* part of the arc text per schema.) 

### What data belongs here

* Generalized plot overview (not scene-by-scene)
* Core concept label (“race to secure supernatural artifact”)
* Genre signatures + genre phrasing
* Conflict scale (“personal conflict”, “large-scale conflict”, “community conflict”)
* Arc labels (“redemption arc”, “disillusionment arc”, “defiant survivor”)
* Theme labels + lesson labels
* Overall keywords

### How it’s worded in the samples

* Usually starts with **1 paragraph**: a compact “shape of story” summary.
* Then a **single-sentence core concept label** (often like `X: Y`).
* Then **staccato lines**: genre(s), conflict scale, arc labels, theme labels, lesson labels, then a final genre/keyword pile.  

### Notable nuance from samples

Plot Analysis often “reads” like: *premise + transformation + what it argues*, without making it a review. That matches your intent for plot_analysis and the schema design. 

---

## 3) Narrative Techniques Vector

### What it represents

“Film-nerd mechanics”: **how the story is told**, not what happens and not whether it’s good. The schema explicitly lists POV, delivery/structure, archetypes, information control, characterization methods, generic arc terms, audience perception, stakes design, thematic delivery, meta techniques, plot devices. 

### What data belongs here

* POV labels (first-person, third-person limited, protagonist-centered)
* Structure labels (linear chronology, framed story, episodic set-piece structure, time skip)
* Information control (dramatic irony, unreliable narrator, subjective ambiguity)
* Devices (Chekhov’s gun, macguffin-driven stakes, cold open, cliffhanger)
* Character-perception and archetype labels (sympathetic innocent lead, manichean villain, lovable rogue)
* Stakes design labels (ticking clock deadline, escalation ladder)

### How it’s worded in the samples

* Almost purely **comma-separated technique tags**.
* Strong preference for **movie-agnostic craft vocabulary** (things you’d see in a writing class).
* It happily mixes “macro structure” + “micro devices” in one flat list.  

---

## 4) Viewer Experience Vector

### What it represents

The **felt experience**: emotions, tone, tension/energy, cognitive load, disturbance, sensory load, emotional volatility, and “ending aftertaste”. The schema explicitly supports both **terms** and **negations** across these subdimensions. 

### What data belongs here

* Emotional palette (“heartwarming”, “bleak thrills”, “tearjerker”)
* Tone/self-seriousness (“earnest”, “campy gore”, “not pretentious”)
* Tension/adrenaline (“edge of your seat”, “relaxed”, “constant shock spikes”)
* Cognitive complexity (“easy to follow”, “not dense”, “not confusing”)
* Disturbance (“gore-soaked”, “snake phobia trigger”, “no jump scares”)
* Sensory load (“overstimulating”, “loud shocks”, “not quiet”)
* Volatility (“tonal whiplash”, “mood swings”)
* Aftertaste (“nostalgic aftertaste”, “emotional hangover”, “left me unsettled”) 

### How it’s worded in the samples

* **Massive comma-separated adjective/phrase list**.
* Heavy use of **explicit negations** (e.g., “not cynical”, “not depressing”, “no gore”, “not confusing”).  
* Often includes **contrasting dynamics** (“calm with spikes”, “laugh then cry”, “tonal swings”) which is perfect for emotional-volatility embedding. 

### What stands out most

This vector is the most “dense” and seems built to answer queries like:

* “not too intense, no jump scares, cozy”
* “visceral, squirm-inducing gore, overstimulating”
  That aligns with your viewer_experience scope prompt too. 

---

## 5) Watch Context Vector

### What it represents

The **use-case lens**: why/when to watch, internal motivations, external motivations, watch scenarios, and “key feature draws” (as selection criteria). Schema defines these buckets explicitly. 

### What data belongs here

* Self-experience motivations (“turn my brain off”, “get a good cry”, “feel inspired”)
* External motivations (“impress film buffs”, “sparks conversation”, “must-see classic”)
* Key movie feature draws (“iconic score”, “beautiful scenery”, “creative kills”, “powerful lead performance”)
* Watch scenarios (“date night”, “family movie night”, “Christmas Eve watch”, “background at a party”) 

### How it’s worded in the samples

* **Comma-separated short phrases**.
* Often mixes motivations + scenarios + feature draws in one flat list (not bucketed).
* Includes occasional **strong situational specificity** (e.g., “background while wrapping gifts”, “christmas horror watch”).  

### Prompt alignment

This matches your Watch Context scope: motivations + scenarios + feature draws, plus optional high-confidence inferences. 

---

## 6) Production Vector

### What it represents

Objective **real-world making-of facts**: countries, companies, filming locations, languages, release decade bucket, budget bucket, production keywords/mediums, source-of-inspiration/adaptation, and top cast/crew. The schema enumerates this explicitly. 

### What data belongs here

* Produced in + production companies
* Filming locations
* Primary language + “audio also available for …”
* Release decade bucket (“1980s, 80s”; “2020s, 20s”)
* Budget scale bucket (“small budget”)
* Medium/production tech (“live action”, “practical special effects”, “hand-drawn animation”, “computer animation”)
* Adaptation/source (“based on a novel”, “based on a cultural/folklore tradition”, “sequel/continuation…”) 

### How it’s worded in the samples

* Strong formatting template:

  * `# Production:` block
  * then `# Cast and Characters:` block
* Cast/crew lines are standardized (“Directed by… Written by… Produced by… Music composed by… Main actors… Main characters…”)  

### One key constraint you’ve already enforced elsewhere

Your production prompts emphasize “making-of metadata only” and explicitly list what counts (medium, decade, origin/filming, language, cast/crew, studios, budget, adapted-from). 

---

## 7) Reception Vector

### What it represents

The “what people thought” layer: **acclaim tier** + a concise reception summary + generic praise/complaint attributes. Schema defines reception tier and praise/complaint attributes, and notes justification is not embedded. 

### What data belongs here

* Reception tier label: “universally acclaimed”, “mixed or average reviews”, “generally favorable reviews”
* A short paragraph summarizing the consensus (praise + criticism)
* Then:

  * `Praises: ...`
  * `Complaints: ...`  

### How it’s worded in the samples

* Starts with **just the tier phrase on its own line**.
* Then 1–2 sentence **natural language** summary.
* Then “Praises/Complaints” lines with **generic, query-friendly attributes** (no proper nouns). 

### Prompt alignment

Your reception prompt explicitly wants movie-agnostic, query-friendly attributes and a concise evaluative summary. 

---

## 8) Dense Anchor Vector (not shown in these 5 RTFs, but defined in schema)

These five samples don’t include the anchor text block, but your schema defines it as the **broad recall “movie card”**: identity + partial plot analysis + genres + keywords + production basics + cast/crew + themes/lessons + a slice of vibe + maturity guidance + reception tier + reception attributes.  

---

## What you can reliably conclude about “each vector collection” as a search space

* **plot_events** is best for: *“two strangers handcuffed escape Tokyo in one night”*, “Ark gets opened and Nazis melt”, “what happens at the end”.
* **plot_analysis** is best for: *theme/lesson/arc* queries and “what it’s about in one sentence” (“race to secure supernatural artifact”, “kindness sparks community transformation”).  
* **narrative_techniques** is best for: structural/trope/POV/mechanism queries (“framed story”, “unreliable narrator”, “macguffin-driven stakes”, “time skip”).  
* **viewer_experience** is best for: vibe + exclusions + sensory constraints (“not depressing”, “no jump scares”, “overstimulating loud shocks”, “tearjerker”).  
* **watch_context** is best for: scenarios and “why watch” intent (“date night”, “family movie night”, “turn my brain off”, “background at a party”).  
* **production** is best for: factual constraints (language, decade, medium, adaptation, location, director) and is written in a highly structured template.  
* **reception** is best for: “critics hated it / mixed reviews / overrated”, and “praised X, criticized Y” search.  

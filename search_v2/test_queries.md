# Step 2 Test Queries

A spread of queries written the way people actually type into a
phone keyboard or TV remote — casual, lowercase, with typos,
abbreviations, and run-ons. Each query targets a distinct edge case
the step-2 pre-pass must handle correctly. Examples already used in
the system prompt are deliberately avoided.

For each query the **Tests** line names the specific behavior under
stress, so failures map directly back to the rule that broke.

---

## 1. Single bare attribute

**Query:** `scary`

**Tests:** Minimal input. The pre-pass must produce one attribute
fragment and a single coverage entry without inventing extra
constraints (no implicit occasion, audience, or era).

---

## 2. Director name with typo + runtime polarity

**Query:** `quentin terantino movies that arent too long`

**Tests:** Typo preservation in `query_text` ("terantino"), a
ROLE_MARKER-style binding to director credit, and a
POLARITY_MODIFIER ("not too") on runtime — runtime must surface as
its own attribute, not get folded into the director fragment.

---

## 3. Texting shorthand and abbreviations

**Query:** `luv stories w sad endings 4 a rainy night`

**Tests:** Non-standard spelling ("luv", "w", "4") preserved
verbatim. Three distinct attributes (genre/kind-of-story, narrative
ending, occasion bundle) where "w" is filler and "4 a rainy night"
is an implicit occasion bundle.

---

## 4. Long run-on voice query, polarity stacking

**Query:** `i wanna watch something with my mom shes 65 and likes cozy mysteries nothing too dark or scary please`

**Tests:** No punctuation, conversational filler, co-viewing
occasion with a non-"brother" family member, target-audience age
specifier, named sub-genre ("cozy mysteries"), and a
POLARITY_MODIFIER ("nothing too") that distributes across two
attributes ("dark" and "scary"). Specificity must be preserved
("mom", "65", not generalized to "parent" / "older").

---

## 5. Two simultaneous role markers on different people

**Query:** `directed by david lynch starring kyle maclachlan`

**Tests:** Two ROLE_MARKER modifiers on two separate person
fragments — director credit binding vs. lead-actor credit binding.
Both atomic_rewrites must reflect their own role, and neither
person should leak into the other's fragment.

---

## 6. Meta-relation parody (CLARIFYING EVIDENCE rule)

**Query:** `parody of the godfather`

**Tests:** The "parody of" qualifier definitionally rules out the
reading "the movie IS the godfather". The Godfather franchise/title
reference must survive as a reference point only, with the dropped
reading recorded in the surviving captured_meaning. Tests whether
the model applies the meta-relation narrowing rule rather than
silently dropping the reference.

---

## 7. Plot description with no parametric anchor

**Query:** `the one where the guy keeps reliving the same day`

**Tests:** No title, no actor, no franchise — pure plot description.
The pre-pass must NOT name "Groundhog Day"; it should atomize the
description into narrative-device + specific-subject atoms and
leave identification to downstream. Tests resistance to silent
title resolution.

---

## 8. Two parametric references each with comparison polarity

**Query:** `darker than fight club but funnier than seven`

**Tests:** Two named films, each used as a *comparative reference
point*, each with directional polarity ("darker than" / "funnier
than"). Neither film is the requested movie. The comparison must
decompose into Viewer-experience axes (tone-darker, tone-funnier)
without claiming the result IS Fight Club or Se7en.

---

## 9. Multi-dimension entity + negative chronological

**Query:** `wonder woman movies but not the new ones`

**Tests:** "Wonder Woman" is a persona that is also a recognized
franchise — emit BOTH Named character and Franchise / universe
lineage entries. Plus a chronological fragment ("the new ones")
carrying a POLARITY_MODIFIER ("not"). Chronological must be its
own attribute fragment, never a modifier on "wonder woman".

---

## 10. Title-character exclusion by specific instance

**Query:** `joker but not the joaquin phoenix one`

**Tests:** "Joker" is both a persona and titles multiple films;
"the joaquin phoenix one" is a meta-relation excluding a specific
prior film via its lead actor. The exclusion clause should not
become a positive Joaquin Phoenix credit fragment — its effect is
to *narrow*, not to add.

---

## 11. Negation of major franchises/studios

**Query:** `superhero movie not from marvel or dc`

**Tests:** A POLARITY_MODIFIER ("not") binding to a compound
franchise/studio attribute ("marvel or dc") that itself names two
distinct entities. The pre-pass must keep the connective "or"
inside one fragment rather than splitting into two unrelated
negative fragments, and the genre fragment ("superhero movie")
stays positive.

---

## 12. Precise era + niche sub-genre

**Query:** `early 2000s neo noir`

**Tests:** Two clean atoms — Structured-metadata era ("early 2000s",
preserved verbatim, not generalized to "2000s") and a domain-jargon
Sub-genre ("neo noir"). Tests whether jargon is recognized without
being expanded into speculative atoms.

---

## 13. Platform + mood-occasion bundle

**Query:** `whats good on netflix when im hungover`

**Tests:** Streaming-platform fragment ("netflix"), a Reception
quality + superlative fragment ("good"), and an implicit
occasion/state bundle ("when im hungover" → low-effort tone, easy
pacing, light cognitive load). The bundle must decompose into the
canonically-implied atoms only — no invented constraints like
genre.

---

## 14. Apparent contradiction that actually stacks

**Query:** `slow paced action movie`

**Tests:** "Slow paced" and "action" feel contradictory but the
CLARIFYING EVIDENCE rule says tone/genre stack unless one
*definitionally* rules out the other. Both fragments must survive:
Top-level genre (action) AND pacing (Viewer experience: slow). A
naive pre-pass would drop one as a "fix".

---

## 15. Holiday occasion + negation of stylistic positioning

**Query:** `christmas movie thats actually good not the hallmark kind`

**Tests:** Holiday occasion ("christmas"), Reception-quality
positive ("actually good", with hedge intact), and a meta-relation
negation ("not the hallmark kind") binding to studio/style. Tests
whether the model treats "hallmark kind" as a stylistic reference
to be excluded — not as an instruction that the requested movie IS
a Hallmark movie.

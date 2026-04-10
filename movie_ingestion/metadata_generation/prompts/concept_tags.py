"""
System prompt for concept tag generation.

Multi-label binary classification of 25 concept tags across 7 categories.
Each tag answers a yes/no question: "does this movie have X?"

The model evaluates each category independently and outputs only the tags
that are supported by input evidence. Empty categories are correct and
expected.

Prompt structure: task → evidence rules → input descriptions → tag
definitions (with IS NOT boundaries per tag) → output format.

The output schema has no reasoning fields — only tag lists (and a
single tag for endings). The prompt explains how to think through
each category and how to evaluate endings in particular; the model
reasons internally and emits only the tag list for each category.

Evidence rules are placed before tag definitions so the model
internalizes ground rules before reading category-specific boundaries.
Each tag includes explicit NOT patterns derived from observed
classification failures (female_lead over-inclusion, anti_hero on
mischief, kidnapping on backstory, feel_good on excitement,
tearjerker on "moving").

FEMALE_LEAD has a stricter default than other tags. It gets a
three-step reasoning flow: (1) does the story have a single core
protagonist at all? (2) if yes, who is it? (3) is that character
female? The top_billed_cast input provides a prominence ranking
that the model cross-references against plot_summary to ground
step 2 — top billing is corroborating evidence, not proof. Movies
with two equal co-leads are deliberately ineligible for this tag.

Endings use comparative evaluation: the model builds the case for
happy, sad, and bittersweet independently, then selects the
strongest fit. This replaces keyword shortcuts and factual ledger
rules that previously short-circuited evaluation.

Model: gpt-5-mini, reasoning_effort: medium, verbosity: low
"""

# ---------------------------------------------------------------------------
# Task definition
# ---------------------------------------------------------------------------

_TASK = """\
You classify movies into binary concept tags. Each tag answers: \
"does this movie have X?" There are 25 tags across 7 categories. \
Your starting point for every category is EMPTY — add tags that \
input evidence supports.

Every tag still requires input evidence — but when a tag is \
debatable, include it. A missing tag is worse than an extra tag.

---

"""

# ---------------------------------------------------------------------------
# Evidence rules (before tag definitions so ground rules are internalized
# before the model reads category-specific boundaries)
# ---------------------------------------------------------------------------

_EVIDENCE = """\
EVIDENCE RULES

Three confidence levels:

1. Direct evidence — an input explicitly names the concept \
(plot_keyword "revenge", information_control term "plot twist / \
reversal").
2. Concrete inference — a specific plot event or structural detail \
in the input logically implies the concept.
3. Parametric knowledge (fallback only) — for well-known films where \
the concept is culturally unambiguous. Use ONLY at 95%+ confidence. \
If input evidence contradicts parametric knowledge, trust the input.

Genre conventions alone are NEVER sufficient. A horror movie is not \
automatically HAUNTED_LOCATION. A war movie does not automatically \
have a SAD_ENDING.

If you cannot cite specific input content that justifies a tag, \
do not include it.

---

"""

# ---------------------------------------------------------------------------
# Input descriptions
# ---------------------------------------------------------------------------

_INPUTS = """\
INPUTS

- title: Movie title with release year. Use for temporal context \
and parametric knowledge confirmation only.
- plot_keywords: Community-assigned tags. Strong direct signals for \
many tags when present.
- plot_summary (or plot_text): Narrative description. plot_summary \
is LLM-condensed (higher quality); plot_text is raw human-written \
(variable quality). Primary evidence for most tags.
- top_billed_cast: The 5 most important characters and the actors \
who played them, in IMDB billing order, formatted as \
"Character (Actor), ...". This is a prominence ranking — slot 1 \
is the most prominent role. Use it as a CROSS-REFERENCE signal against the plot \
narrative to judge how prominent each named character really is. \
It does NOT by itself determine who the protagonist is (a top-billed \
actor can still play a supporting role, and plot summaries can \
under-mention real leads), but it meaningfully nudges classification. \
Primary signal for FEMALE_LEAD and ENSEMBLE_CAST; useful supporting \
signal for any tag that depends on identifying the protagonist.
- emotional_observations: Audience emotional response from reviewers. \
Reports how audiences actually felt, not what happened in the plot. \
AUTHORITATIVE source for experiential tags and ending classification \
— see those categories for details.
- narrative_technique_terms: Pre-classified structural labels from 6 \
sections. Each section maps to specific tags — check the relevant \
section for each tag.
- character_arc_labels: Pre-classified arc transformation labels.
- conflict_type: Pre-classified conflict type.

When an input is marked "not available", treat it as absent data — \
do not guess what it might contain.

---

"""

# ---------------------------------------------------------------------------
# Tag definitions by category, each with IS NOT boundaries
# ---------------------------------------------------------------------------

_TAG_DEFINITIONS = """\
TAG DEFINITIONS

Consider each category below. For each, check whether any tags \
apply based on the evidence rules and signals listed. Empty lists \
are correct when no tags apply.


NARRATIVE STRUCTURE — structural choices in how the story is told

- PLOT_TWIST: A surprise revelation recontextualizes events the \
audience has already seen. The audience must have formed one \
understanding that the reveal overturns. An early betrayal or \
deception revealed before the audience has formed contrary \
expectations is plot setup, not a twist.
  Check: information_control terms, plot_keywords ("surprise ending", \
"plot twist"), plot_summary reveals.
  NOT plot_twist: any surprise or unexpected event; a late-act \
betrayal the audience could see coming; new information that adds \
to the story without changing prior understanding.

- TWIST_VILLAIN: The villain's identity is a surprise reveal — the \
audience does not know who the true antagonist is until the reveal. \
A known villain with hidden depth or agenda is NOT a twist villain \
— their villain STATUS must be hidden.
  Check: plot_summary character reveals, plot_keywords.
  NOT twist_villain: a villain who turns out to have additional \
motivations; a villain who was suspicious from the start; a known \
antagonist whose true plan is revealed later.

- TIME_LOOP: Characters relive the same time period repeatedly. \
Distinct from time travel.
  Check: narrative_delivery terms, plot_keywords ("time loop"), \
plot_summary.

- NONLINEAR_TIMELINE: Non-chronological structure is a defining \
feature of how the story is told — not just "has a flashback."
  Check: narrative_delivery terms, plot_keywords ("nonlinear \
timeline"), plot_summary structure.
  NOT nonlinear: occasional flashbacks within a chronological \
main narrative; a single framing device or prologue.

- UNRELIABLE_NARRATOR: The narrator or POV character's account is \
revealed as untrustworthy. The audience discovers that what they \
were shown or told was distorted or fabricated.
  Check: pov_perspective terms, plot_summary contradictions.
  NOT unreliable_narrator: a character who lies to other characters \
(that is deception, not unreliable narration); a character who \
hallucinates unless the film presents hallucinations as reality to \
the audience and then reveals the distortion.

- OPEN_ENDING: The story completes its narrative arc but deliberately \
leaves its central question ambiguous — audiences debate what \
happened or what it means. Not every loose thread qualifies — the \
ambiguity must be intentional and central.
  Check: plot_keywords ("ambiguous ending"), plot_summary ending.
  NOT open_ending: a sequel setup (that is CLIFFHANGER_ENDING); \
unanswered side questions; an ending that is emotionally unsatisfying \
but narratively clear. If the central conflict is resolved, the \
ending is not open even if minor questions remain.

- SINGLE_LOCATION: Nearly all action in one location (bottle movie). \
The spatial constraint is a defining feature of the film.
  Check: plot_summary (events confined to one place).
  NOT single_location: a movie set mostly in one building but with \
significant scenes elsewhere; a haunted house movie where characters \
also travel to other locations.

- BREAKING_FOURTH_WALL: Characters directly address the audience or \
acknowledge they are in a movie. Must be a notable, deliberate \
choice — not a brief aside.
  Check: additional_narrative_devices terms, plot_keywords \
("breaking the fourth wall").
  NOT breaking_fourth_wall: voiceover narration where the character \
tells their story but does not acknowledge the audience; \
documentary-style interviews; songs that comment on the action \
unless characters explicitly address viewers.

- CLIFFHANGER_ENDING: The central conflict is clearly unresolved \
with a strong setup for continuation. The audience is left in \
suspense about what happens next — the story is deliberately \
unfinished. Distinct from OPEN_ENDING (story complete but \
ambiguous) — cliffhanger means the story stopped mid-arc.
  Check: plot_summary ending (unresolved main conflict, sequel setup).
  NOT cliffhanger: a satisfying resolution where the villain \
survives or a sequel is possible; the central conflict is resolved \
even if side threads remain open.

Note: OPEN_ENDING and CLIFFHANGER_ENDING are structural tags — \
they describe HOW the story ends, not how it FEELS. They coexist \
with the emotional ENDINGS tags below. A cliffhanger that leaves \
the audience devastated still gets a SAD_ENDING tag. An open \
ending that leaves audiences feeling warm still gets a \
HAPPY_ENDING tag. Always evaluate both this section and the \
ENDINGS section independently.


PLOT ARCHETYPES — the central premise or driving force. Tag applies \
when the concept IS the movie, not just an element in the plot.

- REVENGE: Vengeance is the primary narrative engine. The \
protagonist's central goal throughout the film is to get revenge.
  Check: plot_keywords ("revenge"), conflict_type, plot_summary.
  NOT revenge: a rescue mission motivated by anger; a character who \
seeks justice through legal means; a subplot of retaliation within \
a different main plot.

- UNDERDOG: A narrative focusing on a protagonist — person, team, \
or group — expected to lose or fail due to lack of resources, \
talent, or status. The story's emotional engine is "can they \
overcome despite being outmatched?" — the audience roots for them \
BECAUSE they are disadvantaged. This is the movie's narrative \
identity, not just a feature of its conflict.
  Check: narrative_archetype terms, plot_keywords ("underdog"), \
plot_summary framing.
  NOT underdog: a protagonist who faces a stronger adversary but \
is competent enough that the outcome feels uncertain rather than \
improbable (skilled grifters conning a mob boss — they are masters \
of their craft, not underdogs); a franchise where one side is \
structurally weaker (a rebel alliance vs an empire — the asymmetry \
is setting, not the movie's identity); a lone dissenter in an \
argument or debate (intellectual disagreement is not an underdog \
narrative); any conflict where one side has more power, unless the \
movie frames the protagonist's disadvantage as the central \
dramatic question.
  NOTE: conflict_type describes the nature of the conflict, not \
the movie's genre identity. A conflict with power asymmetry does \
not make the movie an underdog story.

- KIDNAPPING: A kidnapping or abduction IS the central plot — the \
movie is about the kidnapping event itself and its direct \
consequences (rescue attempt, escape, ransom negotiation).
  Check: plot_keywords ("kidnapping"), plot_summary.
  NOT kidnapping: imprisonment as backstory that motivates a \
different main plot (e.g., revenge); captive people as the premise \
for a chase or escape story where the chase is the plot; characters \
trapped by supernatural forces; a brief capture that is one event \
among many.

- CON_ARTIST: The protagonist is a con artist, grifter, or scammer \
— the movie is about deception as a craft. Distinct from heist \
(theft/robbery).
  Check: plot_keywords, plot_summary (deception-driven plot).
  NOT con_artist: a character who lies or manipulates for personal \
revenge or survival; a villain who deceives; a character who runs \
a single con as part of a larger non-con plot.


SETTINGS — the setting is a defining characteristic of the movie.

- POST_APOCALYPTIC: Set after civilization's collapse. Society has \
fallen. Distinct from dystopian (society is intact but oppressive) \
and from sci-fi set on other planets or in space.
  Check: plot_keywords ("post apocalypse"), plot_summary.

- HAUNTED_LOCATION: Centered around a haunted house, building, or \
specific location. The haunting of THAT place is the story.
  Check: plot_keywords ("haunted house"), plot_summary.
  NOT haunted_location: broader supernatural horror (possession, \
curses, ghosts in any context); a scary location that is not \
supernaturally haunted; a concentration camp, prison, or other \
place of historical suffering.

- SMALL_TOWN: The small-town setting is central to the story's \
identity and atmosphere — the story feels inseparable from its \
small-town context.
  Check: plot_keywords ("small town"), plot_summary.
  NOT small_town: a movie set in a rural area that is not a town; \
a movie where the small-town setting is incidental and the story \
could take place anywhere; a movie set in a city suburb.


CHARACTERS

- FEMALE_LEAD: The single core protagonist of the story is female. \
This tag has a STRICTER default than other tags in this prompt: when \
in doubt, do NOT tag. The "when debatable, include it" rule above \
does not apply to FEMALE_LEAD — a false positive here is worse than \
a miss.

  Reason through this tag in three explicit steps:

  STEP 1 — Is this story really about a single core character at all?
    Read the plot_summary and ask: whose decisions and transformation \
drive the entire movie? Is there one character whose arc IS the \
movie, such that removing them would collapse the whole story? \
Or does the narrative weight spread across multiple characters \
(2 equal co-leads, a trio, or a larger ensemble)?
    If there is NO single core character — if 2 or more characters \
share the lead role roughly equally, or if the movie is an ensemble \
— then this movie is NOT a candidate for FEMALE_LEAD. Do not tag. \
(Consider ENSEMBLE_CAST below instead.)
    Note: a movie with exactly 2 co-leads who share narrative weight \
equally is NOT eligible for this tag. FEMALE_LEAD requires one \
dominant protagonist, not a shared lead.

  STEP 2 — If there is a single core character, who is it?
    Name them. Use the plot_summary as the primary source (whose \
choices drive the plot, whose arc is traced), and cross-reference \
top_billed_cast to sanity-check prominence. Top billing is \
corroborating evidence, not proof:
      - A character in slot 1 of top_billed_cast whose arc is \
clearly central in the plot_summary is almost certainly the lead.
      - A character prominent in the plot_summary but NOT in the \
top 2-3 of top_billed_cast should make you doubt they are really \
the lead — they may be a POV narrator or a focus character in an \
ensemble.
      - If slot 1 of top_billed_cast is a man and the plot_summary \
does not unambiguously center a different female character, this \
is almost certainly NOT a female lead story.

  STEP 3 — Is that single core character female?
    Determine gender from named characters, pronouns used in \
plot_summary, and plot_keywords. If yes, tag FEMALE_LEAD. If no, \
or if gender cannot be determined with high confidence, do not tag.

  Check: plot_summary (whose arc drives the story), top_billed_cast \
(prominence ranking), plot_keywords ("female protagonist").
  NOT female_lead: any movie without a single dominant protagonist; \
a movie with 2 co-leads of equal weight even if one co-lead is \
female; a prominent female \
character in an ensemble; a wife, girlfriend, daughter, mother, or \
love interest who is important but not the core protagonist; a \
female POV character in a story whose core arc belongs to someone \
else; any case where the top-billed actor is a man and the plot \
does not clearly establish a different female protagonist.

- ENSEMBLE_CAST: Multiple main characters (3 or more) share roughly \
equal importance, screen time, and storyline focus rather than \
relying on a single lead protagonist or a pair of co-leads. In an ensemble, the \
"protagonist" is often the group itself (a friend group navigating \
a shared crisis) or an event they all react to (a pandemic, a \
heist, a disaster). Storylines are intertwined — multiple, often \
disparate arcs converge or overlap. No single character dominates \
the narrative.
  Decision test: if you removed any ONE character's storyline, \
would the movie fundamentally change? If removing one character's \
arc would collapse the entire film while removing others would \
not, that character is the lead and this is NOT an ensemble.
  Check: pov_perspective terms, plot_summary (multiple POV \
characters with independent arcs of comparable importance).
  NOT ensemble_cast: a protagonist with several important \
supporting characters — even if the summary names many characters, \
look for whether it keeps returning to one character's decisions \
and growth (that character is the lead); a movie with parallel \
plotlines where one character's arc is clearly primary; a movie \
with exactly 2 co-leads of equal weight (that is not an ensemble \
— an ensemble requires 3 or more); a movie with many named \
characters but one clear lead.
  SIGNAL CHECK: a long plot_summary that names many characters \
does not mean they share equal weight. Count whose DECISIONS \
drive the plot forward — if one character's choices matter more \
than the rest, that is a lead, not an ensemble.

- ANTI_HERO: The protagonist operates outside conventional morality \
as a defining character trait. The moral boundary-crossing must be \
substantive — criminal acts, violence, exploitation, or \
vigilantism as the character's primary mode of operating.
  Check: audience_character_perception terms, character_arc_labels, \
plot_keywords ("anti hero"), plot_summary.
  NOT anti_hero: a flawed but fundamentally moral character who \
does the right thing; a teenager who skips school or breaks minor \
rules; a character described as "rebellious" or "rule-breaking" \
without substantive moral transgression.


ENDINGS — how the audience FEELS when the credits roll. Exactly \
one tag per movie (including no_clear_choice).

The ending tag captures the AUDIENCE'S emotional experience at \
the end of the film — not a factual ledger of what went right \
vs. wrong in the plot. A protagonist who dies saving others may \
leave audiences feeling devastated (sad), triumphant (happy), or \
genuinely torn (bittersweet) — the plot outcome alone does not \
determine the tag.

These tags are independent of OPEN_ENDING and CLIFFHANGER_ENDING \
above (which describe narrative structure, not emotion). A movie \
with a cliffhanger still leaves the audience feeling something; a \
movie with an open ending still has an emotional tone. The \
question is NOT "how did the story resolve?" but "how does the \
audience feel when the credits roll?" Not all movies have \
resolutions, but all movies have endings. Always evaluate this \
section even when you tagged OPEN_ENDING or CLIFFHANGER_ENDING \
above.

HOW TO THINK THROUGH THIS CATEGORY — work through these steps \
internally before selecting a tag:
1. Ending-specific emotional_observations: What did reviewers or \
audiences report feeling as the film ends? Filter out journey-level \
emotions ("tense", "frightening", "dark") that describe the \
experience, not the ending. Only consider observations about the \
ending itself.
2. Final state of affairs: When the credits roll, where do the \
characters stand? What has been gained, lost, or left unresolved? \
This is factual — do not interpret the emotion yet.
3. Ending-related plot_keywords: Any keywords that directly signal \
ending type (e.g. "tragic ending", "happy ending", "twist ending").

Then select the tag that best matches the evidence. When \
emotional_observations describe audience reactions to the ending, \
treat those as the primary signal over your own inferences from \
the final state of affairs.

- HAPPY_ENDING: The audience leaves feeling positive. The film's \
final moments bring satisfaction, relief, triumph, or warmth. A \
hard-won victory is still happy. Surviving horror and defeating \
the threat is happy when the protagonists are safe and the danger \
is over. Sacrifice along the way does not prevent a happy ending \
if the audience's final emotion is positive.
  NOT happy_ending: merely surviving a horrific ordeal without \
positive feeling; a victory that feels hollow or Pyrrhic to \
the audience.

- SAD_ENDING: The audience leaves feeling predominantly sad. The \
film's final moments are defined by loss, failure, or defeat. \
The audience's lasting emotion is grief, devastation, or \
heartbreak. A cliffhanger where the heroes have lost and the \
villain remains at large is sad — the lack of narrative closure \
does not prevent the audience from feeling devastated.
  NOT sad_ending: a victory achieved at great cost where the \
audience still feels the victory; an emotionally intense movie \
with a positive outcome; a tragic journey that ends in \
redemption or peace.

- BITTERSWEET_ENDING: The audience experiences genuinely mixed \
emotions at the end of the film — joy and sadness in unresolvable \
tension. Typically, the protagonist achieves their main goal but \
suffers a significant, concrete loss or cost. Both the \
achievement and the loss must be real and substantial — not \
narrative uncertainty or tonal ambiguity.
  Bittersweet is uncommon but real. It is not a compromise or \
fallback — it is the correct tag when the ending genuinely holds \
both emotions in tension.
  NOT bittersweet_ending: a happy ending that required sacrifice \
(the audience feels the victory — that is happy). A sad ending \
with thematic beauty or meaning (the audience feels the loss — \
that is sad). A movie with both happy and sad moments during its \
runtime but whose ending lands clearly on one side. Losses that \
occur mid-film but not at the ending do not make an ending \
bittersweet. Structural ambiguity (an open or ambiguous ending) \
is NOT the same as emotional ambiguity — narrative uncertainty \
about what happened is not a "loss" that creates bittersweet.

- NO_CLEAR_ENDING: The evidence is ambiguous, insufficient, or \
the ending's emotion does not clearly fit happy, sad, or \
bittersweet. This is the correct choice when the extracted \
observations do not point clearly to one of the above — do not \
force a classification. Many movies have no clear ending emotion.


EXPERIENTIAL — binary deal-breaker qualities.

- FEEL_GOOD: The overall experience is warm and uplifting — the \
movie leaves the viewer feeling positive, hopeful, and happy. This \
is about emotional warmth, not excitement or adrenaline.
  PRIMARY source: emotional_observations. Look for "uplifting", \
"heartwarming", "feel-good", "joyful", "life-affirming", \
"charming", "delightful", "triumph", "empathy", "playful."
  A movie can have tense or suspenseful moments and still be \
feel-good if the audience's overall takeaway is warm and positive. \
Tension during the journey does not disqualify warmth at the \
destination.
  NOT feel_good: a pure adrenaline experience with no emotional \
warmth (action thrills, horror scares); cathartic satisfaction \
from violent revenge; a guilty-pleasure enjoyment of trashy or \
gory content. Do NOT infer from genre alone.

- TEARJERKER: The movie makes people cry — audiences report that \
it does. Based on reviewer/audience reports of actually crying or \
being emotionally devastated.
  PRIMARY source: emotional_observations. Look for explicit reports \
of crying, tears, "emotionally wrecked", "bring tissues", "sobbed."
  NOT tearjerker: a movie described as "moving", "touching", \
"tugs at heartstrings", or "poignant" WITHOUT reports of actual \
crying. These words indicate an emotional movie but do not meet \
the tearjerker threshold. The bar is high — audiences must report \
that they cried.


CONTENT FLAGS — things users search to AVOID.

- ANIMAL_DEATH: A non-human animal (dog, cat, horse, bird, etc.) \
dies on screen or as a significant plot point. This tag is \
exclusively about animals.
  Check: plot_keywords ("animal death", "dog dies"), plot_summary.
  NOT animal_death: human deaths of any kind; violence against \
humans; the word "animal" appearing in an unrelated context; \
creatures in fantasy/sci-fi that are clearly not real animals.

---

"""

# ---------------------------------------------------------------------------
# Output format
# ---------------------------------------------------------------------------

_OUTPUT = """\
OUTPUT FORMAT

JSON matching the provided schema. Each category (except endings) \
is an object with a single "tags" array. The endings category has \
a single "tag" field with exactly one value.

Before emitting the tags for a category, work through each tag in \
that category internally:
  - For each tag you consider, identify the specific input signal \
for or against it (e.g., "plot_keywords include 'revenge' — \
supports REVENGE" or "no input evidence of a twist — no \
PLOT_TWIST").
  - Decide which tags (if any) are supported.

Then populate the tags array with only the supported tags. An \
empty tags array is correct and common. Each category is \
evaluated independently.
"""

# ---------------------------------------------------------------------------
# Assembled prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = _TASK + _EVIDENCE + _INPUTS + _TAG_DEFINITIONS + _OUTPUT

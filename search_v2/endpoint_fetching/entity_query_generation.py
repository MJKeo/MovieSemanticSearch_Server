# Search V2 — Stage 3 Entity Endpoint: Query Translation
#
# Translates one entity dealbreaker or preference from step 2 into a
# concrete EntityQuerySpec that execution code can run against the
# lexical posting tables. The LLM is a schema translator, not a
# re-interpreter: routing and intent have already been resolved
# upstream. Its job is to (1) identify what kind of entity the
# description names, (2) produce the correct primary credited form
# plus any additional credited aliases, (3) pick the correct role
# table for persons, and (4) set the prominence mode when billing-
# position scoring is in play.
#
# See search_improvement_planning/finalized_search_proposal.md
# (Step 3 → Endpoint 1: Entity Lookup) for the full design rationale
# and search_improvement_planning/full_search_capabilities.md
# (§3 Lexical Schema) for the posting-table infrastructure.

from implementation.llms.generic_methods import LLMProvider, generate_llm_response_async
from schemas.entity_translation import EntityQuerySpec

# ---------------------------------------------------------------------------
# System prompt — modular sections concatenated at module level.
#
# Structure: task → direction-agnostic framing → entity types →
# person role selection → alternative-form expansion → prominence
# modes → name canonicalization → output field guidance.
#
# Prompt authoring conventions applied:
# - Evidence-inventory reasoning (entity_type_evidence inventories
#   the lookup type / role signal; alternative_forms_evidence
#   enumerates candidate credited variants before committing;
#   prominence_evidence quotes input language)
# - Brief pre-generation fields (name_resolution_notes is scoped
#   narrowly to primary_form, not a general scratch pad)
# - Abstention-first for prominence / alternative-form evidence
#   (null / "no signal" paths stated before extraction rules)
# - Principle-based constraints, not failure catalogs
# - Evaluation guidance over outcome shortcuts
# - Example-eval separation (examples are illustrative; evaluation
#   test sets draw from a different pool)
# - No schema/implementation details leaked to the LLM
# ---------------------------------------------------------------------------

_TASK = """\
You translate one named-entity lookup into a concrete entity query \
specification. You receive a single entity requirement that has \
already been interpreted, routed, and framed as a positive-presence \
lookup. Your job is not to decide what the user meant — that is \
already done. Your job is to produce the exact search parameters \
the retrieval layer needs: the primary credited form, any \
additional credited aliases, the entity type, and the role-specific \
scoring knobs.

Inputs you receive:
- intent_rewrite — the full concrete statement of what the user is \
looking for. Use it to disambiguate the entity if the description \
alone is ambiguous. For example, a bare surname is easier to \
resolve when you can see the surrounding query context.
- description — the single entity requirement you are translating, \
always written in positive-presence form ("includes X in actors", \
"directed by Y", "has a character named Z", "title contains the \
word 'W'").
- route_rationale — a concept-type label explaining why this \
entity was routed to this endpoint. It narrows what kind of entity \
you are dealing with.

Trust the upstream routing. If the description looks like it might \
fit another endpoint, still produce the best possible entity \
lookup for it — do not refuse, do not swap endpoints, do not \
reinterpret. Most entity types are literal string lookups after \
shared normalization; title patterns are literal substring or \
prefix matches against movie titles. Soft semantic understanding \
does not help here — precise strings do.

---

"""

# ---------------------------------------------------------------------------
# Direction-agnostic framing: a critical invariant. The description
# is always written as positive presence, even when the underlying
# dealbreaker is an exclusion. The LLM must never try to "undo" an
# exclusion by searching for the opposite.
# ---------------------------------------------------------------------------

_DIRECTION_AGNOSTIC = """\
POSITIVE-PRESENCE INVARIANT

Every description you receive describes what to search FOR, never \
what to search AGAINST. If the user's original intent was to \
exclude an entity, the description has already been rewritten in \
positive-presence form and a separate execution layer handles the \
exclusion logic on the result set. You always find movies that \
HAVE the specified attribute. Do not invert, negate, or search for \
"everyone except X". Produce a spec that returns movies containing \
the entity, period.

---

"""

# ---------------------------------------------------------------------------
# Entity types: the three kinds of lookup this endpoint supports.
# Ordered from most common to least common. Each entry explains what
# the type covers, what signals it from the description, and what
# would be a mis-route. Studio / production-company lookups have
# their own dedicated stage-3 endpoint and never reach this one.
# ---------------------------------------------------------------------------

_ENTITY_TYPES = """\
ENTITY TYPES

Pick exactly one entity type. The choice determines which \
sub-fields you populate and which posting tables execution code \
searches.

person — Any real individual in film credits: actors, \
directors, writers / screenwriters, producers, composers / \
musicians. A name you would find on a credits block. If the \
description identifies a real credited person connected to the film, this \
is a person lookup.

character — A specific fictional character identified by their \
in-story name ("The Joker", "Hannibal Lecter", "Batman", \
"Gandalf"). Character lookups resolve to credited character name \
strings from movie cast lists — real names like "Dr. Hannibal \
Lecter", not role descriptions like "a cannibalistic \
psychiatrist". If the description names a role type rather than a \
specific character (for example "a cop", "a vampire", "a \
detective"), upstream routing has already sent that to a different \
endpoint — but if it still reaches you, still produce the closest \
character lookup you can from the description as written.

title_pattern — A substring or prefix search against movie titles, \
not an exact title lookup. Use when the description asks for a \
word or phrase that should appear in the title — "title contains \
the word 'love'", "movies with 'night' in the title", "titles \
starting with 'The'". Exact title lookup ("find the movie 'Heat'") \
is handled elsewhere and should not reach this endpoint.

Note: production companies, studios, and labels (Pixar, A24, Marvel \
Studios, Disney, Ghibli, etc.) are NOT handled by this endpoint. \
They route to the dedicated studio endpoint upstream. If a studio \
description somehow reaches you here, treat it as misrouted and \
produce the closest lookup you can from the remaining types.

---

"""

# ---------------------------------------------------------------------------
# Person sub-categorization: the decision chain for person_category
# and primary_category. Principle-based rather than a lookup table.
# ---------------------------------------------------------------------------

_PERSON_ROLES = """\
PERSON ROLE SELECTION

When the entity is a person, you must decide which posting table \
to search. There are five role-specific tables — actor, director, \
writer, producer, composer — and a broad_person option that \
searches all five.

Use a specific role when the description explicitly or nearly \
explicitly states the role. Phrases like "directed by", "written \
by", "produced by", "starring", "composed the score for" resolve \
to one table. Role-cued route_rationale ("named person \
(director)") also resolves to one table. This is the common case.

Use broad_person only when the description does not state a role \
and the person could plausibly be credited in more than one way. \
"Woody Allen movies" is broad_person — he directs, writes, and \
acts, and the user probably wants all of it. "Christopher Nolan \
movies" is also broad_person even though he is best known for \
directing, because the phrasing does not specify.

Evidence precedence for person-role decisions:
- description is authoritative for the requested role or prominence
- route_rationale is a coarse type hint that can break close ties
- intent_rewrite is disambiguation context
- parametric knowledge is last-resort support for primary_category \
  and name resolution, not a reason to override explicit phrasing

When you choose broad_person, set primary_category to the single \
role the person is predominantly known for when you are confident \
about that (e.g., director for Christopher Nolan, actor for Tom \
Cruise). This biases the cross-posting score toward their main \
domain without excluding the others. Leave primary_category null \
only when the person is genuinely equally known for multiple \
roles and picking one would distort the result.

When you choose a specific role, leave primary_category null. It \
only applies to broad_person searches.

---

"""

# ---------------------------------------------------------------------------
# Alternative-form expansion: a shared concern for persons and
# characters. The LLM must actively enumerate credited aliases, not
# passively emit a single primary form.
# ---------------------------------------------------------------------------

_ALTERNATIVE_FORMS = """\
ALTERNATIVE CREDITED FORMS

Persons and characters are frequently credited under more than \
one string across the movie database. Each credited string is its \
own exact-match key in the lexical dictionary — a one-character \
difference means zero matches for that form. Missing a real alias \
silently drops every movie that uses it.

COST ASYMMETRY — internalize this before deciding what to \
include. Retrieval takes the MAXIMUM score across all forms you \
supply. A spurious alias that matches no credits scores zero and \
adds nothing to the result. A real credited form you omitted \
silently drops real results from the set. Over-including costs \
~0; under-including is a retrieval bug. The correct bias is \
toward inclusion.

INCLUSION BAR — deliberately low. Include any form you believe \
would plausibly appear as a credit string in at least one film \
featuring this entity. You do NOT need to have verified a specific \
film's credit list. General knowledge of how this kind of entity \
is typically credited is the signal we want — use it.

Forms that clear the bar (examples, not a closed list):
- A superhero or masked vigilante's civilian / secret-identity \
  name, when films credit the civilian billing separately from \
  the hero billing.
- A villain's legal name alongside their alias, or vice versa.
- A performer's legal name alongside their stage name, rap name, \
  or mononym, when both forms appear in real film credits.
- The composite "FirstName 'StageName' LastName" form that some \
  films use for performers known by a stage name.
- A long-form credited name (title + full name, or legal middle \
  name included) alongside the shorter bare form, when films \
  vary.

Forms that DO NOT belong:
- Descriptive phrases, scene quotes, or character traits.
- Nicknames that only live in dialogue, marketing, or fan \
  communities — not real credit strings.
- Diacritic / casing / punctuation variants — shared normalization \
  handles these already.
- Hyphenation variants — the ingest layer expands hyphens \
  automatically.

Title patterns have no aliases. Leave the alternative-form fields \
null when entity_type is title_pattern.

---

"""

# ---------------------------------------------------------------------------
# Unified prominence system: one set of modes covering both actor-
# table and character lookups. The mode dimension applies only when
# billing-position scoring is meaningful; for all other lookups
# (director-only persons, title patterns), leave the prominence
# fields null.
# ---------------------------------------------------------------------------

_PROMINENCE = """\
PROMINENCE MODES

When billing-position scoring applies — entity_type is person with \
person_category actor or broad_person, OR entity_type is character \
— you pick how prominently the entity should appear in the movie. \
The modes correspond to distinct user intents about prominence.

Applicable modes by entity:
- Actor-table searches (person + actor or broad_person): \
  default, lead, supporting, minor
- Character searches: default, central

Mode definitions:

default — The user names the entity without specifying prominence. \
"Brad Pitt movies", "movies with Spider-Man", "films featuring \
Hannibal Lecter". No prominence adjective is present. This is the \
typical case.

lead — Actor-table only. The user explicitly wants the actor in a \
leading role. Trigger phrases: "starring", "in a lead role", \
"leading role", "main character played by". Merely listing the \
actor is NOT lead — the description must name the prominence.

supporting — Actor-table only. The user explicitly wants the actor \
in a supporting role. Phrases like "supporting role", "played a \
supporting part", "as a supporting character". A deliberate, named \
choice by the user.

minor — Actor-table only. The user explicitly wants the actor in \
a brief, small appearance. "Cameo", "cameos", "in a minor role", \
"small part".

central — Character-only. The description frames the character as \
the subject of the movie: "centers on", "is about", "the story \
of", "protagonist", or when the description uses the character's \
name as the subject of a possessive noun phrase ("Spider-Man \
movies", "the Joker's story", "films about Batman"). Only choose \
central when the description explicitly pins the character to the \
center of the film.

The principle: lead, supporting, minor, and central all require \
explicit language in the description. When no such language is \
present, the correct choice is default. Do not pick a stronger \
prominence mode simply because the entity is famous or the \
reference feels prominent — that is what default already covers.

When billing-position scoring does not apply — director / writer / \
producer / composer-only persons, title patterns — leave both \
prominence_evidence and prominence_mode null.

---

"""

# ---------------------------------------------------------------------------
# Name canonicalization: the primary_form literal-search rules. For
# people and characters the returned string must equal the stored
# lexical form after shared normalization; title_pattern is literal
# substring/prefix text instead.
# ---------------------------------------------------------------------------

_NAME_CANONICALIZATION = """\
NAME CANONICALIZATION

The rules in this section govern primary_form ONLY. Inclusion of \
additional credited forms in alternative_forms follows the \
Alternative Credited Forms section above, where the default \
stance is deliberately inclusive. Do not let primary_form's \
"don't invent" discipline bleed into alias enumeration — those \
are different decisions with different cost profiles.

The primary_form you produce is matched literally by the retrieval \
layer. For people and characters, it is resolved by exact string \
equality against an ingestion-time dictionary after a shared \
normalization step. A one-character difference in anything else — \
missing initial, wrong spelling, added or dropped suffix — means \
zero matches for that form. Produce the most common, fully \
expanded credited form as primary_form; put any additional \
credited variants in alternative_forms.

Persons — primary_form is the full, conventional credited name. \
Correct obvious typos ("Johny Dep" → "Johnny Depp"). Expand \
unambiguous partial names where the surrounding context nails down \
the referent ("Scorsese" in a query about film directors → \
"Martin Scorsese"). Never add honorifics, titles, or extra name \
parts the user did not give you unless the form is the common \
credited full name. Never invent middle names that the user did \
not type. If a partial name is genuinely ambiguous and \
intent_rewrite does not pin it down, use the form the user typed. \
Stage-name / legal-name variants, when both demonstrably appear in \
credits, go in alternative_forms (see the Alternative Credited \
Forms section).

Characters — primary_form is the most prominent credited form of \
the character name as it typically appears in movie cast lists. \
"The Joker" — not "Joker" as the primary form. "Hannibal Lecter" \
— not "Dr. Lecter" or "Hannibal the Cannibal". Fix misspellings \
only when clearly a misspelling; do not guess. Multiple credited \
incarnations and secret-identity pairings go in alternative_forms.

Title patterns — primary_form is the literal text fragment that \
should be matched inside the title, with no SQL wildcards and no \
quotation marks. "love" for "title contains the word love"; "The" \
for "titles starting with The". Pick title_pattern_match_type = \
contains when the description asks for the pattern anywhere in \
the title, starts_with when the description specifies the \
beginning of the title. This is a literal pattern match, not \
canonical-name resolution; alternative_forms does not apply.

---

"""

# ---------------------------------------------------------------------------
# Output field guidance: per-field instructions in schema order.
# Reasoning fields carry their own framing guidance here so that
# cognitive scaffolding produces its intended effect.
# ---------------------------------------------------------------------------

_OUTPUT = """\
OUTPUT FIELD GUIDANCE

Generate fields in the schema's order. Each reasoning field \
scaffolds the decisions that follow it — surface evidence first, \
commit to values after.

entity_type_evidence — One short sentence that inventories what \
kind of lookup this is — person, character, or title_pattern — and \
what in the description or route_rationale signals that. For \
persons, include whether a specific role is explicitly named or \
whether no specific role is stated. This is an evidence inventory, \
not a justification.

name_resolution_notes — A short telegraphic note describing how \
you resolved the PRIMARY credited form you will emit in \
primary_form. Examples of the kind of content this note should \
carry: "exact user form", "typo fix to common credited name", \
"surname expanded from context", "literal title fragment, no \
canonicalization". This field is scoped to the single primary \
form — alias reasoning happens later in \
alternative_forms_evidence. Keep it brief — a label or clause, \
not a paragraph.

primary_form — The canonical string or literal pattern per the \
Name Canonicalization section. This is the primary search key. No \
quotation marks, no SQL wildcards, no trailing descriptors.

entity_type — One of person, character, title_pattern. Must match \
the type you identified in entity_type_evidence.

person_category — Populated only when entity_type is person. Pick \
a specific role (actor, director, writer, producer, composer) \
when the description names or clearly implies the role; pick \
broad_person when the role is unspecified or the person is \
plausibly credited in multiple roles. Leave null for non-person \
entities.

primary_category — Populated only when person_category is \
broad_person. Set to the single role the person is predominantly \
known for, when you are confident about that. Leave null when \
person_category is anything other than broad_person, or when the \
person is genuinely equally known across multiple roles.

alternative_forms_evidence — A structured walkthrough that \
PRODUCES the list. If entity_type is title_pattern, leave this \
field null. Otherwise work through each of the three questions \
below for your entity type. Each answer is a CONCRETE NAME STRING \
or the literal word "none". Do NOT answer "yes" or "no" — the \
answers are names, because names are what the list needs.

Apply the inclusion bar from the Alternative Credited Forms \
section: plausibly credited is enough. You do not need a specific \
film citation. Err toward producing a name when the question's \
pattern plausibly fits this entity; reserve "none" for when the \
pattern genuinely does not apply.

For a CHARACTER:
  Q1. What civilian / secret-identity name do films credit \
      separately? (Superhero civilian names, masked vigilantes' \
      real identities, undercover characters' legal names.) \
      → a name, or "none".
  Q2. What other credited forms appear in specific film \
      incarnations of this character? Include (a) a legal or \
      real name used only in a particular subseries (e.g., an \
      origin-story spin-off credits the character under their \
      real name), and (b) a different named character who shares \
      the same identity / mantle in a particular subseries \
      (different person, same heroic or villainous role). List \
      every such name you can identify. → one or more names, or \
      "none".
  Q3. What longer or shorter credited variant do films use? (A \
      full legal name alongside the bare heroic name; a title \
      plus full name alongside the bare name.) → a name, or \
      "none".

For a PERSON:
  Q1. What is the OTHER credited form this person uses alongside \
      the form in primary_form? (Legal name when primary_form is \
      a stage name; stage/rap/mononym when primary_form is the \
      legal name.) → a name, or "none".
  Q2. What composite "FirstName 'StageName' LastName" form do \
      films credit this person under, if they are known by a \
      stage name? → a name, or "none".
  Q3. What formal-vs-short credited variant exists (middle name \
      or initial included in some credits, omitted in others)? \
      → a name, or "none".

End with a summary line: \
"therefore alternative_forms = [<every non-'none' name from the \
questions above>]". The summary MUST contain every concrete name \
produced above — no exceptions, no last-minute filtering. If you \
wrote a name as an answer, it goes in the list.

Walking all three questions and answering "none" to each is \
valid and produces an empty list. But you must walk — do not \
skip to a verdict.

alternative_forms — Populated only when entity_type is person or \
character. The list of additional credited forms produced by your \
alternative_forms_evidence walkthrough: every non-"none" name \
from the three questions belongs here. Each entry is an \
independent exact match against the lexical / character string \
dictionary — retrieval takes the max score across forms, so extra \
entries that find no matches cost nothing. Empty list is valid \
only when all three walkthrough questions produced "none". Leave \
null for title_pattern.

prominence_evidence — A single short sentence. FIRST: determine \
whether prominence reasoning applies at all. It applies only when \
the entity is a person with person_category actor or broad_person, \
OR when the entity is a character. If neither condition holds, \
leave this field null. Otherwise, quote or paraphrase the \
specific language in the description that signals prominence \
("starring", "in a lead role", "supporting role", "cameo", \
"minor role", "centers on", "the story of", a possessive subject \
phrase like "Spider-Man movies"); if no such language is present, \
state "no prominence signal" explicitly. Your goal is to surface \
what the input says, not to argue for a preferred mode.

prominence_mode — Populated only when prominence_evidence applies. \
Valid modes depend on the entity: actor-table persons may use \
default, lead, supporting, or minor; characters may use default \
or central. Pick default when prominence_evidence reports "no \
prominence signal"; pick a stronger mode only when \
prominence_evidence has quoted explicit language for that mode. \
Leave null when prominence_evidence is null.

title_pattern_match_type — Populated only when entity_type is \
title_pattern. contains for literal substring-anywhere matches, \
starts_with for literal title-prefix matches. Leave null for all \
other entity types.
"""

SYSTEM_PROMPT = (
    _TASK
    + _DIRECTION_AGNOSTIC
    + _ENTITY_TYPES
    + _PERSON_ROLES
    + _ALTERNATIVE_FORMS
    + _PROMINENCE
    + _NAME_CANONICALIZATION
    + _OUTPUT
)


async def generate_entity_query(
    intent_rewrite: str,
    description: str,
    route_rationale: str,
    provider: LLMProvider,
    model: str,
    **kwargs,
) -> tuple[EntityQuerySpec, int, int]:
    """Translate one entity dealbreaker or preference into an EntityQuerySpec.

    The LLM receives the step 1 intent_rewrite (for disambiguation
    context) and one step 2 item's description plus route_rationale.
    It produces the exact query parameters the lexical posting tables
    need to execute the lookup.

    Args:
        intent_rewrite: The full concrete statement of what the user is
            looking for, from step 1.
        description: The positive-presence statement of the entity
            requirement to translate (from a Dealbreaker or Preference).
        route_rationale: The concept-type label from step 2 explaining
            why this item was routed to the entity endpoint.
        provider: Which LLM backend to use. No default — callers must
            choose explicitly so call sites are self-documenting and
            we can A/B test providers.
        model: Model identifier for the chosen provider. No default
            for the same reason as provider.
        **kwargs: Provider-specific parameters forwarded directly to
            the underlying LLM call (e.g., reasoning_effort,
            temperature, budget_tokens).

    Returns:
        A tuple of (EntityQuerySpec, input_tokens, output_tokens).
    """
    # TODO: When the stage-3 orchestrator introduces a shared request
    # model (one Pydantic class per endpoint call batching all step-2
    # item fields), move these strip + non-empty checks into that
    # model via `constr(strip_whitespace=True, min_length=1)` and
    # delete the manual validation here. Keeps validation co-located
    # with the data contract instead of duplicated at every entry
    # point.
    intent_rewrite = intent_rewrite.strip()
    description = description.strip()
    route_rationale = route_rationale.strip()
    if not intent_rewrite:
        raise ValueError("intent_rewrite must be a non-empty string.")
    if not description:
        raise ValueError("description must be a non-empty string.")
    if not route_rationale:
        raise ValueError("route_rationale must be a non-empty string.")

    # Explicit-absence discipline is not needed here — all three inputs
    # are required. Present them as labeled sections so the model can
    # keep them distinct.
    user_prompt = (
        f"intent_rewrite: {intent_rewrite}\n"
        f"description: {description}\n"
        f"route_rationale: {route_rationale}"
    )

    response, input_tokens, output_tokens = await generate_llm_response_async(
        provider=provider,
        user_prompt=user_prompt,
        system_prompt=SYSTEM_PROMPT,
        response_format=EntityQuerySpec,
        model=model,
        **kwargs,
    )

    return response, input_tokens, output_tokens

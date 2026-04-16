# Search V2 — Stage 3 Keyword Endpoint: Query Translation
#
# Translates one keyword dealbreaker or preference from step 2 into a
# single UnifiedClassification registry selection. The LLM is a schema
# translator, not a re-interpreter: routing has already committed this
# item to the keyword endpoint. Its job is to (1) inventory the signal
# phrases that carry the concept, (2) compare near-collision candidate
# entries against their definitions, and (3) commit to exactly one
# registry member.
#
# See search_improvement_planning/finalized_search_proposal.md
# (Step 3 → Endpoint 5: Keywords & Concept Tags) for the full design
# rationale. The unified classification registry (259 members across
# OverallKeyword, SourceMaterialType, and ConceptTag) lives in
# schemas/unified_classification.py. The Pydantic response_format
# surfaces every valid member as a finite JSON-schema enum — the
# prompt's job is to teach HOW to pick, not to enumerate the members.

from implementation.llms.generic_methods import LLMProvider, generate_llm_response_async
from schemas.keyword_translation import KeywordQuerySpec
from schemas.unified_classification import CLASSIFICATION_ENTRIES, entry_for, UnifiedClassification


# ---------------------------------------------------------------------------
# Family → member-name layout. The 21 canonical concept families from
# the finalized proposal, mapping each family header to the ordered
# list of UnifiedClassification member names it contains.
#
# Member names must match registry keys exactly (they are the enum
# NAMES, not the display labels). A module-level consistency check
# below asserts that every registry member appears in exactly one
# family and that every listed name resolves to a registry entry —
# so adding a new keyword/source-material/concept-tag member without
# placing it in a family will fail loudly at import time.
# ---------------------------------------------------------------------------

_FAMILIES: list[tuple[str, list[str]]] = [
    (
        "1. Action / Combat / Heroics",
        [
            "ACTION", "ACTION_EPIC", "B_ACTION", "CAR_ACTION", "GUN_FU",
            "KUNG_FU", "MARTIAL_ARTS", "ONE_PERSON_ARMY_ACTION", "SAMURAI",
            "SUPERHERO", "SWORD_AND_SANDAL", "WUXIA",
        ],
    ),
    (
        "2. Adventure / Journey / Survival",
        [
            "ADVENTURE", "ADVENTURE_EPIC", "ANIMAL_ADVENTURE",
            "DESERT_ADVENTURE", "DINOSAUR_ADVENTURE", "DISASTER",
            "GLOBETROTTING_ADVENTURE", "JUNGLE_ADVENTURE",
            "MOUNTAIN_ADVENTURE", "QUEST", "ROAD_TRIP", "SEA_ADVENTURE",
            "SURVIVAL", "SWASHBUCKLER", "URBAN_ADVENTURE",
        ],
    ),
    (
        "3. Crime / Mystery / Suspense / Espionage",
        [
            "BUDDY_COP", "BUMBLING_DETECTIVE", "CAPER", "CONSPIRACY_THRILLER",
            "COZY_MYSTERY", "CRIME", "CYBER_THRILLER", "DRUG_CRIME",
            "EROTIC_THRILLER", "FILM_NOIR", "GANGSTER",
            "HARD_BOILED_DETECTIVE", "HEIST", "LEGAL_THRILLER", "MYSTERY",
            "POLICE_PROCEDURAL", "POLITICAL_THRILLER",
            "PSYCHOLOGICAL_THRILLER", "SERIAL_KILLER", "SPY",
            "SUSPENSE_MYSTERY", "THRILLER", "WHODUNNIT",
        ],
    ),
    (
        "4. Comedy / Satire / Comic Tone",
        [
            "BODY_SWAP_COMEDY", "BUDDY_COMEDY", "COMEDY", "DARK_COMEDY",
            "FARCE", "HIGH_CONCEPT_COMEDY", "PARODY", "QUIRKY_COMEDY",
            "RAUNCHY_COMEDY", "ROMANTIC_COMEDY", "SATIRE",
            "SCREWBALL_COMEDY", "SLAPSTICK", "STONER_COMEDY",
        ],
    ),
    (
        "5. Drama / History / Institutions",
        [
            "COP_DRAMA", "COSTUME_DRAMA", "DRAMA", "EPIC", "FINANCIAL_DRAMA",
            "HISTORICAL_EPIC", "HISTORY", "LEGAL_DRAMA", "MEDICAL_DRAMA",
            "PERIOD_DRAMA", "POLITICAL_DRAMA", "PRISON_DRAMA",
            "PSYCHOLOGICAL_DRAMA", "SHOWBIZ_DRAMA", "TRAGEDY",
            "WORKPLACE_DRAMA",
        ],
    ),
    (
        "6. Horror / Macabre / Creature",
        [
            "B_HORROR", "BODY_HORROR", "FOLK_HORROR", "FOUND_FOOTAGE_HORROR",
            "GIALLO", "HORROR", "MONSTER_HORROR", "PSYCHOLOGICAL_HORROR",
            "SLASHER_HORROR", "SPLATTER_HORROR", "SUPERNATURAL_HORROR",
            "VAMPIRE_HORROR", "WEREWOLF_HORROR", "WITCH_HORROR",
            "ZOMBIE_HORROR",
        ],
    ),
    (
        "7. Fantasy / Sci-Fi / Speculative",
        [
            "ALIEN_INVASION", "ARTIFICIAL_INTELLIGENCE", "CYBERPUNK",
            "DARK_FANTASY", "DYSTOPIAN_SCI_FI", "FAIRY_TALE", "FANTASY",
            "FANTASY_EPIC", "KAIJU", "MECHA", "SCI_FI", "SCI_FI_EPIC",
            "SPACE_SCI_FI", "STEAMPUNK", "SUPERNATURAL_FANTASY",
            "SWORD_AND_SORCERY", "TIME_TRAVEL",
        ],
    ),
    (
        "8. Romance / Relationship",
        [
            "DARK_ROMANCE", "FEEL_GOOD_ROMANCE", "ROMANCE", "ROMANTIC_EPIC",
            "STEAMY_ROMANCE", "TRAGIC_ROMANCE",
        ],
    ),
    (
        "9. War / Western / Frontier",
        [
            "WAR", "WAR_EPIC", "WESTERN", "CLASSICAL_WESTERN",
            "CONTEMPORARY_WESTERN", "SPAGHETTI_WESTERN", "WESTERN_EPIC",
        ],
    ),
    (
        "10. Music / Musical / Performance",
        [
            "CLASSIC_MUSICAL", "CONCERT", "JUKEBOX_MUSICAL", "MUSIC",
            "MUSICAL", "POP_MUSICAL", "ROCK_MUSICAL",
        ],
    ),
    (
        "11. Sports / Competitive Activity",
        [
            "BASEBALL", "BASKETBALL", "BOXING", "EXTREME_SPORT", "FOOTBALL",
            "MOTORSPORT", "SOCCER", "SPORT", "WATER_SPORT",
        ],
    ),
    (
        "12. Audience / Age / Life Stage",
        [
            "FAMILY", "COMING_OF_AGE", "TEEN_ADVENTURE", "TEEN_COMEDY",
            "TEEN_DRAMA", "TEEN_FANTASY", "TEEN_HORROR", "TEEN_ROMANCE",
        ],
    ),
    (
        "13. Animation / Anime Form / Technique",
        [
            "ADULT_ANIMATION", "ANIMATION", "ANIME", "COMPUTER_ANIMATION",
            "HAND_DRAWN_ANIMATION", "ISEKAI", "IYASHIKEI", "JOSEI", "SEINEN",
            "SHOJO", "SHONEN", "SLICE_OF_LIFE", "STOP_MOTION_ANIMATION",
        ],
    ),
    (
        "14. Seasonal / Holiday",
        [
            "HOLIDAY", "HOLIDAY_ANIMATION", "HOLIDAY_COMEDY",
            "HOLIDAY_FAMILY", "HOLIDAY_ROMANCE",
        ],
    ),
    (
        "15. Nonfiction / Documentary / Real-World Media",
        [
            "CRIME_DOCUMENTARY", "DOCUDRAMA", "DOCUMENTARY",
            "FAITH_AND_SPIRITUALITY_DOCUMENTARY", "FOOD_DOCUMENTARY",
            "HISTORY_DOCUMENTARY", "MILITARY_DOCUMENTARY",
            "MUSIC_DOCUMENTARY", "NATURE_DOCUMENTARY", "NEWS",
            "POLITICAL_DOCUMENTARY",
            "SCIENCE_AND_TECHNOLOGY_DOCUMENTARY", "SPORTS_DOCUMENTARY",
            "TRAVEL_DOCUMENTARY", "TRUE_CRIME",
        ],
    ),
    (
        "16. Program / Presentation / Form Factor",
        [
            "BUSINESS_REALITY_TV", "COOKING_COMPETITION", "GAME_SHOW",
            "MOCKUMENTARY", "PARANORMAL_REALITY_TV", "REALITY_TV", "SHORT",
            "SITCOM", "SKETCH_COMEDY", "SOAP_OPERA", "STAND_UP", "TALK_SHOW",
        ],
    ),
    (
        "17. Cultural / National Cinema Tradition",
        [
            "ARABIC", "BENGALI", "CANTONESE", "DANISH", "DUTCH", "FILIPINO",
            "FINNISH", "FRENCH", "GERMAN", "GREEK", "HINDI", "ITALIAN",
            "JAPANESE", "KANNADA", "KOREAN", "MALAYALAM", "MANDARIN",
            "MARATHI", "NORWEGIAN", "PERSIAN", "PORTUGUESE", "PUNJABI",
            "RUSSIAN", "SPANISH", "SWEDISH", "TAMIL", "TELUGU", "THAI",
            "TURKISH", "URDU",
        ],
    ),
    (
        "18. Source Material / Adaptation / Real-World Basis",
        [
            "NOVEL_ADAPTATION", "SHORT_STORY_ADAPTATION", "STAGE_ADAPTATION",
            "TRUE_STORY", "BIOGRAPHY", "COMIC_ADAPTATION",
            "FOLKLORE_ADAPTATION", "VIDEO_GAME_ADAPTATION", "REMAKE",
            "TV_ADAPTATION",
        ],
    ),
    (
        "19. Narrative Mechanics / Endings",
        [
            "PLOT_TWIST", "TWIST_VILLAIN", "TIME_LOOP", "NONLINEAR_TIMELINE",
            "UNRELIABLE_NARRATOR", "OPEN_ENDING", "SINGLE_LOCATION",
            "BREAKING_FOURTH_WALL", "CLIFFHANGER_ENDING", "HAPPY_ENDING",
            "SAD_ENDING", "BITTERSWEET_ENDING",
        ],
    ),
    (
        "20. Story Engine / Setting / Character Archetype",
        [
            "REVENGE", "UNDERDOG", "KIDNAPPING", "CON_ARTIST",
            "POST_APOCALYPTIC", "HAUNTED_LOCATION", "SMALL_TOWN",
            "FEMALE_LEAD", "ENSEMBLE_CAST", "ANTI_HERO",
        ],
    ),
    (
        "21. Viewer Response / Content Sensitivity",
        [
            "FEEL_GOOD", "TEARJERKER", "ANIMAL_DEATH",
        ],
    ),
]


def _build_classification_registry_section() -> str:
    """Render the 21-family classification registry with full definitions.

    Walks _FAMILIES in order; for each member, looks up the registry
    entry and emits `NAME — definition` one per line. Runs a
    consistency check against CLASSIFICATION_ENTRIES so adding a new
    member to the underlying enums without placing it in a family
    fails loudly at import time.
    """
    listed: set[str] = set()
    lines: list[str] = [
        "CLASSIFICATION REGISTRY",
        "",
        (
            "Every member of the registry is listed below, grouped into "
            "the twenty-one canonical concept families. Each entry shows "
            "the exact member name you must emit in the classification "
            "field, followed by its definition. Your final selection must "
            "match one of these names exactly. When a concept could "
            "plausibly fit more than one family, compare the candidate "
            "definitions directly — the definition that names the concept "
            "specifically wins over one that only covers it incidentally."
        ),
        "",
    ]

    for family_header, member_names in _FAMILIES:
        lines.append(family_header)
        lines.append("")
        for name in member_names:
            if name in listed:
                raise RuntimeError(
                    f"UnifiedClassification member {name!r} listed in more "
                    f"than one family in keyword_query_generation._FAMILIES"
                )
            if name not in CLASSIFICATION_ENTRIES:
                raise RuntimeError(
                    f"UnifiedClassification member {name!r} referenced by "
                    f"keyword_query_generation._FAMILIES is not in the "
                    f"registry — update _FAMILIES or the registry to "
                    f"stay in sync."
                )
            listed.add(name)
            entry = entry_for(UnifiedClassification(name))
            lines.append(f"{entry.name} — {entry.definition}")
        lines.append("")

    missing = set(CLASSIFICATION_ENTRIES) - listed
    if missing:
        raise RuntimeError(
            "Registry members not placed in any family in "
            "keyword_query_generation._FAMILIES: "
            + ", ".join(sorted(missing))
        )

    lines.append("---")
    lines.append("")
    return "\n".join(lines)

# ---------------------------------------------------------------------------
# System prompt — modular sections concatenated at module level.
#
# Structure: task → positive-presence invariant → classification
# taxonomy (21 canonical concept families) → near-collision
# disambiguation principles → scope discipline (one pick, no
# abstention) → output field guidance (reasoning fields first,
# then the enum selection).
#
# Prompt authoring conventions applied:
# - Evidence-inventory reasoning (concept_analysis quotes signal
#   phrases with concept-type angles; candidate_shortlist cites the
#   discriminating test from each candidate's definition rather than
#   justifying the final pick)
# - Brief pre-generation fields (both reasoning fields are
#   telegraphic / label form, not prose; no consistency-coupling
#   language like "your choice must be consistent with...")
# - Cognitive-scaffolding field ordering (concept_analysis →
#   candidate_shortlist → classification, with reasoning adjacent to
#   the decision it grounds)
# - Teach evaluation, not outcome shortcuts (no "if word X appears
#   → pick Y" shortcuts; near-collision pairs are taught via the
#   discriminating-test pattern the model applies on its own)
# - Principle-based constraints, not failure catalogs (the 21
#   families are defined by what they cover, not by enumerated
#   miscoding triggers)
# - Example-eval separation (examples used are canonical ones from
#   the finalized proposal routing-guidance section, not drawn from
#   any evaluation test pool)
# - No schema/implementation leakage (no mention of movie_card
#   columns, GIN overlap, source_id integers, backing-column
#   dispatch; the LLM is told "execution resolves the backing
#   storage" without naming the mechanism)
# ---------------------------------------------------------------------------

_TASK = """\
You translate one classification requirement into a single \
best-fitting registry selection. You receive a requirement that has \
already been interpreted, routed, and framed as a positive-presence \
lookup. Your job is not to decide what the user meant — that is \
already done. Your job is to pick exactly one registry member that \
matches the described concept.

Inputs you receive:
- intent_rewrite — the full concrete statement of what the user is \
looking for. Use it to disambiguate the concept when the description \
alone is narrow (for example, a bare concept word is easier to \
resolve when the surrounding query context is visible).
- description — the single classification requirement you are \
translating, always written in positive-presence form ("is a horror \
movie", "is a feel-good movie", "is a French film", "has a twist \
ending").
- routing_rationale — a concept-type label explaining why this \
requirement was routed to this endpoint. Use it as context for what \
kind of classification the concept represents. Do NOT treat it as \
the evidence itself — the signal phrases live in description and \
intent_rewrite.

Trust the upstream routing. If the description looks like it might \
fit another endpoint, still produce the best possible classification \
for it — do not refuse, do not swap endpoints, do not reinterpret. \
Your only output is one member of the classification registry.

---

"""

# ---------------------------------------------------------------------------
# Positive-presence invariant: matches the entity, metadata, award,
# and franchise endpoints. Exclusion ("not a horror movie") arrives
# here as a separate item already flipped to positive form; step 4
# code applies the exclusion on the result set.
# ---------------------------------------------------------------------------

_DIRECTION_AGNOSTIC = """\
POSITIVE-PRESENCE INVARIANT

Every description you receive describes what to search FOR, never \
what to search AGAINST. If the user's original intent was to exclude \
a classification, the description has already been rewritten in \
positive-presence form upstream and a separate execution layer \
handles exclusion on the result set. You always produce the \
classification that identifies movies whose data matches the \
requirement.

---

"""

# ---------------------------------------------------------------------------
# The classification registry, rendered programmatically from
# CLASSIFICATION_ENTRIES and grouped by the 21 canonical families
# defined in _FAMILIES above. Every member name is emitted with its
# full definition so the LLM can disambiguate near-collision pairs
# against the actual definitional text rather than its parametric
# guesses. Built once at module import.
# ---------------------------------------------------------------------------

_CLASSIFICATION_FAMILIES = _build_classification_registry_section()

# ---------------------------------------------------------------------------
# Near-collision disambiguation: this is where the endpoint is most
# likely to silently misfire. The prompt teaches comparative
# discrimination principles rather than enumerated bad pairs, per the
# "teach evaluation, not outcome shortcuts" convention.
# ---------------------------------------------------------------------------

_DISAMBIGUATION = """\
NEAR-COLLISION DISAMBIGUATION

Within a family, several members often cover overlapping territory. \
The final selection is almost always a choice between two or three \
members whose definitions disagree on one specific feature. Four \
comparison principles decide those choices:

Breadth vs. specificity — Prefer the broader member when the query \
gives no signal for a narrower one. "Scary movies" picks the broad \
horror classification, not a specific sub-form (slasher, \
psychological, supernatural) unless the query explicitly signals \
that sub-form's premise. Picking a narrow member on weak evidence \
silently rejects every movie in the broad category that lacks the \
specific tag.

Explicit premise signal — Prefer a narrower member only when the \
query's phrasing cites the premise that defines it. A slasher \
classification needs a stalker / killer premise cited. A zombie \
classification needs zombies cited. A heist classification needs \
the theft / crew / plan premise cited. If the premise is not in the \
query text, the broader family member is the right answer.

Cross-family proximity — Some concepts sit near the boundary \
between two families. "Coming-of-age" is an audience / life-stage \
family member; a "teen drama" is a drama-family-with-teen-audience \
member. "True story" is a real-world-basis adaptation member; a \
"biography" is also in that family but specifically focuses on one \
named person's life. A "remake" (generic retelling) is a source- \
material member; franchise-structural remakes inside a tracked \
franchise are a different endpoint's concern. Decide by which \
feature the query actually emphasizes — the audience, the life \
stage, the person, the retelling motion — and pick the family whose \
definition names that feature.

Mutually exclusive ending / tag pairs — When a query describes an \
ending or a viewer-response tag, the members inside families 19 and \
21 are near-mutually-exclusive. "A movie that makes you cry" points \
to tearjerker; "a movie that leaves you uplifted" points to \
feel-good; "a movie with an unexpected ending" points to plot twist \
rather than any ending-type value. Cite the query phrase that \
names the effect, not your own summary.

When in doubt between two candidates, include BOTH in the \
candidate_shortlist with the test that decides them, even if one \
looks obvious. The explicit comparison is what prevents the \
first-strong-match-wins failure mode — silently skipping it lets a \
surface-word match on the wrong family starve the correct member.

---

"""

# ---------------------------------------------------------------------------
# Scope discipline: single best fit, no abstention, exactly one pick.
# Explicitly addresses the "pick best of an imperfect set" failure
# mode — the model must not refuse or invent.
# ---------------------------------------------------------------------------

_SCOPE_DISCIPLINE = """\
SCOPE AND ABSTENTION

Your output is exactly one member of the classification registry. \
You do not abstain. You do not invent new classifications. You do \
not emit a list.

If no registry member is a clean fit, pick the best of the \
shortlisted candidates anyway — the broader family member is \
usually the right fallback because it has the widest coverage \
and does not require a specific premise signal the query lacks. \
Routing has already committed this item to the keyword endpoint; \
an empty or refused output breaks the pipeline. A best-available \
fit is always preferable.

Do not let the routing_rationale bias you toward over-specificity. \
It is a concept-type label, not a definition — the actual evidence \
is in description and intent_rewrite. If the routing_rationale \
suggests one angle (e.g., "genre classification") and the \
description's phrasing clearly points elsewhere (e.g., a story \
engine rather than a genre), trust the description.

---

"""

# ---------------------------------------------------------------------------
# Output field guidance: per-field instructions in schema order. The
# two reasoning fields carry their framing here so that cognitive
# scaffolding produces its intended effect on the single enum
# selection.
# ---------------------------------------------------------------------------

_OUTPUT = """\
OUTPUT FIELD GUIDANCE

Generate fields in the schema's order. The two reasoning fields \
come first — they scaffold the single selection that follows. \
Inventory the signals, compare the candidates, then commit.

concept_analysis — FIRST field. An evidence inventory that grounds \
the selection. For each signal phrase you see in description and \
intent_rewrite, quote it verbatim and pair it with the concept-type \
angle it implies. The angles correspond to the twenty-one families \
above; use one of these labels:
- genre-like / sub-genre
- cultural or national cinema tradition
- source material or real-world basis
- form factor or presentation
- audience tier, age, or life stage
- narrative device or ending type
- story engine, setting, or character archetype
- viewer-response tag or content sensitivity

Telegraphic form. Semicolon-separated phrase → angle pairs. \
Examples of the form:
- "\\"scary\\" → genre-like; horror-family"
- "\\"Bollywood\\" → cultural tradition (Hindi)"
- "\\"twist ending\\" → narrative device"
- "\\"growing up\\" → audience tier / story engine (ambiguous \
between coming-of-age and teen drama)"

If the phrasing is genuinely ambiguous between two angles, surface \
the ambiguity in one short clause (as in the last example) so the \
candidate_shortlist below is deliberate rather than a default pick. \
Do not hedge by listing every possible angle — just the ones the \
query text actually signals.

Ignore routing_rationale when extracting signal phrases. It is an \
already-interpreted label and anchoring on it re-introduces routing \
bias. Use it only as background context, not as evidence.

One to three phrase → angle pairs is typical. Do not pad.

candidate_shortlist — SECOND field, placed immediately before the \
final selection. Two or three candidate registry members (by exact \
UnifiedClassification member name), each annotated with the one \
test from its definition that decides fit. Bar-separated. Label \
form, not sentences.

Format: `MEMBER_NAME: discriminator — present/absent`. Examples of \
the form:
- "HORROR: broad horror-genre match | PSYCHOLOGICAL_HORROR: \
requires mental-unraveling premise — absent | SLASHER_HORROR: \
requires stalker/killer premise — absent"
- "TRUE_STORY: real events dramatization — present | BIOGRAPHY: \
focused on one person's life — no person named"
- "FEEL_GOOD_ROMANCE: requires romance as the vehicle — absent | \
FEEL_GOOD: emotional-uplift tag, broader — present"
- "COMING_OF_AGE: growing-up as the axis — present | TEEN_DRAMA: \
teen-audience drama — query emphasizes theme, not audience tier"

Procedure:
1. Using the concept-type angle(s) from concept_analysis, identify \
two to three plausible registry members that could cover the \
concept. Prefer candidates in the same family; cross-family \
candidates belong in the shortlist only when concept_analysis \
surfaced cross-family ambiguity.
2. For each candidate, cite the ONE distinguishing test from its \
definition — the premise, scope, or feature that separates it from \
its siblings. Mark whether the query signals that feature as \
present or absent.
3. Apply the breadth-vs-specificity rule: absent specific premise \
signals, the broader member wins. Apply the cross-family proximity \
rule when the shortlist spans families.
4. ALWAYS include the near-collision candidate even when one member \
looks obvious. The explicit comparison is the anti-shortcut \
mechanism — skipping it invites surface-word matching.

Do NOT write a sentence arguing for the final pick. The \
discriminator-plus-present/absent form is the reasoning; the \
selection follows from it.

classification — The UnifiedClassification registry member you \
commit to. Exactly one member. This must be one of the candidates \
you named in candidate_shortlist — the shortlist is not ornamental, \
it is the set you chose from.
"""

SYSTEM_PROMPT = (
    _TASK
    + _DIRECTION_AGNOSTIC
    + _CLASSIFICATION_FAMILIES
    + _DISAMBIGUATION
    + _SCOPE_DISCIPLINE
    + _OUTPUT
)


async def generate_keyword_query(
    intent_rewrite: str,
    description: str,
    routing_rationale: str,
    provider: LLMProvider,
    model: str,
    **kwargs,
) -> tuple[KeywordQuerySpec, int, int]:
    """Translate one keyword dealbreaker or preference into a KeywordQuerySpec.

    The LLM receives the step 1 intent_rewrite (for disambiguation
    context) and one step 2 item's description plus routing_rationale.
    It produces exactly one UnifiedClassification registry selection,
    preceded by the two reasoning fields that scaffold the choice.

    Args:
        intent_rewrite: The full concrete statement of what the user is
            looking for, from step 1.
        description: The positive-presence statement of the
            classification requirement to translate (from a Dealbreaker
            or Preference).
        routing_rationale: The concept-type label from step 2 explaining
            why this item was routed to the keyword endpoint.
        provider: Which LLM backend to use. No default — callers must
            choose explicitly so call sites are self-documenting and
            we can A/B test providers.
        model: Model identifier for the chosen provider. No default
            for the same reason as provider.
        **kwargs: Provider-specific parameters forwarded directly to
            the underlying LLM call (e.g., reasoning_effort,
            temperature, budget_tokens).

    Returns:
        A tuple of (KeywordQuerySpec, input_tokens, output_tokens).
    """
    intent_rewrite = intent_rewrite.strip()
    description = description.strip()
    routing_rationale = routing_rationale.strip()
    if not intent_rewrite:
        raise ValueError("intent_rewrite must be a non-empty string.")
    if not description:
        raise ValueError("description must be a non-empty string.")
    if not routing_rationale:
        raise ValueError("routing_rationale must be a non-empty string.")

    # Three labeled sections so the model can keep inputs distinct.
    user_prompt = (
        f"intent_rewrite: {intent_rewrite}\n"
        f"description: {description}\n"
        f"routing_rationale: {routing_rationale}"
    )

    response, input_tokens, output_tokens = await generate_llm_response_async(
        provider=provider,
        user_prompt=user_prompt,
        system_prompt=SYSTEM_PROMPT,
        response_format=KeywordQuerySpec,
        model=model,
        **kwargs,
    )

    return response, input_tokens, output_tokens

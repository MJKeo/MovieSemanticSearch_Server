# Search V2 — Stage 2B: retrieval action planning.
#
# Stage 2B consumes a Stage 2A `Step2AResponse` (plus the Stage 1
# `intent_rewrite` for grounding) and produces one `CompletedSlot`
# per planning slot. Each per-slot call runs independently as its
# own LLM request — N slots → N parallel calls — and produces either
# a sibling group of retrieval actions or a slot-level skip.
#
# Design principles (see search_improvement_planning/
# steps_1_2_improving.md, "Step 2B Redesign Proposal"):
#   - 2A is context, not truth (rerouting / expansion / refusal allowed).
#   - Concept = slot (each call emits exactly one sibling group).
#   - Expansion is a decision with three named motives.
#   - Positive-framing discipline: descriptions always say what we
#     want to match, regardless of role.
#   - Per-atom verdict scaffold with skip-as-first-class.
#
# The system prompt is monolithic — one prompt covers all endpoint
# families. Branch-dynamic dispatch would architecturally commit to
# 2A's family choice, which contradicts 2B's rerouting autonomy.

from __future__ import annotations

import asyncio
import logging

from implementation.llms.generic_methods import LLMProvider, generate_llm_response_async
from schemas.query_understanding import (
    CompletedSlot,
    PlanningSlot,
    QueryUnderstandingResponse,
    Step2AResponse,
    Step2BResponse,
)

logger = logging.getLogger(__name__)


_STEP_2B_SYSTEM_PROMPT = """\
You plan retrieval actions for ONE planning slot of a movie search
query. For that slot, decide how to retrieve the movies it describes
— or refuse the slot if it cannot be coherently expressed.

YOUR INPUTS
- Full query intent (`intent_rewrite`). Use this to resolve ambiguous
  wording in the focal slot's atoms against the full query.
- The focal slot: handle, scope (the atoms you must plan for),
  retrieval_shape (2A's advisory hint), cohesion (why 2A grouped
  these atoms), confidence (literal / inferred).
- Sibling slots: handle, retrieval_shape, scope. Provided for
  coverage awareness only — you never plan for them.

YOUR OUTPUT
All actions you produce are SIBLINGS under the focal slot. Downstream
scoring combines them as MAX-within-slot, additive-across-slots. An
empty action list (with a `skip_rationale` set) is a legitimate
answer when the slot cannot be expressed.

WHAT THIS STAGE IS NOT
- Not Stage 2A: the slot is given. Do not re-split, merge, or
  renegotiate its scope. Either plan for it or skip it.
- Not Stage 3: do not write query bodies, enum values, or specific
  metadata column names. Describe at the level of "what we want to
  match."
- Not Stage 4: do not worry about combining across slots.
- Do not plan for sibling slots — each gets its own call.

=== REASONING FLOW ===

The `atom_analysis` field carries a per-atom verdict trace. For every
atom in `focal_slot.scope`, write one line that commits three
decisions BEFORE you write any action:

  "<atom verbatim>" -> coverage: <verdict> | role: <verdict> | route: <endpoint> (<≤8 words why>)

Coverage verdict (how many actions this atom needs):
  single                         — one action covers it (DEFAULT)
  expand:ambiguity_fan_out       — atom spans multiple distinct angles
  expand:paraphrase_redundancy   — multiple lexical framings of one target
  expand:defensive_retrieval     — multiple endpoints because each has a gap

Expansion is a decision, not a reflex. The ≤8-word why must name a
SPECIFIC mechanism — which angles, which paraphrases, which endpoint
gap. Vague appeals ("to be thorough", "more robust", "for safety")
do not justify expansion. If you cannot name a concrete mechanism,
`coverage: single` is the correct answer.

Role verdict (how Stage 4 uses this atom):
  inclusion            — hard requirement phrased positively
  exclusion            — hard requirement phrased negatively
  preference:core      — dominant ranking instruction for this slot
  preference:supporting — secondary ranking refinement

Role follows USER PHRASING strength, not your sense of what would be
useful:
  "must", "only", "has to", implicit must → inclusion
  "no", "not", "without", "avoid", "except" → exclusion
  "prefer", "better", "more", "especially", "ideally" → preference

Kind-layering within one slot is legitimate. A slot may naturally
want BOTH a hard filter AND a preference (e.g., a keyword-horror
inclusion + a semantic-scariest preference). Do not force one role
per slot.

Confidence (`literal` / `inferred`) is ONE signal among several —
not a gate. An inferred atom with unambiguous "must have" phrasing
is still inclusion. An inferred atom with soft phrasing is a
preference. Read the phrasing.

Route verdict names the committed endpoint. One of: entity, studio,
metadata, awards, franchise_structure, keyword, semantic, trending.
The ≤8-word why cites the ROUTE's capability, not the atom's wording.

Slot-level skip is a first-class verdict. If the slot cannot be
coherently expressed (every atom hits a capability mismatch, or 2A
grouped atoms that do not cohere retrievably), emit this line INSTEAD
of per-atom lines, or AFTER them if some atoms resolved and the slot
still cannot stand on what's resolvable:

  skip_slot: <reason tied to a specific atom or capability mismatch>

When you write `skip_slot`, set `actions = []` and
`skip_rationale = "<the same reason>"`. When you do NOT write
`skip_slot`, set `skip_rationale = null`.

=== ROUTE CAPABILITIES ===

`retrieval_shape` is 2A's advisory hint. Honor it when its capability
fits the atom. Reroute when an atom reveals a capability mismatch
2A did not catch. Neither deference nor rerouting is virtuous —
capability match is the only criterion.

entity
  CAN: match named actors, directors, writers, or specific fictional
  characters; preserve prominence wording verbatim ("starring",
  "supporting", "cameo", "centers on", "includes").
  CANNOT: infer creative influence ("in the style of X"), match
  role-types beyond the named person, or catch uncredited appearances.

studio
  CAN: match production companies, labels, and studios that MADE the
  film.
  CANNOT: represent watch-availability (Netflix-the-platform is not
  Netflix-the-studio); mix up distribution with production.

metadata
  CAN: match quantitative / factual attributes — release_date ranges,
  runtime, maturity rating, watch availability on tracked streamers,
  audio language, country of origin, budget, box office, general
  reception (critics + audience), and general (global) popularity.
  CANNOT: do demographic popularity (only global); personalize to a
  user; match anything that is not a structured attribute.

awards
  CAN: match award-related lookups — specific ceremonies / categories
  / years / outcomes, or a generic "award-winning" intent.
  CANNOT: represent generic critical acclaim without a specific award
  (route that to semantic); represent internet buzz or informal honors.

franchise_structure
  CAN: match franchise membership and structural role (sequel,
  prequel, reboot, spinoff, crossover, launcher).
  CANNOT: find films that merely FEATURE a franchise character — the
  franchise table only tracks titles formally part of a franchise.
  Character-led retrieval belongs on entity.

keyword
  CAN: match CLOSED-TAXONOMY tags — common genres, tone tags, source
  material, concept tags (examples: horror, coming-of-age,
  based-on-true-story, remake, stop-motion, feel-good, revenge,
  happy-ending).
  CANNOT: extend beyond the existing taxonomy — unknown concepts
  (e.g., "sleeper", "underground") must reroute to semantic. Cannot
  handle subjective vibe gradations.

semantic
  CAN: match subjective, thematic, experiential, plot-description,
  production-style, or reception-trait traits that are not clean
  deterministic fits. Valid as a preference even when another route
  covers the same concept deterministically.
  CANNOT: match exact structural values (year, runtime, specific
  cast); serve as a HARD filter for cleanly deterministic concepts
  (those belong in keyword or metadata).

trending
  CAN: match explicit "right now / currently buzzing / trending this
  week" intent.
  CANNOT: represent historical popularity, all-time bestsellers, or
  anything not time-localized to the present moment.

Boundary reminders (these preserve prior behavior that is easy to
lose under generic capability matching):
  "classic" stays semantic, not keyword.
  "Marvel movies" / "Spider-Man movies" benefit from BOTH
    franchise_structure AND entity (the defensive-retrieval motive).
  Entity prominence wording ("starring", "centers on") must appear
    verbatim in the action's description.

=== DESCRIPTION DISCIPLINE ===

Every `description` is written in the POSITIVE — as "what we want to
match" — regardless of role. Direction is carried by the `role`
field, not by the description text.

Failure → Fix (exclusion still positive):
  role: exclusion, description: "not made in the 1980s"         ← WRONG
  role: exclusion, description: "movies released in the 1980s"  ← RIGHT

Stage 3 sees only the description. It builds a query that matches
the described trait. Stage 4 then applies the role: include matches,
subtract matches, or score matches. This is why the description
must stay positive in every role.

Preserve user phrasing verbatim when it carries signal — especially
entity prominence ("starring X", "centers on the character X") and
evaluative breadth ("best", "top", "classic", "great", "favorite").
Do NOT translate these to your own words.

=== COVERAGE GROUNDING ===

`coverage_atoms` must contain EXACT verbatim atoms from
`focal_slot.scope`. Not paraphrased. Not substituted. Not stripped
of punctuation. If you cannot cite the atom, you did not cover it.

Multi-atom coverage is legitimate when one action naturally handles
multiple atoms — common for paraphrase redundancy or semantic
preferences that blend related traits into one vector query.

Every atom in `focal_slot.scope` must be covered by at least one
action, OR the whole slot must be skipped. Uncovered atoms silently
drop user intent.

=== ROUTE RATIONALE ===

`route_rationale` cites the ROUTE's capability — not the atom's
wording. ≤15 words.

  CORRECT:   "metadata exposes release_date as a numeric range"
  INCORRECT: "the atom talks about dates"

When rerouting away from `retrieval_shape`, cite the capability
mismatch that triggered the reroute (name what the advised route
CANNOT do).

=== BREVITY CAPS ===
- Each per-atom trace line: ≤30 words.
- `route_rationale`: ≤15 words, cites capability.
- `skip_rationale`: ≤20 words, ties to atom or capability.
- `description`: one sentence, ≤25 words, positive framing.

=== EXAMPLES ===

Each example illustrates ONE principle. Treat them as principle
illustrations, not templates to pattern-match.

1. Single action (canonical).
Focal slot: handle="tokyo-setting", scope=["set in Tokyo"],
retrieval_shape="semantic location atmosphere".
atom_analysis:
  atoms:
  - "set in Tokyo" -> coverage: single | role: inclusion | route: semantic (location atmosphere is a vector trait)
Actions:
  - coverage_atoms=["set in Tokyo"], description="movies set in Tokyo",
    route_rationale="semantic vectors capture setting / location mood",
    route=semantic, role=inclusion.

2. Ambiguity fan-out.
Focal slot: handle="loud-films", scope=["loudest"],
retrieval_shape="semantic intensity".
atom_analysis:
  atoms:
  - "loudest" -> coverage: expand:ambiguity_fan_out (sonic intensity vs cultural controversy) | role: preference:core | route: semantic (both angles are vector traits)
Actions:
  - covers "loudest", description="movies with extreme sonic intensity and
    high-volume action", route=semantic, role=preference, strength=core.
  - covers "loudest", description="culturally controversial, widely discussed
    movies", route=semantic, role=preference, strength=core.

3. Paraphrase redundancy.
Focal slot: handle="holiday-flavor", scope=["christmas"],
retrieval_shape="keyword holiday".
atom_analysis:
  atoms:
  - "christmas" -> coverage: expand:paraphrase_redundancy (christmas + holiday taxonomy tags) | role: inclusion | route: keyword (both tags are in the taxonomy)
Actions:
  - covers "christmas", description="christmas movies", route=keyword,
    role=inclusion.
  - covers "christmas", description="holiday movies", route=keyword,
    role=inclusion.

4. Defensive retrieval.
Focal slot: handle="indiana-jones", scope=["Indiana Jones"],
retrieval_shape="franchise membership".
atom_analysis:
  atoms:
  - "Indiana Jones" -> coverage: expand:defensive_retrieval (franchise table misses character-led spinoffs) | role: inclusion | route: franchise_structure and entity
Actions:
  - covers "Indiana Jones", description="Indiana Jones franchise films",
    route=franchise_structure, role=inclusion.
  - covers "Indiana Jones", description="movies that center on the character
    Indiana Jones", route=entity, role=inclusion.

5. Kind-layering within one slot.
Focal slot: handle="scariest-horror", scope=["horror", "scariest"],
retrieval_shape="keyword horror plus semantic scariest".
atom_analysis:
  atoms:
  - "horror" -> coverage: single | role: inclusion | route: keyword (horror is a closed-taxonomy genre)
  - "scariest" -> coverage: single | role: preference:core | route: semantic (subjective intensity is a vector trait)
Actions:
  - covers "horror", description="horror movies", route=keyword,
    role=inclusion.
  - covers "scariest", description="the scariest, most frightening horror
    movies", route=semantic, role=preference, strength=core.

6. Rerouting away from retrieval_shape.
Focal slot: handle="sleeper-vibe", scope=["underground sleeper"],
retrieval_shape="keyword sleeper".
atom_analysis:
  atoms:
  - "underground sleeper" -> coverage: single | role: preference:core | route: semantic (keyword taxonomy has no sleeper tag)
Actions:
  - covers "underground sleeper", description="underground sleeper-hit
    movies that quietly gained a devoted following",
    route_rationale="keyword taxonomy cannot represent sleeper / underground
    status; semantic vectors can", route=semantic, role=preference,
    strength=core.

7. Exclusion with positive-framed description.
Focal slot: handle="not-eighties", scope=["avoid the 1980s"],
retrieval_shape="metadata decade".
atom_analysis:
  atoms:
  - "avoid the 1980s" -> coverage: single | role: exclusion | route: metadata (decade is a date range attribute)
Actions:
  - covers "avoid the 1980s", description="movies released in the 1980s",
    route=metadata, role=exclusion.

8. Slot-level skip.
Focal slot: handle="boomer-loved", scope=["films especially beloved by
boomers"], retrieval_shape="metadata demographic popularity".
atom_analysis:
  atoms:
  - "films especially beloved by boomers" -> capability mismatch
  skip_slot: metadata exposes global popularity only; no demographic breakdown exists
Output: actions=[],
skip_rationale="metadata exposes global popularity only; no demographic
breakdown exists".

=== OUTPUT FIELDS (in schema order) ===

1. `atom_analysis` — the per-atom verdict trace, plus a `skip_slot`
   line when skipping.
2. `skip_rationale` — non-null iff the slot is skipped; otherwise
   null.
3. `actions` — one RetrievalAction per committed (atom / expansion)
   pair. Empty iff `skip_rationale` is set. Each action's fields
   appear in decision order: coverage_atoms, description,
   route_rationale, route, role, preference_strength.
"""


def _clean_atom(atom: str) -> str:
    """Strip embedded double-quote characters so atoms can be wrapped
    in quotes when rendered without delimiter collision. 2A atoms
    occasionally carry user-phrased quote marks (e.g. a "best" films
    atom); inline quotes collide with our quote-delimited prompt form
    and confuse the model. Coverage validation applies the same strip
    to both sides so the LLM's echoed atoms match the cleaned scope.
    """
    return atom.replace('"', "")


def _render_sibling_line(sibling: PlanningSlot) -> str:
    """One compact line per sibling — handle, retrieval_shape, scope."""
    scope_str = ", ".join(f'"{_clean_atom(atom)}"' for atom in sibling.scope)
    return (
        f"- {sibling.handle}: {sibling.retrieval_shape} "
        f"— covers: {scope_str}"
    )


def _render_user_prompt(
    *,
    intent_rewrite: str,
    focal: PlanningSlot,
    siblings: list[PlanningSlot],
) -> str:
    """Assemble the per-slot user prompt.

    Sibling slots are rendered in compact form (no cohesion, no
    confidence, no verdicts). Intent rewrite anchors the whole
    request. The focal slot is rendered in full so the model has
    cohesion and confidence available for role decisions.
    """
    focal_scope = ", ".join(f'"{_clean_atom(atom)}"' for atom in focal.scope)

    if siblings:
        sibling_block = "\n".join(_render_sibling_line(s) for s in siblings)
    else:
        sibling_block = "- (none)"

    return (
        f"Full query intent:\n{intent_rewrite.strip()}\n\n"
        f"Focal slot:\n"
        f"- Handle: {focal.handle}\n"
        f"- Scope: {focal_scope}\n"
        f"- Advisory retrieval shape: {focal.retrieval_shape}\n"
        f"- Cohesion: {focal.cohesion}\n"
        f"- Confidence: {focal.confidence}\n\n"
        f"Sibling slots (for coverage awareness; do not plan for them):\n"
        f"{sibling_block}\n"
    )


def _validate_coverage(focal: PlanningSlot, response: Step2BResponse) -> None:
    """Enforce partition-completeness between the slot scope and actions.

    Pydantic validators on Step2BResponse only check shape-level
    invariants (skip XOR actions, strength pairing). This check runs
    against the slot to confirm the action set (if non-empty) covers
    every atom exactly once or more — same math as 2A's validator,
    one level deeper. Both sides are quote-cleaned so the comparison
    matches the atoms as rendered in the prompt.
    """
    if not response.actions:
        # Skip path; no coverage check. The Pydantic XOR validator
        # has already ensured skip_rationale is non-empty.
        return

    scope = {_clean_atom(atom) for atom in focal.scope}
    covered: set[str] = set()
    for action in response.actions:
        for atom in action.coverage_atoms:
            cleaned = _clean_atom(atom)
            if cleaned not in scope:
                raise ValueError(
                    f"Action coverage_atom {atom!r} does not match any atom "
                    f"in focal slot {focal.handle!r} scope {sorted(scope)}."
                )
            covered.add(cleaned)

    uncovered = scope - covered
    if uncovered:
        raise ValueError(
            f"Slot {focal.handle!r} has atoms not covered by any action: "
            f"{sorted(uncovered)}. Either cover them or skip the slot."
        )


async def _single_slot_attempt(
    *,
    intent_rewrite: str,
    focal: PlanningSlot,
    siblings: list[PlanningSlot],
    provider: LLMProvider,
    model: str,
    **kwargs,
) -> tuple[Step2BResponse, int, int]:
    """One attempt at a per-slot LLM call. Validates coverage before
    returning — any validation failure propagates to the retry layer."""
    user_prompt = _render_user_prompt(
        intent_rewrite=intent_rewrite,
        focal=focal,
        siblings=siblings,
    )
    response, input_tokens, output_tokens = await generate_llm_response_async(
        provider=provider,
        user_prompt=user_prompt,
        system_prompt=_STEP_2B_SYSTEM_PROMPT,
        response_format=Step2BResponse,
        model=model,
        **kwargs,
    )
    _validate_coverage(focal, response)
    return response, input_tokens, output_tokens


async def _run_step_2b_for_slot(
    *,
    intent_rewrite: str,
    focal: PlanningSlot,
    siblings: list[PlanningSlot],
    provider: LLMProvider,
    model: str,
    **kwargs,
) -> tuple[Step2BResponse, int, int]:
    """Single-retry wrapper around `_single_slot_attempt`.

    Retries once on ANY exception — LLM transport errors, Pydantic
    validation errors (shape / strength pairing / skip XOR), and
    runtime coverage validation errors are all retryable here. The
    second attempt uses the same prompt; in practice the stochastic
    decode produces different output often enough that one retry
    meaningfully improves success rate without blowing up cost.

    If the retry also fails, the exception from the second attempt
    propagates. `run_stage_2b` catches it at the gather boundary so
    one failing slot never takes down the whole branch.
    """
    try:
        return await _single_slot_attempt(
            intent_rewrite=intent_rewrite,
            focal=focal,
            siblings=siblings,
            provider=provider,
            model=model,
            **kwargs,
        )
    except Exception as first_exc:  # noqa: BLE001 — retry policy is broad by design
        logger.warning(
            "Step 2B slot %r first attempt failed: %s. Retrying once.",
            focal.handle,
            first_exc,
        )
        return await _single_slot_attempt(
            intent_rewrite=intent_rewrite,
            focal=focal,
            siblings=siblings,
            provider=provider,
            model=model,
            **kwargs,
        )


async def run_stage_2b(
    *,
    intent_rewrite: str,
    stage_2a: Step2AResponse,
    provider: LLMProvider,
    model: str,
    **kwargs,
) -> tuple[QueryUnderstandingResponse, int, int]:
    """Plan retrieval actions for every slot produced by Stage 2A.

    Fans out one parallel LLM call per slot. Each call receives the
    intent_rewrite, its focal slot in full, and its sibling slots in
    compact form. Per-slot failures are isolated: a slot that fails
    both its attempts is dropped from the output with an error log;
    siblings that succeeded are preserved. The whole branch only fails
    if EVERY slot fails — a soft-fail posture consistent with Stage 4's
    endpoint handling.

    Args:
        intent_rewrite: Stage 1 rewrite; grounds ambiguous atom wording.
        stage_2a: the completed Step 2A response; `slots` drives the
            fan-out, other fields are NOT forwarded to the LLM (2B
            treats 2A's trace as internal scaffolding, not evidence).
        provider, model, kwargs: passed to the LLM router.

    Returns:
        (QueryUnderstandingResponse with one CompletedSlot per
         successful slot, summed input tokens, summed output tokens).

    If stage_2a.slots is empty (2A produced no actionable partition),
    returns a QueryUnderstandingResponse with an empty completed_slots
    list and zero token counts. No LLM calls fire.
    """
    intent_rewrite = intent_rewrite.strip()
    if not intent_rewrite:
        raise ValueError("intent_rewrite must be a non-empty string.")

    slots = stage_2a.slots
    if not slots:
        return (
            QueryUnderstandingResponse(completed_slots=[]),
            0,
            0,
        )

    async def one(index: int) -> tuple[Step2BResponse, int, int]:
        focal = slots[index]
        siblings = [s for i, s in enumerate(slots) if i != index]
        return await _run_step_2b_for_slot(
            intent_rewrite=intent_rewrite,
            focal=focal,
            siblings=siblings,
            provider=provider,
            model=model,
            **kwargs,
        )

    # return_exceptions=True isolates failures: one slot's exception
    # no longer cancels every other in-flight slot call. The token
    # budget spent on siblings is preserved regardless of outcome.
    results = await asyncio.gather(
        *(one(i) for i in range(len(slots))),
        return_exceptions=True,
    )

    completed_slots: list[CompletedSlot] = []
    total_input = 0
    total_output = 0
    failures: list[tuple[str, BaseException]] = []

    for slot, result in zip(slots, results, strict=True):
        if isinstance(result, BaseException):
            failures.append((slot.handle, result))
            logger.error(
                "Step 2B slot %r failed after retry; dropping. Error: %s",
                slot.handle,
                result,
            )
            continue
        response, in_tok, out_tok = result
        completed_slots.append(CompletedSlot(slot=slot, response=response))
        total_input += in_tok
        total_output += out_tok

    # Preserve the "every slot failed" signal — downstream gets a
    # useful error rather than an empty completed_slots list that
    # silently degrades into a browse-flow fallback.
    if failures and not completed_slots:
        _, first_exc = failures[0]
        raise RuntimeError(
            f"Step 2B failed for all {len(failures)} slot(s). "
            f"First error: {first_exc}"
        ) from first_exc

    return (
        QueryUnderstandingResponse(completed_slots=completed_slots),
        total_input,
        total_output,
    )

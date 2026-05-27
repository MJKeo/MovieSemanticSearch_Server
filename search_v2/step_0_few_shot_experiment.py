# Step 0 few-shot experiment harness.
#
# Compares three prompt variants on the same test set:
#   V1 — current SYSTEM_PROMPT (baseline)
#   V2 — V1 with softened COVERAGE PRINCIPLE that defers to QUALIFIER
#        RULE on packaging tokens
#   V3 — V2 with appended per-route few-shot examples (packaging vs
#        qualifier contrast pairs)
#
# For each (query, variant) cell we record the selected entity flow
# and whether standard co-fires, then print a comparison table plus
# the model's reasoning for every disagreement with the expected
# routing.
#
# Usage:
#   python -m search_v2.step_0_few_shot_experiment

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

from implementation.llms.generic_methods import generate_llm_response_async  # noqa: E402
from schemas.step_0_flow_routing import EntityFlow, Step0Response  # noqa: E402
from search_v2.step_0 import (  # noqa: E402
    _AMBIGUITY_PRINCIPLE,
    _COVERAGE_PRINCIPLE,
    _OUTPUT_FIELD_GUIDANCE,
    _QUALIFIER_RULE,
    _RESOLUTION_PRINCIPLE,
    _SIMILARITY_FRAME_RULE,
    _STEP_0_KWARGS,
    _STEP_0_MODEL,
    _STEP_0_PROVIDER,
    _TASK_AND_OUTCOME,
    _ZONE_STRUCTURE,
)


# ---------------------------------------------------------------------------
# Variant pieces
# ---------------------------------------------------------------------------

# V2 swap: the entity span no longer has to equal the entire query
# verbatim. After packaging tokens are dropped per the QUALIFIER RULE,
# the entity must cover what remains. Real qualifiers still disqualify
# entity routing — only "packaging" gets the pass.
_COVERAGE_PRINCIPLE_SOFTENED = """\
COVERAGE PRINCIPLE — entity flows require the entity to cover all \
non-packaging content

For an entity flow to apply, the entity span must equal the entire \
query AFTER dropping packaging tokens per the QUALIFIER RULE. \
Packaging includes politeness, speech-act framing, conversational \
filler and discourse markers, vacuous quantifiers, bare result-type \
words (e.g. "movies", "films", "series", "stuff", "recs"), and \
similarity-frame phrasing where that frame is the entity reading \
being recognized. Only real qualifiers — phrases that change the \
meaning of the search relative to a bare-entity lookup — disqualify \
entity routing. A query like "tom hanks movies" routes to the person \
flow because "movies" is a bare result-type word; a query like \
"<entity> in space" carries leftover content ("in space") that pushes \
it to none_of_the_above. When in doubt, apply the operating test from \
the QUALIFIER RULE: would removing this token meaningfully change the \
result set, ranking, or selection criteria? If no, it is packaging \
and does not disqualify entity routing.

The same coverage rule applies to similarity_to_titles, with one \
allowance: the similarity-frame phrasing itself ("movies like", \
"similar to", "in the vein of", "meets", "crossed with", "X but Y" \
when Y is itself a film reference) is part of the frame, not leftover \
content. The reference titles still have to be the only content \
inside the frame after packaging is dropped.

The studio and person flows are also list-shaped: a query may name \
one entity ("Pixar", "Tom Hanks", "Christopher Nolan") or several \
joined by a neutral conjunction ("Pixar and Studio Ghibli", "Tom \
Hanks and Woody Harrelson", "Spielberg and John Williams"). The \
connectors "and", "&", and comma-separated lists are part of the \
list shape, not leftover content. ANY other content that survives the \
packaging-drop step — descriptors, role markers, mixed entity kinds, \
similarity framing — disqualifies the studio / person flow and pushes \
the query to none_of_the_above. A list-shaped studio or person flow \
must be homogeneous: all entries must be studios, or all entries must \
be people. Mixing kinds (e.g. one person and one studio) is \
none_of_the_above.

Role markers in the query ("directed by", "starring", "score by", \
"written by", "produced by") count as content, not packaging — they \
disqualify the person flow because they constrain the search beyond \
the bare-name lookup. Surface them as qualifiers and route to \
none_of_the_above. A query that is just a person's name (with no \
role marker) fires the person flow regardless of which role that \
person is most known for.

---

"""


# V3 addition: per-route few-shot examples. Each route has a "fires"
# group (entity wins despite packaging) and a contrast group (an
# extra qualifier pushes the same shape to none_of_the_above). Routes
# are ordered by how often the baseline trips over packaging.
_FEW_SHOT_EXAMPLES = """\
EXAMPLES — packaging vs qualifiers across each entity flow

These examples illustrate the boundary between packaging (which does \
NOT disqualify entity routing) and real qualifiers (which force \
none_of_the_above). For each route, the first group shows queries \
where the entity flow fires despite surface packaging. The second \
group shows superficially similar queries where one real qualifier \
forces none_of_the_above so the standard flow can apply the \
constraint.

PERSON flow — fires:
- "tom hanks movies" — "movies" is a bare result-type word
- "anything with meryl streep" — "anything with" is conversational \
packaging
- "show me spielberg films" — speech-act + result-type word
- "denis villeneuve" — bare name
- "scorsese and de niro" — list of two people, neutral conjunction
- "i wanna watch some keanu reeves" — politeness + speech-act + \
vacuous quantifier
PERSON flow — does NOT fire (qualifier present → none_of_the_above):
- "funny tom hanks movies" — "funny" is a genre/mood qualifier
- "meryl streep musicals" — "musicals" is a genre constraint
- "spielberg sci-fi on netflix" — genre + streaming qualifiers
- "christopher nolan but lighter" — comparison/mood modifier

STUDIO flow — fires:
- "pixar movies" — "movies" is a result-type word
- "anything by disney" — "anything by" is conversational packaging
- "a24 films" — "films" is a result-type word
- "studio ghibli" — bare studio name
- "stuff from blumhouse" — "stuff from" is conversational filler
- "pixar and studio ghibli" — list of two studios
STUDIO flow — does NOT fire:
- "pixar movies for adults" — "for adults" is an audience qualifier
- "disney animated classics" — format + era qualifiers
- "scary a24" — genre/mood qualifier
- "blumhouse movies under 90 minutes" — runtime qualifier

CHARACTER_FRANCHISE flow — fires:
- "james bond movies" — result-type word
- "batman" — bare character name
- "the spider-man films" — result-type packaging
- "any john wick" — vacuous quantifier
- "show me indiana jones" — speech-act framing
CHARACTER_FRANCHISE flow — does NOT fire:
- "batman movies from the 90s" — era qualifier
- "funniest james bond" — quality + tone qualifier
- "spider-man but animated" — format modifier
- "dark batman" — tone qualifier
- "indiana jones for kids" — audience qualifier

NON_CHARACTER_FRANCHISE flow — fires:
- "star wars movies" — result-type word
- "the fast and furious franchise" — packaging naming the result kind
- "mcu films" — result-type word
- "lord of the rings" — bare franchise name
- "anything in the alien universe" — conversational packaging around \
the IP umbrella
NON_CHARACTER_FRANCHISE flow — does NOT fire:
- "best star wars" — quality qualifier
- "mcu phase 4" — sub-slice qualifier
- "underrated star trek" — reception qualifier
- "jurassic park sequels only" — subset qualifier

SPECIFIC_TITLE flow — fires:
- "inception" — bare title
- "the movie inception" — result-type packaging
- "watch parasite" — speech-act framing
- "play the godfather" — speech-act framing
- "everything everywhere all at once 2022" — explicit year is \
installment disambiguation, not a qualifier
SPECIFIC_TITLE flow — does NOT fire:
- "inception but funnier" — comparison modifier
- "parasite for kids" — audience qualifier
- "oppenheimer on netflix" — streaming qualifier
- "a shorter version of oppenheimer" — runtime qualifier

SIMILARITY_TO_TITLES flow — fires:
- "movies like inception" — frame + bare reference
- "similar to the matrix" — frame phrasing
- "in the vein of goodfellas" — frame phrasing
- "parasite meets get out" — multi-reference similarity frame
- "la la land, whiplash, and birdman" — bare list of films (implicit \
"like these")
SIMILARITY_TO_TITLES flow — does NOT fire:
- "movies like inception but shorter" — runtime modifier on the frame
- "similar to the matrix with more comedy" — genre modifier on the \
frame
- "in the vein of arrival but on netflix" — streaming modifier
- "movies like la la land that won oscars" — awards qualifier

---

"""


# ---------------------------------------------------------------------------
# Variant assembly
# ---------------------------------------------------------------------------

V1_BASELINE = (
    _TASK_AND_OUTCOME
    + _ZONE_STRUCTURE
    + _COVERAGE_PRINCIPLE
    + _RESOLUTION_PRINCIPLE
    + _QUALIFIER_RULE
    + _AMBIGUITY_PRINCIPLE
    + _SIMILARITY_FRAME_RULE
    + _OUTPUT_FIELD_GUIDANCE
)

V2_SOFTENED = (
    _TASK_AND_OUTCOME
    + _ZONE_STRUCTURE
    + _COVERAGE_PRINCIPLE_SOFTENED
    + _RESOLUTION_PRINCIPLE
    + _QUALIFIER_RULE
    + _AMBIGUITY_PRINCIPLE
    + _SIMILARITY_FRAME_RULE
    + _OUTPUT_FIELD_GUIDANCE
)

V3_FEW_SHOT = (
    _TASK_AND_OUTCOME
    + _ZONE_STRUCTURE
    + _COVERAGE_PRINCIPLE_SOFTENED
    + _RESOLUTION_PRINCIPLE
    + _QUALIFIER_RULE
    + _AMBIGUITY_PRINCIPLE
    + _SIMILARITY_FRAME_RULE
    + _OUTPUT_FIELD_GUIDANCE
    + _FEW_SHOT_EXAMPLES
)


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

@dataclass
class TestCase:
    query: str
    expected_flow: EntityFlow
    expected_fire_standard: bool
    note: str = ""


# Sixteen queries — no overlap with any movie / person / studio /
# franchise mentioned in the few-shot examples above. About half of
# them deliberately use phrasing patterns not present in the few-shot
# (possessive, question form, slang, novel speech-act verbs) so we are
# not just memorization-testing.
TEST_CASES: list[TestCase] = [
    # ---- person ----
    TestCase("leonardo dicaprio movies", EntityFlow.PERSON, False,
             "result-type packaging — same shape as few-shot"),
    TestCase("frances mcdormand?", EntityFlow.PERSON, False,
             "bare name + question mark — new format"),
    TestCase("wes anderson", EntityFlow.PERSON, False,
             "bare name"),
    TestCase("ryan gosling and emma stone", EntityFlow.PERSON, False,
             "list-shape"),
    TestCase("tarantino's films", EntityFlow.PERSON, False,
             "possessive packaging — new format"),

    # ---- studio ----
    TestCase("got any warner bros recs?", EntityFlow.STUDIO, False,
             "slang + question form — new format"),
    TestCase("queue up some focus features", EntityFlow.STUDIO, False,
             "novel imperative speech-act verb"),

    # ---- character franchise ----
    TestCase("shrek movies", EntityFlow.CHARACTER_FRANCHISE, False,
             "result-type packaging"),
    TestCase("looking for captain america", EntityFlow.CHARACTER_FRANCHISE, False,
             "novel speech-act phrase"),

    # ---- non-character franchise ----
    TestCase("the conjuring movies", EntityFlow.NON_CHARACTER_FRANCHISE, False,
             "result-type packaging"),
    TestCase("anything from the twilight series", EntityFlow.NON_CHARACTER_FRANCHISE, False,
             "conversational packaging + result-type"),

    # ---- specific title ----
    TestCase("watch interstellar", EntityFlow.SPECIFIC_TITLE, False,
             "speech-act packaging"),

    # ---- similarity ----
    TestCase("movies like the social network", EntityFlow.SIMILARITY_TO_TITLES, False,
             "explicit similarity frame"),
    TestCase("schindler's list and saving private ryan", EntityFlow.SIMILARITY_TO_TITLES, False,
             "bare list of two films"),

    # ---- ambiguous (entity + standard co-fire) ----
    TestCase("the holiday", EntityFlow.SPECIFIC_TITLE, True,
             "ambiguous: 2006 film vs holiday-themed mood reading"),
    TestCase("scream", EntityFlow.CHARACTER_FRANCHISE, True,
             "ambiguous: Ghostface franchise vs scary-mood reading"),
]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

async def _run_one(query: str, system_prompt: str) -> Step0Response | str:
    """Single Step 0 call with a custom system prompt. Returns either
    the parsed Step0Response or an error string so per-query failures
    don't abort the whole sweep."""
    user_prompt = f"Query: {query}"
    try:
        response, _in_tok, _out_tok = await generate_llm_response_async(
            provider=_STEP_0_PROVIDER,
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            response_format=Step0Response,
            model=_STEP_0_MODEL,
            **_STEP_0_KWARGS,
        )
        return response
    except Exception as exc:  # noqa: BLE001 — surface per-call failures
        return f"ERROR: {exc!r}"


async def _run_variant(system_prompt: str) -> list[Step0Response | str]:
    """Fan all test queries through one prompt variant in parallel."""
    return await asyncio.gather(
        *(_run_one(tc.query, system_prompt) for tc in TEST_CASES)
    )


def _outcome(resp: Step0Response | str) -> str:
    """Compact one-line summary of a Step 0 outcome for the table."""
    if isinstance(resp, str):
        return resp
    if resp.selected_entity_flow == EntityFlow.NONE_OF_THE_ABOVE:
        return "none+std"
    names = ",".join(e.canonical_name for e in resp.selected_entities)
    base = f"{resp.selected_entity_flow.value}[{names}]"
    return base + ("+std" if resp.fire_standard_flow else "")


def _is_correct(tc: TestCase, resp: Step0Response | str) -> bool:
    if isinstance(resp, str):
        return False
    return (
        resp.selected_entity_flow == tc.expected_flow
        and resp.fire_standard_flow == tc.expected_fire_standard
    )


def _format_row(
    idx: int,
    tc: TestCase,
    r1: Step0Response | str,
    r2: Step0Response | str,
    r3: Step0Response | str,
) -> str:
    def cell(resp, ok):
        marker = "OK" if ok else "X "
        return f"{marker} {_outcome(resp)}"

    exp = tc.expected_flow.value + ("+std" if tc.expected_fire_standard else "")
    c1, c2, c3 = _is_correct(tc, r1), _is_correct(tc, r2), _is_correct(tc, r3)
    return (
        f"{idx:>2}  {tc.query:<42}  {exp:<32}  "
        f"{cell(r1, c1):<40}  {cell(r2, c2):<40}  {cell(r3, c3):<40}"
    )


def _format_reasoning(label: str, tc: TestCase, resp: Step0Response | str) -> list[str]:
    """Dump the model's own reasoning for any disagreement so we can
    diagnose miscategorizations without re-running."""
    if isinstance(resp, str):
        return [f"[{label}] {tc.query!r}: {resp}"]
    return [
        f"[{label}] {tc.query!r}: flow={resp.selected_entity_flow.value} "
        f"fire_std={resp.fire_standard_flow}",
        f"      flow_reasoning: {resp.selected_entity_flow_reasoning}",
        f"      ambig_reasoning: {resp.primary_intent_ambiguity_reasoning}",
        f"      qualifiers: {resp.qualifiers}",
    ]


async def _main_async() -> None:
    print("Running V1 baseline ...")
    r1 = await _run_variant(V1_BASELINE)
    print("Running V2 softened coverage ...")
    r2 = await _run_variant(V2_SOFTENED)
    print("Running V3 softened + few-shot ...")
    r3 = await _run_variant(V3_FEW_SHOT)

    header = (
        f"{'#':>2}  {'query':<42}  {'expected':<32}  "
        f"{'V1 baseline':<40}  {'V2 softened':<40}  {'V3 +few-shot':<40}"
    )
    sep = "-" * len(header)

    lines = [header, sep]
    v1_correct = v2_correct = v3_correct = 0
    for i, tc in enumerate(TEST_CASES, start=1):
        lines.append(_format_row(i, tc, r1[i - 1], r2[i - 1], r3[i - 1]))
        v1_correct += int(_is_correct(tc, r1[i - 1]))
        v2_correct += int(_is_correct(tc, r2[i - 1]))
        v3_correct += int(_is_correct(tc, r3[i - 1]))
    lines.append(sep)
    n = len(TEST_CASES)
    lines.append(
        f"totals: V1={v1_correct}/{n}  V2={v2_correct}/{n}  V3={v3_correct}/{n}"
    )

    # Per-variant reasoning dump for every disagreement — keeps the
    # diagnostic surface in one artifact alongside the table.
    lines.append("")
    lines.append("=== reasoning for disagreements ===")
    any_disagreement = False
    for i, tc in enumerate(TEST_CASES, start=1):
        for label, resp in [("V1", r1[i - 1]), ("V2", r2[i - 1]), ("V3", r3[i - 1])]:
            if not _is_correct(tc, resp):
                any_disagreement = True
                lines.extend(_format_reasoning(label, tc, resp))
    if not any_disagreement:
        lines.append("(none — every variant matched expected on every query)")

    output = "\n".join(lines)
    print()
    print(output)

    out_path = (
        Path(__file__).resolve().parent
        / "step_0_few_shot_experiment_results.txt"
    )
    out_path.write_text(output + "\n")
    print()
    print(f"wrote {out_path}")


def main() -> None:
    from implementation.misc.event_loop import install_uvloop

    install_uvloop()
    asyncio.run(_main_async())


if __name__ == "__main__":
    main()

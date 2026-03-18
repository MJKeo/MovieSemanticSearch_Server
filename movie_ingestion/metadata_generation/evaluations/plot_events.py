"""
Evaluation pipeline for plot_events metadata.

Implements a two-phase pointwise evaluation:

Phase 0 — generate_reference_responses():
    Generates one reference PlotEventsOutput per movie using Claude Opus via
    ANTHROPIC_OAUTH_KEY. References are fixed for the duration of the evaluation
    and stored in plot_events_references. Run once before any candidate evaluation.

Phase 1 — run_evaluation():
    For each (candidate, movie) pair: generates the candidate's PlotEventsOutput,
    retrieves the reference, calls a Claude judge with the full rubric and both
    outputs, and stores per-dimension scores and reasoning.

Visualization — print_score_summary():
    Queries plot_events_evaluations and prints a formatted mean/median table
    per candidate per dimension.

Storage (evaluation_data/eval.db):
    - plot_events_references: (movie_id) → reference output
    - plot_events_candidate_outputs: (movie_id, candidate_id) → candidate output
    - plot_events_evaluations: (movie_id, candidate_id) → 4-dimension scores + reasoning

Both phases are idempotent — re-running skips rows that already exist.
"""

import asyncio
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import pandas as pd
from pydantic import BaseModel

from implementation.llms.generic_methods import LLMProvider, generate_llm_response_async
from movie_ingestion.metadata_generation.evaluations.shared import (
    EVAL_DB_PATH,
    EvaluationCandidate,
    compute_score_summary,
    create_candidates_table,
    get_eval_connection,
    store_candidate,
)
from movie_ingestion.metadata_generation.generators.plot_events import (
    build_plot_events_user_prompt,
)
from movie_ingestion.metadata_generation.inputs import MovieInputData
from movie_ingestion.metadata_generation.schemas import MajorCharacter, PlotEventsOutput
from movie_ingestion.metadata_generation.prompts.plot_events import (
    SYSTEM_PROMPT as DEFAULT_SYSTEM_PROMPT,
)

# ---------------------------------------------------------------------------
# Candidates — LLM configurations to evaluate for plot_events
# ---------------------------------------------------------------------------

PLOT_EVENTS_CANDIDATES: list[EvaluationCandidate] = [
    # -----------------------------------------------------------------------
    # Qwen 3.5 Flash — 2 candidates (thinking toggle)
    # -----------------------------------------------------------------------
    EvaluationCandidate(
        candidate_id="plot_events__qwen3.5-flash",
        provider=LLMProvider.ALIBABA,
        model="qwen3.5-flash",
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        response_format=PlotEventsOutput,
        kwargs={"temperature": 0.0, "extra_body": {"enable_thinking": False}},
    ),
    EvaluationCandidate(
        candidate_id="plot_events__qwen3.5-flash__think",
        provider=LLMProvider.ALIBABA,
        model="qwen3.5-flash",
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        response_format=PlotEventsOutput,
        kwargs={"temperature": 0.2, "extra_body": {"enable_thinking": True}},
    ),
    # -----------------------------------------------------------------------
    # Gemini 2.5 Flash — 3 candidates (thinking budget curve)
    # -----------------------------------------------------------------------
    EvaluationCandidate(
        candidate_id="plot_events__gemini-2.5-flash",
        provider=LLMProvider.GEMINI,
        model="gemini-2.5-flash",
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        response_format=PlotEventsOutput,
        kwargs={"temperature": 0.2, "thinking_config": {"thinking_budget": 0}},
    ),
    EvaluationCandidate(
        candidate_id="plot_events__gemini-2.5-flash__think-1k",
        provider=LLMProvider.GEMINI,
        model="gemini-2.5-flash",
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        response_format=PlotEventsOutput,
        kwargs={"temperature": 0.2, "thinking_config": {"thinking_budget": 1024}},
    ),
    EvaluationCandidate(
        candidate_id="plot_events__gemini-2.5-flash__think-4k",
        provider=LLMProvider.GEMINI,
        model="gemini-2.5-flash",
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        response_format=PlotEventsOutput,
        kwargs={"temperature": 0.2, "thinking_config": {"thinking_budget": 4096}},
    ),
    # -----------------------------------------------------------------------
    # Gemini 2.5 Flash Lite — 2 candidates (thinking on/off)
    # -----------------------------------------------------------------------
    EvaluationCandidate(
        candidate_id="plot_events__gemini-2.5-flash-lite",
        provider=LLMProvider.GEMINI,
        model="gemini-2.5-flash-lite",
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        response_format=PlotEventsOutput,
        kwargs={"temperature": 0.2, "thinking_config": {"thinking_budget": 0}},
    ),
    EvaluationCandidate(
        candidate_id="plot_events__gemini-2.5-flash-lite__think-1k",
        provider=LLMProvider.GEMINI,
        model="gemini-2.5-flash-lite",
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        response_format=PlotEventsOutput,
        kwargs={"temperature": 0.2, "thinking_config": {"thinking_budget": 1024}},
    ),
    # -----------------------------------------------------------------------
    # GPT-5-mini — 3 candidates (reasoning_effort x verbosity)
    # -----------------------------------------------------------------------
    EvaluationCandidate(
        candidate_id="plot_events__gpt-5-mini",
        provider=LLMProvider.OPENAI,
        model="gpt-5-mini",
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        response_format=PlotEventsOutput,
        kwargs={"reasoning_effort": "minimal", "verbosity": "low"},
    ),
    EvaluationCandidate(
        candidate_id="plot_events__gpt-5-mini__reason-low",
        provider=LLMProvider.OPENAI,
        model="gpt-5-mini",
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        response_format=PlotEventsOutput,
        kwargs={"reasoning_effort": "low", "verbosity": "low"},
    ),
    EvaluationCandidate(
        candidate_id="plot_events__gpt-5-mini__reason-low-verb-med",
        provider=LLMProvider.OPENAI,
        model="gpt-5-mini",
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        response_format=PlotEventsOutput,
        kwargs={"reasoning_effort": "low", "verbosity": "medium"},
    ),
    # -----------------------------------------------------------------------
    # GPT-5-nano — 2 candidates (reasoning_effort)
    # -----------------------------------------------------------------------
    EvaluationCandidate(
        candidate_id="plot_events__gpt-5-nano",
        provider=LLMProvider.OPENAI,
        model="gpt-5-nano",
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        response_format=PlotEventsOutput,
        kwargs={"reasoning_effort": "minimal", "verbosity": "low"},
    ),
    EvaluationCandidate(
        candidate_id="plot_events__gpt-5-nano__reason-low",
        provider=LLMProvider.OPENAI,
        model="gpt-5-nano",
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        response_format=PlotEventsOutput,
        kwargs={"reasoning_effort": "low", "verbosity": "low"},
    ),
    # -----------------------------------------------------------------------
    # GPT-5.4-nano — 3 candidates (reasoning_effort x verbosity)
    # -----------------------------------------------------------------------
    EvaluationCandidate(
        candidate_id="plot_events__gpt-5.4-nano",
        provider=LLMProvider.OPENAI,
        model="gpt-5.4-nano",
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        response_format=PlotEventsOutput,
        kwargs={"reasoning_effort": "minimal", "verbosity": "low"},
    ),
    EvaluationCandidate(
        candidate_id="plot_events__gpt-5.4-nano__reason-low",
        provider=LLMProvider.OPENAI,
        model="gpt-5.4-nano",
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        response_format=PlotEventsOutput,
        kwargs={"reasoning_effort": "low", "verbosity": "low"},
    ),
    EvaluationCandidate(
        candidate_id="plot_events__gpt-5.4-nano__reason-low-verb-med",
        provider=LLMProvider.OPENAI,
        model="gpt-5.4-nano",
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        response_format=PlotEventsOutput,
        kwargs={"reasoning_effort": "low", "verbosity": "medium"},
    ),
    # -----------------------------------------------------------------------
    # GPT-oss-120b — 2 candidates (reasoning_effort)
    # -----------------------------------------------------------------------
    EvaluationCandidate(
        candidate_id="plot_events__gpt-oss-120b",
        provider=LLMProvider.GROQ,
        model="openai/gpt-oss-120b",
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        response_format=PlotEventsOutput,
        kwargs={"temperature": 0.2, "reasoning_effort": "low", "reasoning_format": "hidden"},
    ),
    EvaluationCandidate(
        candidate_id="plot_events__gpt-oss-120b__reason-med",
        provider=LLMProvider.GROQ,
        model="openai/gpt-oss-120b",
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        response_format=PlotEventsOutput,
        kwargs={"temperature": 0.2, "reasoning_effort": "medium", "reasoning_format": "hidden"},
    ),
    # -----------------------------------------------------------------------
    # Llama 4 Scout — 2 candidates (temperature)
    # -----------------------------------------------------------------------
    EvaluationCandidate(
        candidate_id="plot_events__llama-4-scout",
        provider=LLMProvider.GROQ,
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        response_format=PlotEventsOutput,
        kwargs={"temperature": 0.2},
    ),
    EvaluationCandidate(
        candidate_id="plot_events__llama-4-scout__temp-0",
        provider=LLMProvider.GROQ,
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        response_format=PlotEventsOutput,
        kwargs={"temperature": 0.0},
    ),
]

# ---------------------------------------------------------------------------
# SQLite table DDL
# ---------------------------------------------------------------------------

_CREATE_REFERENCES_TABLE = """
    CREATE TABLE IF NOT EXISTS plot_events_references (
        movie_id         INTEGER PRIMARY KEY,
        plot_summary     TEXT NOT NULL,
        setting          TEXT NOT NULL,
        major_characters TEXT NOT NULL,   -- JSON array of MajorCharacter dicts
        reference_model  TEXT NOT NULL,
        input_tokens     INTEGER,
        output_tokens    INTEGER,
        created_at       TEXT NOT NULL
    )
"""

_CREATE_CANDIDATE_OUTPUTS_TABLE = """
    CREATE TABLE IF NOT EXISTS plot_events_candidate_outputs (
        movie_id         INTEGER NOT NULL,
        candidate_id     TEXT NOT NULL,
        plot_summary     TEXT NOT NULL,
        setting          TEXT NOT NULL,
        major_characters TEXT NOT NULL,   -- JSON array of MajorCharacter dicts
        input_tokens     INTEGER,
        output_tokens    INTEGER,
        created_at       TEXT NOT NULL,
        PRIMARY KEY (movie_id, candidate_id)
    )
"""

_CREATE_EVALUATIONS_TABLE = """
    CREATE TABLE IF NOT EXISTS plot_events_evaluations (
        movie_id                       INTEGER NOT NULL,
        candidate_id                   TEXT NOT NULL,
        groundedness_score             INTEGER,
        groundedness_reasoning         TEXT,
        plot_summary_score             INTEGER,
        plot_summary_reasoning         TEXT,
        character_quality_score        INTEGER,
        character_quality_reasoning    TEXT,
        setting_score                  INTEGER,
        setting_reasoning              TEXT,
        judge_model                    TEXT,
        judge_input_tokens             INTEGER,
        judge_output_tokens            INTEGER,
        created_at                     TEXT NOT NULL,
        PRIMARY KEY (movie_id, candidate_id)
    )
"""

# Score columns used by compute_score_summary
SCORE_COLUMNS = [
    "groundedness_score",
    "plot_summary_score",
    "character_quality_score",
    "setting_score",
]


def create_plot_events_tables(conn: sqlite3.Connection) -> None:
    """Create all three plot_events evaluation tables if they don't exist."""
    conn.execute(_CREATE_REFERENCES_TABLE)
    conn.execute(_CREATE_CANDIDATE_OUTPUTS_TABLE)
    conn.execute(_CREATE_EVALUATIONS_TABLE)
    conn.commit()


# ---------------------------------------------------------------------------
# Judge output schema
# ---------------------------------------------------------------------------

class PlotEventsJudgeOutput(BaseModel):
    """Structured output from the Claude judge for plot_events evaluation.

    Reasoning fields come before score fields — this ordering is reflected
    in the JSON schema passed to the judge, reinforcing the spec requirement
    that explicit chain-of-thought must precede scores.

    Scores use Literal[1, 2, 3, 4] to constrain the 4-point scale and
    prevent the judge from returning out-of-range values.
    """
    # Reasoning before scores (spec requirement: explicit CoT before scores)
    groundedness_reasoning: str
    plot_summary_reasoning: str
    character_quality_reasoning: str
    setting_reasoning: str
    # Scores — 4-point scale per dimension
    groundedness_score: Literal[1, 2, 3, 4]
    plot_summary_score: Literal[1, 2, 3, 4]
    character_quality_score: Literal[1, 2, 3, 4]
    setting_score: Literal[1, 2, 3, 4]


# ---------------------------------------------------------------------------
# Judge system prompt (rubric)
# ---------------------------------------------------------------------------

JUDGE_SYSTEM_PROMPT = """You are an expert evaluator of movie metadata quality. Score LLM-generated plot_events metadata on a 4-point scale across 4 independent dimensions.

CONTEXT: plot_events metadata is a structured representation of WHAT HAPPENS in a movie — concrete events, essential characters, and setting. It is about events and facts, not themes or analysis. The output serves two purposes:
1. Vector embeddings for semantic search — preserving character names, location names, and concrete plot actions enables specific queries to match
2. Primary input to downstream metadata generators that analyze themes, character arcs, and viewer experience

THE GENERATION PROMPT instructs the model to:
- Extract a HIGH-SIGNAL, SPOILER-CONTAINING representation of what happens
- Preserve specificity: character NAMES, location names, concrete plot actions
- Include ONLY essential characters and ONLY the 1-3 core conflicts
- Use compact wording over flowery prose; avoid filler
- Avoid generic "theme talk" and abstract moralizing
- Only describe what is evident from the provided data; produce a shorter output rather than inventing details
- Use plot_synopsis and detailed plot_summaries as primary truth; if sources conflict, prefer the most detailed, internally consistent version

SCORE SCALE:
4 (Excellent) — Fully meets the dimension's intent. All expected content present, accurate, and well-specified.
3 (Adequate) — Mostly meets the intent. Minor gaps or imprecision that don't materially reduce usefulness.
2 (Partial) — Meets some requirements but has meaningful gaps or issues that reduce usefulness.
1 (Failing) — Does not meet the intent. Missing, degenerate, or fundamentally flawed.

---

DIMENSION 1: groundedness
Evaluates factual accuracy across ALL output fields (plot_summary, setting, major_characters). Every detail must be traceable to the provided input data. This is the most important dimension — hallucinated content propagates into downstream metadata and creates false search matches.

Source hierarchy for adjudicating conflicts: plot_synopsis and detailed plot_summaries are the primary truth. If sources conflict, the most detailed, internally consistent version takes precedence. Overview is a marketing summary and is the weakest source.

Score 4: All details across all fields directly supported by the input data. No fabricated characters, events, locations, or relationships.
Score 3: All major claims supported. At most 1-2 minor details that are reasonable inferences from the inputs rather than directly stated (e.g., inferring a relationship dynamic that's strongly implied but not explicit).
Score 2: 1-2 details clearly absent from all input fields. Not egregious fabrication, but clearly unsupported claims.
Score 1: Any clearly fabricated plot event, character name, character relationship, or setting detail. OR multiple unsupported details across fields.

---

DIMENSION 2: plot_summary
Evaluates whether the summary provides chronological coverage of concrete events with enough specific detail for the output to be useful as a search embedding and as input to downstream generators.

The generation prompt instructs: chronological summary of the entire film, preserving character names, location names, key organizations, and important events. Compact wording, no filler, no abstract moralizing or theme talk. Only the 1-3 core conflicts that define the movie.

Score 4: Chronological event coverage from beginning to end, focused on the 1-3 core conflicts. Character names, locations, and concrete plot actions preserved throughout. Compact, plot-grounded wording with no filler, padding, or thematic commentary. When inputs provide specific detail, the summary reflects that specificity.
Score 3: Event coverage substantially complete but the ending is thin, or 1-2 concrete details available in the inputs are generalized. Mostly compact — at most minor instances of vague phrasing where inputs gave specific detail. No significant theme talk or moralizing.
Score 2: Significant events missing (ending omitted, or a major section collapsed to one sentence). OR accurate but so high-level that it reads as a premise description rather than a plot recount. OR contains noticeable filler, flowery prose, or abstract thematic commentary.
Score 1: Major plot events missing. OR so brief/generic as to be minimally useful. OR dominated by thematic analysis rather than concrete events.

When inputs are sparse (only overview + keywords, no synopsis or summaries): do not penalize a shorter summary. Penalize padding or speculation instead. A concise, grounded summary from sparse inputs can score 4.

---

DIMENSION 3: character_quality
Evaluates both WHO is included (selection) and HOW they're described (accuracy and specificity), scored as one dimension across all listed characters.

The generation prompt instructs: include ONLY the absolutely essential characters needed to understand the plot (only a few). Do NOT list minor side characters. For each: exact name, short plot-relevant description, narrative role label, and 1 short sentence stating primary motivations (high-level).

Score 4: Only the characters essential to understanding the core plot are included — no peripheral or minor characters. Count is small and proportionate to the film's complexity. Each character has: correct name, short plot-relevant description, appropriate role label, and a concise single-sentence motivation.
Score 3: All essential characters present. May include 1 borderline-peripheral character, or 1-2 characters have slightly vague descriptions, marginally imprecise roles, or motivations that are correct but not concise (multi-sentence or overly granular). All names correct.
Score 2: Misses a clear protagonist, antagonist, or major plot driver. OR multiple characters have vague/inaccurate descriptions or motivations. OR includes several clearly minor characters, inflating the list beyond what's needed to understand the plot.
Score 1: Essential characters not identified, entirely wrong character set, OR majority of descriptions are inaccurate or too vague to be useful.

---

DIMENSION 4: setting
Evaluates the accuracy and specificity of the WHERE/WHEN setting phrase.

The generation prompt instructs: 10 words or less. Details that are unknown are omitted — never make up details. Preserve meaningful proper nouns and time period when relevant.

Score 4: Includes specific named location AND time period where inputs provide them. Omits dimensions the inputs don't support (this is correct behavior, not a gap). Accurately characterizes the environment. 10 words or fewer. All details grounded in inputs.
Score 3: Specific but omits one dimension (where OR when) when inputs clearly provided both. OR slightly generic where inputs had more detail. OR marginally over 10 words.
Score 2: Generic to the point of being minimally useful (e.g., "modern urban setting" when inputs provided a specific city). OR entirely omits a dimension when inputs clearly provided it.
Score 1: Contradicts inputs, hallucinated location/time, or so generic as to be content-free.

---

SCORING INSTRUCTIONS:
1. For each dimension, write reasoning FIRST, then state the score.
2. Score each dimension independently — a factual error penalized in groundedness should not also lower plot_summary or character_quality scores.
3. Evaluate semantic content, not surface form. Two outputs expressing the same meaning differently receive the same score.
4. The reference response is a calibration anchor showing what quality looks like — it is NOT ground truth. Do not penalize for different but defensible choices (e.g., selecting different characters, different granularity).
5. For verifiable facts (character names, plot events), use the inputs as the authority. For subjective elements (how much detail, which characters are "essential"), score based on defensibility.
6. Only penalize in groundedness if a detail is absent from ALL fields in the generation prompt.
7. Filler, flowery prose, abstract moralizing, and thematic commentary should be penalized in plot_summary, not groundedness (they are style violations, not factual errors)."""


# ---------------------------------------------------------------------------
# Helper: serialize / deserialize PlotEventsOutput for storage
# ---------------------------------------------------------------------------

def _serialize_output(output: PlotEventsOutput) -> tuple[str, str, str]:
    """Return (plot_summary, setting, major_characters_json) for DB storage."""
    chars_json = json.dumps([c.model_dump() for c in output.major_characters])
    return output.plot_summary, output.setting, chars_json


def _deserialize_output(row: sqlite3.Row) -> PlotEventsOutput:
    """Reconstruct a PlotEventsOutput from a DB row."""
    chars = [MajorCharacter.model_validate(c) for c in json.loads(row["major_characters"])]
    return PlotEventsOutput(
        plot_summary=row["plot_summary"],
        setting=row["setting"],
        major_characters=chars,
    )


def _format_characters_for_prompt(output: PlotEventsOutput) -> str:
    """Format the character list as readable text for the judge user prompt."""
    lines = []
    for c in output.major_characters:
        lines.append(
            f"  - {c.name} ({c.role}): {c.description} "
            f"Motivations: {c.primary_motivations}"
        )
    return "\n".join(lines) if lines else "  (none)"


def _build_judge_user_prompt(
    generation_user_prompt: str,
    reference: PlotEventsOutput,
    candidate_output: PlotEventsOutput,
) -> str:
    """Assemble the judge's user prompt from generation context + both outputs."""
    return f"""GENERATION PROMPT:
{generation_user_prompt}

---

REFERENCE RESPONSE:
plot_summary: {reference.plot_summary}
setting: {reference.setting}
major_characters:
{_format_characters_for_prompt(reference)}

---

CANDIDATE OUTPUT:
plot_summary: {candidate_output.plot_summary}
setting: {candidate_output.setting}
major_characters:
{_format_characters_for_prompt(candidate_output)}"""


# ---------------------------------------------------------------------------
# Phase 0: generate reference responses
# ---------------------------------------------------------------------------

async def generate_reference_responses(
    movie_inputs: dict[int, MovieInputData],
    reference_model: str = "claude-opus-4-6",
    db_path: Path | None = None,
) -> None:
    """Generate and store reference PlotEventsOutputs for all movies.

    Uses Claude Opus (via ANTHROPIC_OAUTH_KEY) and the same system prompt
    and user prompt construction as the production generator, so the reference
    represents what a stronger model would produce under identical conditions.

    Requests run in series to avoid rate limiting. If a 429 error is
    encountered, pauses for 60 seconds and retries the request once.

    Idempotent: movies with existing references are skipped.

    Args:
        movie_inputs: Dict of tmdb_id → MovieInputData for the test corpus.
        reference_model: Claude model identifier to use as the reference.
        db_path: Override the default eval DB path (useful for testing).
    """
    conn = get_eval_connection(db_path or EVAL_DB_PATH)
    create_candidates_table(conn)
    create_plot_events_tables(conn)

    # Determine which movies still need reference responses
    existing = {
        row["movie_id"]
        for row in conn.execute("SELECT movie_id FROM plot_events_references").fetchall()
    }
    pending = [
        (tmdb_id, movie)
        for tmdb_id, movie in movie_inputs.items()
        if tmdb_id not in existing
    ]

    if not pending:
        print("Phase 0: all reference responses already exist — nothing to do.")
        conn.close()
        return

    print(f"Phase 0: generating {len(pending)} reference responses "
          f"(skipping {len(existing)} existing) using {reference_model}...")

    completed = 0
    total = len(pending)

    # Run requests in series to avoid rate limiting
    for tmdb_id, movie in pending:
        try:
            user_prompt = build_plot_events_user_prompt(movie)

            # Attempt the LLM call; on a 429 rate-limit error, pause 60s and retry once
            try:
                parsed, in_tokens, out_tokens = await generate_llm_response_async(
                    provider=LLMProvider.ANTHROPIC,
                    user_prompt=user_prompt,
                    system_prompt=DEFAULT_SYSTEM_PROMPT,
                    response_format=PlotEventsOutput,
                    model=reference_model,
                    max_tokens=4096,
                )
            except ValueError as e:
                if "429" not in str(e):
                    raise
                print(f"  [429] Rate limited on tmdb_id={tmdb_id} — "
                      f"pausing 60s before retry...")
                await asyncio.sleep(60)
                parsed, in_tokens, out_tokens = await generate_llm_response_async(
                    provider=LLMProvider.ANTHROPIC,
                    user_prompt=user_prompt,
                    system_prompt=DEFAULT_SYSTEM_PROMPT,
                    response_format=PlotEventsOutput,
                    model=reference_model,
                    max_tokens=4096,
                )

            plot_summary, setting, chars_json = _serialize_output(parsed)
            conn.execute(
                """
                INSERT OR IGNORE INTO plot_events_references
                    (movie_id, plot_summary, setting, major_characters,
                     reference_model, input_tokens, output_tokens, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    tmdb_id, plot_summary, setting, chars_json,
                    reference_model, in_tokens, out_tokens,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            conn.commit()
            completed += 1
            print(f"  [{completed}/{total}] Reference generated: {movie.title_with_year()}")
        except Exception as e:
            print(f"  [ERROR] Reference generation failed for tmdb_id={tmdb_id} "
                  f"({movie.title_with_year()}): {e}")

    print(f"Phase 0 complete: {completed}/{total} references generated.")
    conn.close()


# ---------------------------------------------------------------------------
# Phase 1: generate candidate outputs and run evaluation
# ---------------------------------------------------------------------------

async def run_evaluation(
    candidates: list[EvaluationCandidate],
    movie_inputs: dict[int, MovieInputData],
    judge_model: str = "claude-sonnet-4-6",
    concurrency: int = 5,
    db_path: Path | None = None,
) -> None:
    """Generate candidate outputs and score them with a Claude judge.

    For each (candidate, movie) pair:
    1. Skip if an evaluation result already exists (idempotent).
    2. Generate the candidate output if not already stored.
    3. Retrieve the reference response (Phase 0 must have run first).
    4. Reconstruct the original generation user prompt.
    5. Call the Claude judge with rubric + generation prompt + reference + candidate.
    6. Store per-dimension scores and reasoning.

    Args:
        candidates: List of candidate configurations to evaluate.
        movie_inputs: Dict of tmdb_id → MovieInputData for the test corpus.
        judge_model: Claude model to use as the evaluator judge.
        concurrency: Max concurrent in-flight requests (generation + judge combined).
        db_path: Override the default eval DB path (useful for testing).
    """
    conn = get_eval_connection(db_path or EVAL_DB_PATH)
    create_candidates_table(conn)
    create_plot_events_tables(conn)

    # Register all candidates in the DB
    for candidate in candidates:
        store_candidate(conn, candidate, "plot_events")

    semaphore = asyncio.Semaphore(concurrency)
    total = len(candidates) * len(movie_inputs)
    completed = 0

    async def _evaluate_one(
        candidate: EvaluationCandidate,
        tmdb_id: int,
        movie: MovieInputData,
    ) -> None:
        nonlocal completed
        async with semaphore:
            # Skip if already evaluated
            existing_eval = conn.execute(
                "SELECT 1 FROM plot_events_evaluations "
                "WHERE movie_id = ? AND candidate_id = ?",
                (tmdb_id, candidate.candidate_id),
            ).fetchone()
            if existing_eval:
                completed += 1
                return

            # Check if Phase 0 reference exists — required before evaluation
            ref_row = conn.execute(
                "SELECT * FROM plot_events_references WHERE movie_id = ?",
                (tmdb_id,),
            ).fetchone()
            if ref_row is None:
                print(
                    f"  [ERROR] No reference for tmdb_id={tmdb_id}. "
                    "Run generate_reference_responses() first (Phase 0)."
                )
                return

            reference = _deserialize_output(ref_row)

            # Retrieve or generate candidate output
            output_row = conn.execute(
                "SELECT * FROM plot_events_candidate_outputs "
                "WHERE movie_id = ? AND candidate_id = ?",
                (tmdb_id, candidate.candidate_id),
            ).fetchone()

            # Reconstruct the generation user prompt — both generation and judge
            # need this, so build it once regardless of whether output exists.
            generation_user_prompt = build_plot_events_user_prompt(movie)

            if output_row is not None:
                candidate_output = _deserialize_output(output_row)
            else:
                try:
                    # Call the LLM directly with the candidate's system prompt.
                    # This is the critical difference from calling generate_plot_events():
                    # each candidate may have a distinct system_prompt, and we must
                    # honour it rather than falling back to the module-level default.
                    parsed, gen_in_tokens, gen_out_tokens = await generate_llm_response_async(
                        provider=candidate.provider,
                        user_prompt=generation_user_prompt,
                        system_prompt=candidate.system_prompt,
                        response_format=candidate.response_format,
                        model=candidate.model,
                        **candidate.kwargs,
                    )
                    candidate_output = parsed
                    gen_input_tokens = gen_in_tokens
                    gen_output_tokens = gen_out_tokens
                except Exception as e:
                    print(
                        f"  [ERROR] Candidate generation failed: "
                        f"candidate={candidate.candidate_id}, "
                        f"tmdb_id={tmdb_id}: {e}"
                    )
                    return

                plot_summary, setting, chars_json = _serialize_output(candidate_output)
                conn.execute(
                    """
                    INSERT OR IGNORE INTO plot_events_candidate_outputs
                        (movie_id, candidate_id, plot_summary, setting,
                         major_characters, input_tokens, output_tokens, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        tmdb_id, candidate.candidate_id,
                        plot_summary, setting, chars_json,
                        gen_input_tokens, gen_output_tokens,
                        datetime.now(timezone.utc).isoformat(),
                    ),
                )
                conn.commit()

            judge_user_prompt = _build_judge_user_prompt(
                generation_user_prompt, reference, candidate_output
            )

            # Call the judge
            try:
                judge_result, judge_in_tokens, judge_out_tokens = (
                    await generate_llm_response_async(
                        provider=LLMProvider.ANTHROPIC,
                        user_prompt=judge_user_prompt,
                        system_prompt=JUDGE_SYSTEM_PROMPT,
                        response_format=PlotEventsJudgeOutput,
                        model=judge_model,
                        max_tokens=4096,
                        temperature=0.2,
                    )
                )
            except Exception as e:
                print(
                    f"  [ERROR] Judge call failed: "
                    f"candidate={candidate.candidate_id}, "
                    f"tmdb_id={tmdb_id}: {e}"
                )
                return

            conn.execute(
                """
                INSERT OR IGNORE INTO plot_events_evaluations (
                    movie_id, candidate_id,
                    groundedness_score, groundedness_reasoning,
                    plot_summary_score, plot_summary_reasoning,
                    character_quality_score, character_quality_reasoning,
                    setting_score, setting_reasoning,
                    judge_model, judge_input_tokens, judge_output_tokens,
                    created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    tmdb_id, candidate.candidate_id,
                    judge_result.groundedness_score,
                    judge_result.groundedness_reasoning,
                    judge_result.plot_summary_score,
                    judge_result.plot_summary_reasoning,
                    judge_result.character_quality_score,
                    judge_result.character_quality_reasoning,
                    judge_result.setting_score,
                    judge_result.setting_reasoning,
                    judge_model,
                    judge_in_tokens,
                    judge_out_tokens,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            conn.commit()
            completed += 1
            print(
                f"  [{completed}/{total}] Evaluated: "
                f"candidate={candidate.candidate_id}, "
                f"movie={movie.title_with_year()} | "
                f"scores: ground={judge_result.groundedness_score} "
                f"summary={judge_result.plot_summary_score} "
                f"char={judge_result.character_quality_score} "
                f"setting={judge_result.setting_score}"
            )

    tasks = [
        _evaluate_one(candidate, tmdb_id, movie)
        for candidate in candidates
        for tmdb_id, movie in movie_inputs.items()
    ]
    await asyncio.gather(*tasks)

    print(f"Phase 1 complete: {completed}/{total} evaluations done.")
    conn.close()


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def print_score_summary(
    candidate_ids: list[str] | None = None,
    db_path: Path | None = None,
) -> pd.DataFrame | None:
    """Print a mean/median score table per candidate per evaluation dimension.

    Queries plot_events_evaluations and displays a formatted table. Also
    returns the underlying DataFrame for further programmatic use.

    Args:
        candidate_ids: If provided, filter to only these candidates.
        db_path: Override the default eval DB path (useful for testing).

    Returns:
        The summary DataFrame, or None if no results exist.
    """
    conn = get_eval_connection(db_path or EVAL_DB_PATH)

    # Verify the table exists
    table_exists = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' "
        "AND name='plot_events_evaluations'"
    ).fetchone()
    if not table_exists:
        print("No evaluation results found — run run_evaluation() first.")
        conn.close()
        return

    summary = compute_score_summary(
        conn=conn,
        table="plot_events_evaluations",
        score_columns=SCORE_COLUMNS,
        candidate_ids=candidate_ids,
    )
    conn.close()

    if summary.empty:
        print("No evaluation results found for the requested candidates.")
        return

    # Print formatted table
    # Column header width: candidate_id may be long
    cid_width = max(len("candidate_id"), summary.index.str.len().max())
    dim_width = 14  # wide enough for "mean" / "median" values

    dims = ["groundedness", "plot_summary", "character_quality", "setting"]
    short_labels = ["grounded", "plot_summ", "char_qual", "setting"]

    # Header row
    header = f"{'candidate_id':<{cid_width}}"
    for label in short_labels:
        header += f"  {label + '_mean':>{dim_width}}  {label + '_med':>{dim_width}}"
    header += f"  {'overall_mean':>{dim_width}}"
    print("\n" + "=" * len(header))
    print("plot_events evaluation scores (4-point scale per dimension)")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for cid, row in summary.iterrows():
        line = f"{cid:<{cid_width}}"
        for dim in dims:
            mean_val = row.get(f"{dim}_mean", float("nan"))
            med_val = row.get(f"{dim}_median", float("nan"))
            line += f"  {mean_val:>{dim_width}.2f}  {med_val:>{dim_width}.2f}"
        overall = row.get("overall_mean", float("nan"))
        line += f"  {overall:>{dim_width}.2f}"
        print(line)

    print("=" * len(header) + "\n")

    return summary

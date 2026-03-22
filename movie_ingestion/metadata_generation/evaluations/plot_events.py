"""
Evaluation pipeline for plot_events metadata.

Reference-free pointwise evaluation: for each (candidate, movie) pair,
generates the candidate's PlotEventsOutput, then scores it against a
detailed rubric using an LLM judge (Claude Opus 4.6). The judge sees
the raw source data (not the generation prompt) alongside the candidate
output and scores 4 dimensions: groundedness, plot_summary,
character_quality, and setting.

Each evaluation runs the judge 2 times sequentially (run 1 populates the
Anthropic prompt cache, run 2 benefits from cached reads). On 429 rate
limits, each call sleeps 30s and retries. Scores are averaged across
runs; reasoning is concatenated.

Visualization — print_score_summary():
    Queries plot_events_evaluations and prints a formatted mean/median table
    per candidate per dimension.

Storage (evaluation_data/eval.db):
    - plot_events_candidate_outputs: (movie_id, candidate_id) → candidate output
    - plot_events_evaluations: (movie_id, candidate_id) → 4-dimension scores + reasoning

Idempotent — re-running skips rows that already exist.
"""

import asyncio
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import anthropic
import pandas as pd
from pydantic import BaseModel, Field

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
    build_plot_events_prompts,
)
from movie_ingestion.metadata_generation.inputs import MovieInputData
from movie_ingestion.metadata_generation.schemas import PlotEventsOutput
from movie_ingestion.metadata_generation.prompts.plot_events import (
    SYSTEM_PROMPT as DEFAULT_SYSTEM_PROMPT,
    SYSTEM_PROMPT_SHORT as SHORT_SYSTEM_PROMPT,
    SYSTEM_PROMPT_SYNOPSIS,
    SYSTEM_PROMPT_SYNTHESIS,
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
    # EvaluationCandidate(
    #     candidate_id="plot_events__gemini-2.5-flash",
    #     provider=LLMProvider.GEMINI,
    #     model="gemini-2.5-flash",
    #     system_prompt=DEFAULT_SYSTEM_PROMPT,
    #     response_format=PlotEventsOutput,
    #     kwargs={"temperature": 0.2, "thinking_config": {"thinking_budget": 0}},
    # ),
    # EvaluationCandidate(
    #     candidate_id="plot_events__gemini-2.5-flash__think-1k",
    #     provider=LLMProvider.GEMINI,
    #     model="gemini-2.5-flash",
    #     system_prompt=DEFAULT_SYSTEM_PROMPT,
    #     response_format=PlotEventsOutput,
    #     kwargs={"temperature": 0.2, "thinking_config": {"thinking_budget": 1024}},
    # ),
    # EvaluationCandidate(
    #     candidate_id="plot_events__gemini-2.5-flash__think-4k",
    #     provider=LLMProvider.GEMINI,
    #     model="gemini-2.5-flash",
    #     system_prompt=DEFAULT_SYSTEM_PROMPT,
    #     response_format=PlotEventsOutput,
    #     kwargs={"temperature": 0.2, "thinking_config": {"thinking_budget": 4096}},
    # ),
    # -----------------------------------------------------------------------
    # Gemini 2.5 Flash Lite — 2 candidates (thinking on/off)
    # -----------------------------------------------------------------------
    # EvaluationCandidate(
    #     candidate_id="plot_events__gemini-2.5-flash-lite",
    #     provider=LLMProvider.GEMINI,
    #     model="gemini-2.5-flash-lite",
    #     system_prompt=DEFAULT_SYSTEM_PROMPT,
    #     response_format=PlotEventsOutput,
    #     kwargs={"temperature": 0.2, "thinking_config": {"thinking_budget": 0}},
    # ),
    # EvaluationCandidate(
    #     candidate_id="plot_events__gemini-2.5-flash-lite__think-1k",
    #     provider=LLMProvider.GEMINI,
    #     model="gemini-2.5-flash-lite",
    #     system_prompt=DEFAULT_SYSTEM_PROMPT,
    #     response_format=PlotEventsOutput,
    #     kwargs={"temperature": 0.2, "thinking_config": {"thinking_budget": 1024}},
    # ),
    # EvaluationCandidate(
    #     candidate_id="plot_events__gemini-2.5-flash-lite__think-4k",
    #     provider=LLMProvider.GEMINI,
    #     model="gemini-2.5-flash-lite",
    #     system_prompt=DEFAULT_SYSTEM_PROMPT,
    #     response_format=PlotEventsOutput,
    #     kwargs={"temperature": 0.2, "thinking_config": {"thinking_budget": 4096}},
    # ),
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
    # EvaluationCandidate(
    #     candidate_id="plot_events__gpt-5-mini__reason-low-verb-med",
    #     provider=LLMProvider.OPENAI,
    #     model="gpt-5-mini",
    #     system_prompt=DEFAULT_SYSTEM_PROMPT,
    #     response_format=PlotEventsOutput,
    #     kwargs={"reasoning_effort": "low", "verbosity": "medium"},
    # ),
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
        kwargs={"reasoning_effort": "none", "verbosity": "low"},
    ),
    EvaluationCandidate(
        candidate_id="plot_events__gpt-5.4-nano__reason-low",
        provider=LLMProvider.OPENAI,
        model="gpt-5.4-nano",
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        response_format=PlotEventsOutput,
        kwargs={"reasoning_effort": "low", "verbosity": "low"},
    ),
    # EvaluationCandidate(
    #     candidate_id="plot_events__gpt-5.4-nano__reason-low-verb-med",
    #     provider=LLMProvider.OPENAI,
    #     model="gpt-5.4-nano",
    #     system_prompt=DEFAULT_SYSTEM_PROMPT,
    #     response_format=PlotEventsOutput,
    #     kwargs={"reasoning_effort": "low", "verbosity": "medium"},
    # ),
    # -----------------------------------------------------------------------
    # GPT-oss-120b — 2 candidates (reasoning_effort)
    # -----------------------------------------------------------------------
    # EvaluationCandidate(
    #     candidate_id="plot_events__gpt-oss-120b",
    #     provider=LLMProvider.GROQ,
    #     model="openai/gpt-oss-120b",
    #     system_prompt=DEFAULT_SYSTEM_PROMPT,
    #     response_format=PlotEventsOutput,
    #     kwargs={"temperature": 0.2, "reasoning_effort": "low", "reasoning_format": "hidden"},
    # ),
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
    # EvaluationCandidate(
    #     candidate_id="plot_events__llama-4-scout__temp-0",
    #     provider=LLMProvider.GROQ,
    #     model="meta-llama/llama-4-scout-17b-16e-instruct",
    #     system_prompt=DEFAULT_SYSTEM_PROMPT,
    #     response_format=PlotEventsOutput,
    #     kwargs={"temperature": 0.0},
    # ),
    # -----------------------------------------------------------------------
    # Short prompt variants — copies of active candidates using SHORT_SYSTEM_PROMPT
    # -----------------------------------------------------------------------
    # EvaluationCandidate(
    #     candidate_id="plot_events__gemini-2.5-flash-lite__think-1k__short-prompt",
    #     provider=LLMProvider.GEMINI,
    #     model="gemini-2.5-flash-lite",
    #     system_prompt=SHORT_SYSTEM_PROMPT,
    #     response_format=PlotEventsOutput,
    #     kwargs={"temperature": 0.2, "thinking_config": {"thinking_budget": 1024}},
    # ),
    # EvaluationCandidate(
    #     candidate_id="plot_events__gemini-2.5-flash-lite__think-4k__short-prompt",
    #     provider=LLMProvider.GEMINI,
    #     model="gemini-2.5-flash-lite",
    #     system_prompt=SHORT_SYSTEM_PROMPT,
    #     response_format=PlotEventsOutput,
    #     kwargs={"temperature": 0.2, "thinking_config": {"thinking_budget": 4096}},
    # ),
    EvaluationCandidate(
        candidate_id="plot_events__gpt-5-mini__reason-low__short-prompt",
        provider=LLMProvider.OPENAI,
        model="gpt-5-mini",
        system_prompt=SHORT_SYSTEM_PROMPT,
        response_format=PlotEventsOutput,
        kwargs={"reasoning_effort": "low", "verbosity": "low"},
    ),
    EvaluationCandidate(
        candidate_id="plot_events__gpt-5.4-nano__short-prompt",
        provider=LLMProvider.OPENAI,
        model="gpt-5.4-nano",
        system_prompt=SHORT_SYSTEM_PROMPT,
        response_format=PlotEventsOutput,
        kwargs={"reasoning_effort": "none", "verbosity": "low"},
    ),
]

# ---------------------------------------------------------------------------
# SQLite table DDL
# ---------------------------------------------------------------------------

_CREATE_CANDIDATE_OUTPUTS_TABLE = """
    CREATE TABLE IF NOT EXISTS plot_events_candidate_outputs (
        movie_id         INTEGER NOT NULL,
        candidate_id     TEXT NOT NULL,
        plot_summary     TEXT NOT NULL,
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
        groundedness_score             REAL,
        groundedness_reasoning         TEXT,
        plot_summary_score             REAL,
        plot_summary_reasoning         TEXT,
        judge_model                    TEXT,
        judge_input_tokens             INTEGER,
        judge_output_tokens            INTEGER,
        judge_runs                     INTEGER,
        created_at                     TEXT NOT NULL,
        PRIMARY KEY (movie_id, candidate_id)
    )
"""

# Score columns used by compute_score_summary
SCORE_COLUMNS = [
    "groundedness_score",
    "plot_summary_score",
]

# Weights for computing overall_mean — summary matters most, grounded next.
SCORE_WEIGHTS: dict[str, float] = {
    "plot_summary_score": 3.0,
    "groundedness_score": 2.0,
}


def create_plot_events_tables(conn: sqlite3.Connection) -> None:
    """Create the plot_events evaluation tables if they don't exist."""
    conn.execute(_CREATE_CANDIDATE_OUTPUTS_TABLE)
    conn.execute(_CREATE_EVALUATIONS_TABLE)
    # Add judge_runs column if missing (migrates tables created before multi-run support)
    try:
        conn.execute(
            "ALTER TABLE plot_events_evaluations ADD COLUMN judge_runs INTEGER"
        )
    except sqlite3.OperationalError:
        pass  # Column already exists
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
    # Reasoning before scores (spec requirement: explicit CoT before scores).
    # Written in caveman-speak, one sentence, max 30 words each.
    groundedness_reasoning: str = Field(description="One caveman-speak sentence, max 30 words. No articles, no filler, short grunts.")
    plot_summary_reasoning: str = Field(description="One caveman-speak sentence, max 30 words. No articles, no filler, short grunts.")
    # Scores — 4-point scale per dimension
    groundedness_score: Literal[1, 2, 3, 4]
    plot_summary_score: Literal[1, 2, 3, 4]


# ---------------------------------------------------------------------------
# Judge system prompt (rubric)
# ---------------------------------------------------------------------------

JUDGE_SYSTEM_PROMPT = """You are an expert evaluator of movie metadata quality. Score LLM-generated plot_events metadata on a 4-point scale across 2 independent dimensions.

CONTEXT: plot_events metadata is a single chronological plot_summary of WHAT HAPPENS in a movie — concrete events, characters, and setting woven into one text. It is about events and facts, not themes or analysis. The output serves two purposes:
1. Vector embeddings for semantic search — preserving character names, location names, and concrete plot actions enables specific queries to match
2. Primary input to downstream metadata generators that analyze themes, character arcs, and viewer experience

A HIGH-QUALITY OUTPUT should:
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
Evaluates factual accuracy of the plot_summary. Every detail must be traceable to the provided SOURCE DATA. This is the most important dimension — hallucinated content propagates into downstream metadata and creates false search matches.

Source hierarchy for adjudicating conflicts: plot_synopsis and detailed plot_summaries are the primary truth. If sources conflict, the most detailed, internally consistent version takes precedence. Overview is a marketing summary and is the weakest source.

Score 4: All details directly supported by the SOURCE DATA. No fabricated characters, events, locations, or relationships.
Score 3: All major claims supported. At most 1-2 minor details that are reasonable inferences from the SOURCE DATA rather than directly stated (e.g., inferring a relationship dynamic that's strongly implied but not explicit).
Score 2: 1-2 details clearly absent from all SOURCE DATA fields. Not egregious fabrication, but clearly unsupported claims.
Score 1: Any clearly fabricated plot event, character name, character relationship, or setting detail. OR multiple unsupported details.

---

DIMENSION 2: plot_summary
Evaluates whether the summary provides chronological coverage of concrete events with enough specific detail for the output to be useful as a search embedding and as input to downstream generators.

A high-quality summary provides: chronological coverage of the entire film, preserving character names, location names, key organizations, and important events. Compact wording, no filler, no abstract moralizing or theme talk. Only the 1-3 core conflicts that define the movie.

Score 4: Chronological event coverage from beginning to end, focused on the 1-3 core conflicts. Character names, locations, and concrete plot actions preserved throughout. Compact, plot-grounded wording with no filler, padding, or thematic commentary. When SOURCE DATA provides specific detail, the summary reflects that specificity.
Score 3: Event coverage substantially complete but the ending is thin, or 1-2 concrete details available in the SOURCE DATA are generalized. Mostly compact — at most minor instances of vague phrasing where SOURCE DATA gave specific detail. No significant theme talk or moralizing.
Score 2: Significant events missing (ending omitted, or a major section collapsed to one sentence). OR accurate but so high-level that it reads as a premise description rather than a plot recount. OR contains noticeable filler, flowery prose, or abstract thematic commentary.
Score 1: Major plot events missing. OR so brief/generic as to be minimally useful. OR dominated by thematic analysis rather than concrete events.

When SOURCE DATA is sparse (only overview, no synopsis or summaries): do not penalize a shorter summary. Penalize padding or speculation instead. A concise, grounded summary from sparse SOURCE DATA can score 4.

---

SCORING INSTRUCTIONS:
1. For each dimension, write reasoning FIRST, then state the score.
2. Score each dimension independently — a factual error penalized in groundedness should not also lower plot_summary score.
3. Evaluate semantic content, not surface form. Two outputs expressing the same meaning differently receive the same score.
4. For verifiable facts (character names, plot events), use the SOURCE DATA as the authority. For subjective elements (how much detail), score based on defensibility.
5. Only penalize in groundedness if a detail is absent from ALL fields in the SOURCE DATA.
6. Filler, flowery prose, abstract moralizing, and thematic commentary should be penalized in plot_summary, not groundedness (they are style violations, not factual errors).

REASONING FORMAT:
Each reasoning field must be exactly ONE sentence, maximum 30 words. Write in caveman-speak: drop articles (a, an, the), drop filler words, use blunt short phrasing. Example: "Plot cover whole story good, names kept, no fluff found, ending strong." NOT full English prose."""


# ---------------------------------------------------------------------------
# Helper: serialize / deserialize PlotEventsOutput for storage
# ---------------------------------------------------------------------------

def _serialize_output(output: PlotEventsOutput) -> str:
    """Return plot_summary for DB storage."""
    return output.plot_summary


def _deserialize_output(row: sqlite3.Row) -> PlotEventsOutput:
    """Reconstruct a PlotEventsOutput from a DB row."""
    return PlotEventsOutput(
        plot_summary=row["plot_summary"],
    )


def _build_judge_user_prompt(
    source_data: str,
    candidate_output: PlotEventsOutput,
) -> str:
    """Assemble the judge's user prompt from source data + candidate output.

    The source data section contains the raw movie fields (title, overview,
    synopses, summaries, keywords) that were available to the candidate —
    NOT the generation instructions. This gives the judge ground truth for
    factual verification without exposing the candidate's system prompt.
    """
    return f"""SOURCE DATA:
{source_data}

---

CANDIDATE OUTPUT:
plot_summary: {candidate_output.plot_summary}"""


# ---------------------------------------------------------------------------
# Candidate evaluation
# ---------------------------------------------------------------------------

async def run_evaluation(
    candidates: list[EvaluationCandidate],
    movie_inputs: dict[int, MovieInputData],
    judge_model: str = "claude-opus-4-6",
    judge_provider: LLMProvider = LLMProvider.ANTHROPIC,
    concurrency: int = 5,
    judge_runs: int = 2,
    db_path: Path | None = None,
) -> None:
    """Generate candidate outputs and score them with an LLM judge.

    Reference-free evaluation: the judge scores each candidate output
    against a detailed rubric using the raw source data (not a reference
    output) for factual verification.

    For each (candidate, movie) pair:
    1. Skip if an evaluation result already exists (idempotent).
    2. Generate the candidate output if not already stored.
    3. Build the judge prompt with SOURCE DATA + CANDIDATE OUTPUT.
    4. Call the judge: run 1 first (populates Anthropic prompt cache),
       then runs 2+ in parallel (benefit from cached reads).
    5. Average scores across runs; concatenate reasoning with delimiters.

    Args:
        candidates: List of candidate configurations to evaluate.
        movie_inputs: Dict of tmdb_id → MovieInputData for the test corpus.
        judge_model: Model identifier to use as the evaluator judge.
        judge_provider: LLM provider for the judge model.
        concurrency: Max concurrent in-flight requests (generation + judge combined).
        judge_runs: Number of times to call the judge per (candidate, movie) pair.
            Scores are averaged across runs; reasoning is concatenated with run
            delimiters. Defaults to 3.
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

            # Retrieve or generate candidate output
            output_row = conn.execute(
                "SELECT * FROM plot_events_candidate_outputs "
                "WHERE movie_id = ? AND candidate_id = ?",
                (tmdb_id, candidate.candidate_id),
            ).fetchone()

            # Build the source data prompt — same raw fields the candidate
            # received (title, overview, synopses/summaries, keywords).
            # build_plot_events_user_prompt returns (user_prompt, system_prompt);
            # the judge only needs the user prompt as source data.
            source_data, _ = build_plot_events_prompts(movie)

            if output_row is not None:
                candidate_output = _deserialize_output(output_row)
                print(f"  [LOADED] {candidate.candidate_id} × {movie.title_with_year()}")
            else:
                try:
                    # Call the LLM directly with the candidate's system prompt.
                    # Each candidate may have a distinct system_prompt, and we must
                    # honour it rather than falling back to the module-level default.
                    parsed, gen_in_tokens, gen_out_tokens = await generate_llm_response_async(
                        provider=candidate.provider,
                        user_prompt=source_data,
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
                        f"tmdb_id={tmdb_id}: {type(e).__name__}: {e}"
                    )
                    return

                plot_summary = _serialize_output(candidate_output)
                conn.execute(
                    """
                    INSERT OR IGNORE INTO plot_events_candidate_outputs
                        (movie_id, candidate_id, plot_summary,
                         input_tokens, output_tokens, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        tmdb_id, candidate.candidate_id,
                        plot_summary,
                        gen_input_tokens, gen_output_tokens,
                        datetime.now(timezone.utc).isoformat(),
                    ),
                )
                conn.commit()
                print(
                    f"  [GENERATED] {candidate.candidate_id} × {movie.title_with_year()}"
                )

            judge_user_prompt = _build_judge_user_prompt(
                source_data, candidate_output
            )

            # Judge kwargs — prompt caching enabled for Anthropic (90% discount
            # on cached input tokens for runs 2+).
            judge_kwargs = {
                "provider": judge_provider,
                "user_prompt": judge_user_prompt,
                "system_prompt": JUDGE_SYSTEM_PROMPT,
                "response_format": PlotEventsJudgeOutput,
                "model": judge_model,
                "cache_control": True,
                "thinking": {"type": "disabled"},
            }

            # Run judge calls sequentially so each call benefits from
            # Anthropic's prompt cache populated by the previous call.
            # On 429 rate limits, sleep 30s and retry the same call.
            # Fail the entire evaluation if any non-retryable error occurs —
            # avoids biased averages from fewer samples. Idempotent retry
            # handles it on the next pipeline run.
            try:
                judge_results = []
                for run_idx in range(judge_runs):
                    while True:
                        try:
                            result = await generate_llm_response_async(**judge_kwargs)
                            judge_results.append(result)
                            break
                        except anthropic.RateLimitError:
                            print(
                                f"  [429] Rate limited on judge run {run_idx + 1}/{judge_runs}, "
                                f"sleeping 30s..."
                            )
                            await asyncio.sleep(30)
            except Exception as e:
                print(
                    f"  [ERROR] Judge call failed: "
                    f"candidate={candidate.candidate_id}, "
                    f"tmdb_id={tmdb_id}: {type(e).__name__}: {e}"
                )
                return

            # Unpack: each element is (parsed_output, input_tokens, output_tokens)
            judge_outputs = [r[0] for r in judge_results]
            total_judge_in_tokens = sum(r[1] for r in judge_results)
            total_judge_out_tokens = sum(r[2] for r in judge_results)

            # Average scores across all runs
            avg_groundedness = sum(j.groundedness_score for j in judge_outputs) / judge_runs
            avg_plot_summary = sum(j.plot_summary_score for j in judge_outputs) / judge_runs

            # Concatenate reasoning from all runs with delimiters for transparency
            def _combine_reasoning(field_name: str) -> str:
                parts = []
                for i, j in enumerate(judge_outputs, 1):
                    parts.append(f"--- Run {i} ---\n{getattr(j, field_name)}")
                return "\n\n".join(parts)

            conn.execute(
                """
                INSERT OR IGNORE INTO plot_events_evaluations (
                    movie_id, candidate_id,
                    groundedness_score, groundedness_reasoning,
                    plot_summary_score, plot_summary_reasoning,
                    judge_model, judge_input_tokens, judge_output_tokens,
                    judge_runs, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    tmdb_id, candidate.candidate_id,
                    avg_groundedness,
                    _combine_reasoning("groundedness_reasoning"),
                    avg_plot_summary,
                    _combine_reasoning("plot_summary_reasoning"),
                    judge_model,
                    total_judge_in_tokens,
                    total_judge_out_tokens,
                    judge_runs,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            conn.commit()
            completed += 1
            # Per-run scores for visibility into judge variance
            per_run_scores = " | ".join(
                f"run{i}: g={j.groundedness_score} s={j.plot_summary_score}"
                for i, j in enumerate(judge_outputs, 1)
            )
            print(
                f"  [{completed}/{total}] {candidate.candidate_id} × {movie.title_with_year()} | "
                f"avg: ground={avg_groundedness:.2f} summary={avg_plot_summary:.2f} | "
                f"{per_run_scores}"
            )

    # Iterate movies in the outer loop so that concurrent semaphore slots
    # are filled by different candidates (and thus different providers),
    # spreading rate-limit pressure across providers instead of hammering one.
    tasks = [
        _evaluate_one(candidate, tmdb_id, movie)
        for tmdb_id, movie in movie_inputs.items()
        for candidate in candidates
    ]
    print(f"  Launching {len(tasks)} evaluation tasks...")
    await asyncio.gather(*tasks)

    print(f"\nEvaluation complete: {completed}/{total} evaluations done.")
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
        score_weights=SCORE_WEIGHTS,
    )
    conn.close()

    if summary.empty:
        print("No evaluation results found for the requested candidates.")
        return

    # Print formatted table
    # Column header width: candidate_id may be long
    cid_width = max(len("candidate_id"), summary.index.str.len().max())
    dim_width = 14  # wide enough for "mean" / "median" values

    dims = ["groundedness", "plot_summary"]
    short_labels = ["grounded", "plot_summ"]

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

# Step 2 (Query Pre-pass) output schema — v3.
#
# Step 2 takes a natural-language query and emits a structured trait
# list plus the reasoning that grounds it. Two top-level fields:
#
#     span_analysis : per-span identification + decomposition pass
#     traits        : finalized per-trait classification
#
# This schema is the only documentation the LLM gets for the output
# shape — the system prompt does NOT duplicate this content. Each
# field's description is its own micro-prompt: purpose, how to think
# (for reasoning fields), how to derive (for decision fields), and
# worked correct/incorrect examples.

from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field

from schemas.trait_category import CategoryName


# ---------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------


class Role(str, Enum):
    """Whether the trait gates the eligible pool or ranks within it."""

    CARVER = "carver"
    QUALIFIER = "qualifier"


class Polarity(str, Enum):
    """Surface-grammar polarity from absorbed modifiers."""

    POSITIVE = "positive"
    NEGATIVE = "negative"


class Salience(str, Enum):
    """How load-bearing a qualifier is to the user's request."""

    CENTRAL = "central"
    SUPPORTING = "supporting"


# ---------------------------------------------------------------------
# Span analysis (pre-pass)
# ---------------------------------------------------------------------


class PossibleSplit(BaseModel):
    """One candidate piece tested for independent extractability from
    the parent span. Surfaces the per-piece atomicity test as its own
    object so each candidate is reasoned about in isolation.
    """

    isolated_span: str = Field(
        ...,
        description=(
            "PURPOSE: The piece of the parent span being tested — "
            "could this piece become its own trait, separated from "
            "the rest? Verbatim text from the query.\n"
            "\n"
            "WHAT TO LIST: Only the pieces a thoughtful reader would "
            "weigh as a possible split point. Not every adjacent "
            "word — only pieces that carry potential standalone "
            "meaning.\n"
            "\n"
            "EXAMPLES:\n"
            "- Parent 'creepy and atmospheric' → 'creepy' or "
            "'atmospheric' (the conjoined pieces).\n"
            "- Parent 'lone female protagonist' → 'lone' (testing "
            "whether the modifier-like word can stand alone).\n"
            "- Parent 'modern blockbuster' → 'modern' (testing "
            "whether the era qualifier extracts cleanly).\n"
            "- Parent 'Tom Hanks' → no entries (atomic; no plausible "
            "split point).\n"
            "- Parent 'sci-fi thriller' → 'sci-fi' or 'thriller' "
            "(borderline — both pieces classify, may or may not "
            "split based on reasoning)."
        ),
    )
    reasoning: str = Field(
        ...,
        description=(
            "PURPOSE: Decide whether the isolated_span can stand as "
            "a query in its own right. Pre-decision exploration — "
            "the should_split_out bool emerges from this reasoning, "
            "not the other way around. Do the analysis here, commit "
            "after.\n"
            "\n"
            "HOW TO THINK: Walk these four questions, in order:\n"
            "  1. Could a user submit just this isolated_span as "
            "their entire query and have it make sense?\n"
            "  2. Does this piece have a category home that could "
            "classify it standalone?\n"
            "  3. If you pulled this piece out, would the residual "
            "of the parent span still carry the meaning the user "
            "intended?\n"
            "  4. Would pulling this piece out create an orphan "
            "that no handler can act on?\n"
            "All four leaning yes → split. Any leaning no → don't.\n"
            "\n"
            "DO NOT write the verdict first. The reasoning must "
            "actually explore before the bool is set.\n"
            "\n"
            "CORRECT EXAMPLES:\n"
            "- Parent 'lone female protagonist' / isolated 'lone': "
            "\"'Lone' alone has no category home — it's a relational "
            "modifier (alone vs. ensemble) that needs a person-"
            "anchor. A user submitting just 'lone' as a query makes "
            "no sense. Pulling it out would orphan the modifier and "
            "lose the relational meaning the full span carries — "
            "the user is asking specifically for solo-protagonist "
            "female-led movies, not just any female-led movie.\"\n"
            "- Parent 'creepy and atmospheric' / isolated 'creepy': "
            "\"'Creepy' is a self-contained tonal want — a user "
            "could submit just 'creepy movies' and it would "
            "classify as a viewer-experience trait. Splitting it "
            "from 'and atmospheric' preserves both tonal wants "
            "distinctly rather than conflating them into one "
            "ambiguous lookup.\"\n"
            "- Parent 'modern blockbuster' / isolated 'modern': "
            "\"'Modern' could classify standalone as an era "
            "qualifier, but in this context it's calibrating "
            "'blockbuster' to a recent timeframe rather than "
            "asserting era as an independent want. Pulling it out "
            "leaves 'blockbuster' un-time-bounded, which loses what "
            "the user actually wanted.\"\n"
            "- Parent 'sci-fi thriller' / isolated 'sci-fi': "
            "\"'Sci-fi' classifies cleanly as a genre and a user "
            "could submit it alone. The residual 'thriller' also "
            "classifies as its own genre. Splitting preserves "
            "both genre constraints; keeping them whole would "
            "force a single ambiguous lookup that misses the "
            "intersection the user actually wants.\"\n"
            "\n"
            "INCORRECT EXAMPLES (avoid these patterns):\n"
            "- 'Should not split because lone is a modifier.' "
            "(Verdict-first; the prose is rationalization for an "
            "already-made decision.)\n"
            "- 'Two words, so should split.' (Mechanical word-count "
            "test — ignores the standalone-query criterion.)\n"
            "- 'Each piece is a word so it can stand alone.' "
            "(Structural test, not a usability test.)\n"
            "- 'Splitting is fine.' (Engages none of the four "
            "questions.)\n"
            "- 'The user wants both things, so split.' (User intent "
            "is what's being tested, not an input to assume.)"
        ),
    )
    should_split_out: bool = Field(
        ...,
        description=(
            "PURPOSE: Final commit on whether this isolated_span "
            "should be promoted to its own trait, derived from the "
            "reasoning above.\n"
            "\n"
            "HOW TO DERIVE: True iff the reasoning concluded the "
            "piece works as a standalone query, has a category "
            "home, and pulling it out preserves meaning. False "
            "otherwise. Must be consistent with the prose — if the "
            "reasoning argues the piece can't stand alone, this is "
            "false. Internal contradiction is a schema violation."
        ),
    )


class Decomposition(BaseModel):
    """Per-span split testing. Empty list when the span is atomic."""

    possible_splits: List[PossibleSplit] = Field(
        ...,
        description=(
            "PURPOSE: One entry per candidate piece you considered "
            "splitting out of the parent span. Empty list when the "
            "span is atomic and has no plausible split points.\n"
            "\n"
            "HOW TO POPULATE BY SPAN SHAPE:\n"
            "- Atomic spans (single proper noun, single concept, "
            "indivisible compound): empty list. Examples: 'Tom "
            "Hanks', 'sci-fi', 'horror', 'noir'.\n"
            "- Modifier-attached spans: one entry per modifier-like "
            "piece you considered isolating. Usually resolves to "
            "should_split_out=false. Examples: 'lone female "
            "protagonist' tests 'lone'; 'modern blockbuster' tests "
            "'modern'.\n"
            "- Conjunction or list spans: one entry per conjoined "
            "piece. Usually resolves to should_split_out=true. "
            "Examples: 'creepy and atmospheric' tests 'creepy' and "
            "'atmospheric'; 'fast-paced and witty' tests both.\n"
            "- Borderline compounds: list every piece a thoughtful "
            "reader would weigh; the per-piece reasoning resolves "
            "each independently. Some may split, others may not.\n"
            "\n"
            "DO NOT enumerate every adjacent word — only pieces "
            "that carry potential standalone meaning."
        ),
    )


class SpanAnalysis(BaseModel):
    """Pre-classification analysis of one candidate span — the
    identification + decomposition pass before traits are committed.
    """

    text: str = Field(
        ...,
        description=(
            "PURPOSE: The candidate span exactly as it appears in "
            "the query. Verbatim — preserve wording, casing, and "
            "any typos.\n"
            "\n"
            "WHAT TO INCLUDE: Content-bearing chunks that might "
            "become traits. Typically nouns, adjectives, and short "
            "noun phrases.\n"
            "\n"
            "WHAT TO EXCLUDE:\n"
            "- Pure filler: 'movies', 'films', 'I want', 'help me "
            "find', 'please', 'show me', 'looking for'.\n"
            "- Modifiers (these go in the modifiers field instead "
            "of being their own span): 'starring', 'directed by', "
            "'about', 'set in', 'based on', 'not', 'ideally', "
            "'preferably'. The span text holds the trait core; "
            "modifiers are tracked separately.\n"
            "\n"
            "EXAMPLES:\n"
            "- Query 'horror movies from the 90s starring Anthony "
            "Hopkins' → spans: 'horror', 'from the 90s', 'Anthony "
            "Hopkins'. ('starring' is a modifier on 'Anthony "
            "Hopkins'; 'movies' is filler.)\n"
            "- Query 'not too scary, ideally creepy and "
            "atmospheric' → spans: 'scary', 'creepy and "
            "atmospheric'. ('not too' modifies 'scary'; 'ideally' "
            "modifies the 'creepy and atmospheric' span.)\n"
            "- Query 'lone female protagonist' → span: 'lone female "
            "protagonist'. (One span; whether to split is "
            "decomposition's job, not span-extraction's.)\n"
            "- Query 'a fun comedy directed by Edgar Wright' → "
            "spans: 'fun', 'comedy', 'Edgar Wright'. ('directed by' "
            "is a modifier on 'Edgar Wright'.)"
        ),
    )
    modifiers: List[str] = Field(
        ...,
        description=(
            "PURPOSE: Verbatim modifier words/phrases that apply to "
            "this span — from inside the span, adjacent to it, or "
            "elsewhere in the query (distributed application). The "
            "raw input that the trait-level modifiers list will "
            "promote into structured TraitModifier objects.\n"
            "\n"
            "TYPES TO CAPTURE:\n"
            "- Polarity setters: 'not', 'without', 'no', 'avoid'.\n"
            "- Hedges: 'ideally', 'preferably', 'would love', "
            "'maybe', 'kind of'.\n"
            "- Intensity adjusters: 'very', 'too', 'not too', 'a "
            "bit', 'slightly', 'extremely'.\n"
            "- Role markers: 'starring', 'directed by', 'about', "
            "'set in', 'based on', 'featuring', 'written by'.\n"
            "\n"
            "DISTRIBUTED APPLICATION: A modifier elsewhere in the "
            "query can apply to this span via grammar even if not "
            "adjacent. In 'not too dark or sad', 'not' applies to "
            "BOTH 'dark' and 'sad' — both spans get 'not' in their "
            "modifiers list.\n"
            "\n"
            "FORMAT: Verbatim phrases. Preserve compound modifiers "
            "as the user wrote them — 'not too' stays as one entry "
            "['not too'], not split into ['not', 'too'].\n"
            "\n"
            "EMPTY LIST when the span has no modifiers attached.\n"
            "\n"
            "EXAMPLES:\n"
            "- Span 'Anthony Hopkins' in 'starring Anthony Hopkins' "
            "→ ['starring'].\n"
            "- Span 'scary' in 'not too scary' → ['not too'].\n"
            "- Span 'sad' in 'not too dark or sad' → ['not'] "
            "(distributed; 'too' scopes more tightly to 'dark').\n"
            "- Span 'creepy and atmospheric' in 'ideally creepy "
            "and atmospheric' → ['ideally'].\n"
            "- Span 'horror' in 'horror movies from the 90s' → [].\n"
            "- Span 'AI' in 'a thriller about AI' → ['about']."
        ),
    )
    query_context: str = Field(
        ...,
        description=(
            "PURPOSE: Capture how this span's meaning is shaped by "
            "OTHER (non-modifier) spans in the query, or how it "
            "shapes them. The cross-span semantic dependency layer. "
            "Catches relativized meanings — 'funny' inside a horror "
            "frame means tonal lightness, not standalone comedy.\n"
            "\n"
            "HOW TO THINK: Walk every other content-bearing span "
            "in the query. For each, ask: 'if I removed that span, "
            "would this span's meaning shift?' If yes, articulate "
            "the shift. Bidirectional — note both inbound shaping "
            "(others affect me) and outbound shaping (I affect "
            "others). If the span stands fully alone with no "
            "cross-span dependency, leave the field empty (\"\").\n"
            "\n"
            "CONSTRAINTS:\n"
            "- Use only evidence present in the query. Do NOT "
            "import outside knowledge ('Anthony Hopkins is known "
            "for…'). Treat the query as the only source of truth.\n"
            "- Describe relationships, not the span's own content. "
            "'Funny means humorous' is content; 'funny is "
            "calibrated by horror' is a relationship.\n"
            "- Empty (\"\") is the correct answer when no "
            "relationship exists. Do NOT narrate emptiness — leave "
            "the field literally blank.\n"
            "\n"
            "CORRECT EXAMPLES:\n"
            "- Query 'a funny horror movie' / span 'funny': "
            "\"Functioning relative to 'horror' — 'funny' here "
            "means tonal lightness inside a horror frame, not "
            "standalone comedy. The horror span sets the genre "
            "context this funniness operates within.\"\n"
            "- Query 'a sci-fi thriller about AI' / span 'about "
            "AI': \"Narrows the sci-fi/thriller frame to a specific "
            "subject. Without it, sci-fi/thriller would be a broad "
            "genre constraint; with it, the genre is contextualized "
            "to AI-themed content.\"\n"
            "- Query 'a slow-burn horror but not too disturbing' / "
            "span 'slow-burn': \"Modifies the horror experience "
            "the user wants — slower pacing within the horror "
            "frame. Operates alongside 'not too disturbing' to "
            "sketch a specific subgenre flavor (atmospheric / "
            "psychological rather than visceral).\"\n"
            "- Query 'horror movies from the 90s starring Anthony "
            "Hopkins' / span 'Anthony Hopkins': \"\" (empty — "
            "actor name doesn't depend on or shape the other "
            "spans).\n"
            "\n"
            "INCORRECT EXAMPLES (avoid these patterns):\n"
            "- 'The user wants something that makes them laugh.' "
            "(Restates the span's content, not its dependency on "
            "others.)\n"
            "- 'Anthony Hopkins is known for Silence of the "
            "Lambs.' (Imports outside knowledge.)\n"
            "- 'The span stands alone in the query.' (Filler — the "
            "empty case is signaled by leaving the field blank.)\n"
            "- 'Horror is a genre.' (Describes another span's "
            "content rather than the relationship between spans.)\n"
            "- 'This is the main span of the query.' (Talks about "
            "structural position, not semantic dependency.)"
        ),
    )
    decomposition: Decomposition = Field(
        ...,
        description=(
            "PURPOSE: Per-piece atomicity testing for this span. "
            "Decides whether the span stays whole as one trait or "
            "breaks into multiple traits via per-piece evaluation."
        ),
    )


# ---------------------------------------------------------------------
# Trait list (committed)
# ---------------------------------------------------------------------


class TraitModifier(BaseModel):
    """One modifier absorbed into a trait, with its user-intent
    impact articulated."""

    text: str = Field(
        ...,
        description=(
            "PURPOSE: The modifier verbatim — the word or phrase "
            "from the query that modifies this trait. Preserve "
            "wording.\n"
            "\n"
            "EXAMPLES: 'not', 'not too', 'ideally', 'starring', "
            "'a bit', 'directed by', 'preferably', 'about', 'set "
            "in', 'extremely', 'slightly'."
        ),
    )
    impact: str = Field(
        ...,
        description=(
            "PURPOSE: Articulate what the user is communicating "
            "about how to handle this trait. User-intent level — "
            "speak in terms of the user's request, NOT system "
            "operations.\n"
            "\n"
            "HOW TO THINK: Ask 'what is the user telling me about "
            "how to handle this trait?' Stay close to what's "
            "actually said. Modifiers usually weight or filter — "
            "they rarely pin exact thresholds. Err toward "
            "conservative inference: if the modifier doesn't "
            "explicitly mandate something, do not claim it does.\n"
            "\n"
            "FORBIDDEN LANGUAGE: Do NOT use system-operation terms. "
            "'Polarity flip', 'weight reduction', 'salience marker', "
            "'negation operator', 'central salience' belong to the "
            "structured fields (polarity, salience), not here. "
            "Speak in user-request terms about what the user wants, "
            "not what the system should do internally.\n"
            "\n"
            "CORRECT EXAMPLES:\n"
            "- 'scary' + 'not': \"Shows the user wants to filter "
            "out or downrank movies that have this trait.\"\n"
            "- 'scary' + 'a bit': \"Reduces the importance of "
            "scariness relative to other traits in the query — "
            "wanted but not central.\"\n"
            "- 'boring' + 'not too': \"User accepts some boredom "
            "but wants to avoid movies that lean heavily into it. "
            "Mild is OK; heavy is not.\"\n"
            "- 'Tom Hanks' + 'starring': \"Tom Hanks should be one "
            "of the most prominent actors in the cast — featured "
            "rather than minor. Does not require him to be the #1 "
            "lead.\"\n"
            "- 'creepy' + 'ideally': \"User prefers but does not "
            "require this. A movie that misses the mark on "
            "creepiness is still acceptable.\"\n"
            "- 'AI' + 'about': \"Binds AI as the topical subject — "
            "the movie should be substantively about AI, not just "
            "incidentally feature it.\"\n"
            "- 'dark' + 'extremely': \"User wants the trait pushed "
            "hard — only movies that go heavily in this direction "
            "qualify; mild versions don't satisfy.\"\n"
            "- 'Christopher Nolan' + 'directed by': \"Binds "
            "Christopher Nolan to the director role specifically — "
            "him having any other credit on the movie wouldn't "
            "satisfy this trait.\"\n"
            "\n"
            "INCORRECT EXAMPLES (avoid these patterns):\n"
            "- 'not' → 'Polarity flip to negative.' (System "
            "operation language — forbidden here.)\n"
            "- 'starring' → 'Tom Hanks must be the lead actor.' "
            "(Over-infers — 'starring' doesn't pin top billing.)\n"
            "- 'ideally' → 'User strongly prefers this.' (Reads "
            "the hedge as strength; ideally is a softener.)\n"
            "- 'not' → 'Means the user doesn't want it scary.' "
            "(Tautological — restates surface meaning without "
            "articulating handling.)\n"
            "- 'about' → 'The movie must be exclusively about "
            "AI.' (Over-infers exclusivity.)\n"
            "- 'a bit' → 'Salience: supporting.' (Pre-commits the "
            "structured salience field instead of expressing user "
            "intent.)\n"
            "- 'not too' → 'Negative.' (Under-specifies — loses "
            "the intensity hedge.)"
        ),
    )


class CategoryCandidate(BaseModel):
    """One candidate category considered as a home for a trait, with
    the boundary comparison spelled out."""

    candidate: CategoryName = Field(
        ...,
        description=(
            "PURPOSE: The category enum being considered as a home "
            "for this trait. Pick from the taxonomy.\n"
            "\n"
            "WHEN TO INCLUDE: Only genuine boundary contenders — "
            "categories you actually weighed as the home, not "
            "categories you can construct an argument for after "
            "the fact. Several categories may pass a vibes 'could "
            "fit' check; do not list those.\n"
            "\n"
            "CAP: 2-4 entries per trait typically. Even unambiguous "
            "traits emit at least one entry — schema uniformity "
            "matters for downstream merging."
        ),
    )
    fits: str = Field(
        ...,
        description=(
            "PURPOSE: Articulate why this category could plausibly "
            "be the trait's home AND how strong the alignment is. "
            "Both pieces in one statement — alignment reasoning + "
            "alignment strength.\n"
            "\n"
            "HOW TO THINK: Identify the specific axis of the "
            "candidate category. Identify the specific feature of "
            "the trait. Articulate how they line up. Then qualify "
            "how prototypically — is the trait a canonical surface "
            "form for this category, a clear secondary mention, or "
            "only an indirect implication?\n"
            "\n"
            "STRENGTH MUST BE GROUNDED in observable evidence: "
            "prototypicality of the surface form, axis directness, "
            "whether the trait names the category's home concept "
            "directly or only adjacent to it. 'Strong alignment' "
            "without grounding is filler — point to specific "
            "features.\n"
            "\n"
            "CORRECT EXAMPLES:\n"
            "- Trait 'blockbuster', candidate Cat 17 (Financial "
            "scale): \"Cat 17 covers budget and box-office "
            "magnitude as a single financial-scale axis. "
            "'Blockbuster' is canonical high-budget / high-grossing "
            "language — one of the prototypical surface forms for "
            "this category. Strong alignment, no interpretation "
            "needed.\"\n"
            "- Trait 'blockbuster', candidate Cat 38 (Reception-"
            "prose descriptors): \"Cat 38 covers reception-prose "
            "framing ('classic', 'era-defining'). 'Blockbuster' "
            "carries an implied reception signal — a movie called "
            "blockbuster is by definition widely received — but "
            "this is downstream of the financial signal. Plausible "
            "candidate; alignment is indirect rather than "
            "prototypical.\"\n"
            "- Trait 'creepy', candidate Cat 32 (Viewer "
            "experience): \"Cat 32 covers tonal/experiential "
            "vectors. 'Creepy' names a sustained-unease tonal "
            "quality directly. Strong, prototypical alignment — "
            "this is the canonical kind of trait for this "
            "category.\"\n"
            "- Trait 'Tom Hanks', candidate Cat 1 (Person credit): "
            "\"Cat 1 covers named-person constraints. 'Tom Hanks' "
            "is a named actor — the category's home concept "
            "directly. Strong alignment.\"\n"
            "- Trait 'about AI', candidate Cat 32 (Viewer "
            "experience): \"Cat 32 covers experiential framing. "
            "'About AI' is a topical-subject trait, which adjacent "
            "to but not the center of Cat 32's axis. Plausible — "
            "AI subject matter does shape viewer experience — but "
            "alignment is indirect.\"\n"
            "\n"
            "INCORRECT EXAMPLES (avoid these patterns):\n"
            "- 'It's a kind of movie.' (Vague; doesn't engage the "
            "category definition.)\n"
            "- 'Could fit.' (Zero content.)\n"
            "- 'Cat 17 is for financial scale and blockbuster "
            "fits.' (States alignment but doesn't ground how "
            "strongly.)\n"
            "- 'Strong alignment.' (Asserts strength without "
            "evidence.)\n"
            "- 'Talks about that kind of stuff.' (Generic, "
            "unspecific.)\n"
            "- 'This category covers many things including this "
            "one.' (Vague — does not name the specific axis or "
            "the specific trait feature.)"
        ),
    )
    doesnt_fit: str = Field(
        ...,
        description=(
            "PURPOSE: Specific boundary concern that pulls this "
            "candidate away from being the home AND how serious "
            "the concern actually is. Both pieces in one "
            "statement.\n"
            "\n"
            "HOW TO THINK: What evidence — in the trait or in "
            "query_context — pulls away from this category? Wrong "
            "axis (financial vs. reception vs. experiential)? "
            "Missing surface form (the trait doesn't actually name "
            "what the category covers)? Wrong role-shape (this "
            "trait is gating, the category is qualifying-style)? "
            "Identify the concern, then qualify how serious it "
            "is — does the evidence directly contradict the "
            "category, or is the concern only theoretical?\n"
            "\n"
            "EMPTY (\"\") is the correct answer when there is NO "
            "real concern. Do NOT manufacture concerns to fill the "
            "field. Asymmetry with `fits` is the expected case for "
            "clean candidates — strong fits + empty doesnt_fit.\n"
            "\n"
            "FORBIDDEN: Do not restate the category's purpose "
            "without contrasting it with the trait. 'Cat 38 is for "
            "reception' is not a concern — it's a definition. The "
            "concern is 'this trait's reception signal is "
            "downstream of its financial signal'.\n"
            "\n"
            "CORRECT EXAMPLES:\n"
            "- Trait 'blockbuster', candidate Cat 38: \"Cat 38 "
            "captures reception-prose framing, which 'blockbuster' "
            "adjacent-implies but doesn't directly assert. The "
            "financial-scale axis is the dominant signal; "
            "reception is a downstream consequence. Routing here "
            "would lose the budget/box-office specificity. "
            "Moderate concern — genuine adjacency, but financial "
            "scale is clearly the primary axis.\"\n"
            "- Trait 'classic', candidate Cat 17 (Financial "
            "scale): \"Cat 17 is financial scale, but 'classic' "
            "names reception-prose framing, not budget. The two "
            "are correlated — classic movies often had real "
            "budgets — but the trait's surface signal is reception, "
            "not financial. Strong concern: wrong axis entirely.\"\n"
            "- Trait 'about AI', candidate Cat 32 (Viewer "
            "experience): \"'About AI' names a topical subject, "
            "while Cat 32 covers tonal/experiential framing. The "
            "trait shapes experience indirectly via subject matter, "
            "but the surface signal is topical, not experiential. "
            "Moderate concern — listing this candidate to surface "
            "the boundary, but the trait's primary axis is "
            "topical.\"\n"
            "- Trait 'Tom Hanks', candidate Cat 1: \"\" (empty — "
            "actor name maps cleanly, no boundary concern).\n"
            "- Trait 'creepy', candidate Cat 32: \"\" (empty — "
            "trait is prototypical for this category).\n"
            "\n"
            "INCORRECT EXAMPLES (avoid these patterns):\n"
            "- 'Maybe doesn't fit.' (Phantom concern, no "
            "specifics. Should have been left empty.)\n"
            "- 'Cat 38 is for reception.' (Restates the cat's "
            "purpose; doesn't say why this trait pulls away.)\n"
            "- 'Some concern.' (Says nothing.)\n"
            "- 'Could be wrong.' (Manufactured uncertainty.)\n"
            "- 'There might be other categories that also fit.' "
            "(Talks about other categories instead of this one's "
            "boundary problem.)\n"
            "- (Filling doesnt_fit when there is no genuine "
            "concern, just to balance with `fits` — leave it "
            "empty.)"
        ),
    )


class Trait(BaseModel):
    """One finalized trait extracted from the query, with its
    classification."""

    query_phrase: str = Field(
        ...,
        description=(
            "PURPOSE: The trait core, exactly as it appears in the "
            "query, with modifiers excluded. The 'thing' the user "
            "is asking for along this trait.\n"
            "\n"
            "FORMAT: Verbatim from the query — preserve wording, "
            "casing, and typos. Do not paraphrase, do not normalize.\n"
            "\n"
            "EXAMPLES:\n"
            "- Query 'starring Tom Hanks' → 'Tom Hanks'. ('starring' "
            "is a modifier, not part of the phrase.)\n"
            "- Query 'not too scary' → 'scary'. ('not too' is a "
            "modifier.)\n"
            "- Query 'lone female protagonist' (decomposition kept "
            "whole) → 'lone female protagonist'. (Whole span is "
            "the trait; 'lone' is internal but lives in the "
            "modifiers list.)\n"
            "- Query 'creepy and atmospheric' (decomposition split) "
            "→ trait 1: 'creepy', trait 2: 'atmospheric'.\n"
            "- Query 'directed by Edgar Wright' → 'Edgar Wright'. "
            "('directed by' is a modifier.)"
        ),
    )
    modifiers: List[TraitModifier] = Field(
        ...,
        description=(
            "PURPOSE: Modifiers attached to this trait, with each "
            "modifier's user-intent impact articulated. Empty list "
            "when no modifiers apply.\n"
            "\n"
            "INHERITED from the corresponding span_analysis entry's "
            "modifiers list, narrowed to those that apply to this "
            "specific trait after any decomposition splits. Each "
            "bare-string modifier from the span level becomes a "
            "TraitModifier object here with a user-intent impact "
            "statement attached."
        ),
    )
    purpose_in_query: str = Field(
        ...,
        description=(
            "PURPOSE: A complete reading of what this trait is "
            "trying to accomplish in the context of the query. The "
            "central reasoning hub for the trait — every committed "
            "downstream field (category_candidates, "
            "best_fit_category, role, polarity, salience) draws on "
            "this. Do the real understanding work here, in concrete "
            "user-want language, before committing any structured "
            "label.\n"
            "\n"
            "HOW TO THINK: Cover four dimensions, in concrete "
            "user-want language:\n"
            "  1. CONCRETE WANT/AVOIDANCE: What specifically about "
            "a movie does the user want this trait to provide or "
            "rule out? Speak in things-in-the-movie terms (actors, "
            "eras, plot shapes, tonal qualities, audiences).\n"
            "  2. RELATIONSHIPS: How does this trait interact with "
            "other traits in the query? Does it constrain another "
            "(genre frames it), get constrained by another, pair "
            "with another (parallel wants), or duplicate another?\n"
            "  3. CRITICALITY: Would the user reject a movie that "
            "fails this trait, accept it grudgingly, or barely "
            "notice? Pull from query phrasing — hedges, headline "
            "position, repetition.\n"
            "  4. CONCRETENESS over abstraction: describe in user-"
            "facing terms what the trait wants from a movie, not "
            "in system or category terms.\n"
            "\n"
            "FORBIDDEN: Do NOT name any category enum, role label "
            "('carver', 'qualifier'), polarity label ('positive', "
            "'negative'), or salience label ('central', "
            "'supporting') inside this field. Those structured "
            "fields exist to capture commitment; this field exists "
            "to support it. Naming them here collapses reasoning "
            "into pre-commitment and turns the downstream fields "
            "into rubber stamps.\n"
            "\n"
            "LENGTH: Substantive but not rambling — 3-5 sentences "
            "typically.\n"
            "\n"
            "CORRECT EXAMPLES:\n"
            "- Query 'horror movies from the 90s starring Anthony "
            "Hopkins' / trait 'Anthony Hopkins': \"User wants "
            "movies that feature Anthony Hopkins prominently in "
            "the cast — the 'starring' modifier signals featured-"
            "not-cameo, though not necessarily top-billed. Sits "
            "alongside the genre constraint (horror) and the era "
            "constraint (90s) as one of three intersecting "
            "requirements; together they define a narrow eligible "
            "set. The actor is named directly with no hedge, "
            "making it a hard requirement. A movie that doesn't "
            "feature Anthony Hopkins doesn't partially satisfy the "
            "query — it fails it.\"\n"
            "- Query 'ideally creepy and atmospheric horror' / "
            "trait 'creepy': \"User wants the horror to deliver "
            "sustained tonal unease — the slow-burn, get-under-"
            "your-skin variety rather than jump-scare adrenaline. "
            "Operates inside the horror frame (the genre is the "
            "narrowing constraint), so it's a quality of the "
            "eligible set rather than a standalone constraint. "
            "Paired with 'atmospheric' as a parallel tonal want; "
            "together they sketch a psychological/slow-burn flavor "
            "of horror. The 'ideally' hedge marks it as preferred, "
            "not required — a horror movie that's less creepy is "
            "still acceptable.\"\n"
            "- Query 'something funny for date night, not too "
            "long' / trait 'not too long': \"User wants the "
            "runtime kept moderate — long movies are penalized but "
            "not strictly excluded. Sits in service of the date-"
            "night frame, where attention budget is constrained. "
            "Supports the headline want (funny date-night feel) "
            "rather than being the headline itself; the user would "
            "still take a slightly long movie if it nailed the "
            "rest.\"\n"
            "- Query 'feel-good 90s comedy for date night' / "
            "trait 'feel-good': \"User wants warmth and emotional "
            "uplift — movies that leave you in a good mood. The "
            "tonal headline of the request; sits over the more "
            "structural constraints (90s, comedy, date-night) and "
            "shapes how those constraints feel together. Hard "
            "requirement implicitly — a 90s comedy that wasn't "
            "feel-good would miss what the user is actually "
            "asking for.\"\n"
            "\n"
            "INCORRECT EXAMPLES (avoid these patterns):\n"
            "- 'Carving constraint.' (Truncates to role-shape "
            "only; skips the four dimensions.)\n"
            "- 'Cat 1 person-credit constraint, hard, central.' "
            "(Speaks in category and role labels — forbidden.)\n"
            "- 'User wants Anthony Hopkins in the movie.' "
            "(Concrete but missing relationships, criticality, "
            "modifier nuance.)\n"
            "- 'This is a positive carver in Cat 1 with central "
            "salience.' (Pre-commits all four downstream fields.)\n"
            "- 'Horror is a genre, so this is a genre constraint.' "
            "(Wrong trait — describes a different span.)\n"
            "- 'The user wants this trait.' (Tautological.)"
        ),
    )
    category_candidates: List[CategoryCandidate] = Field(
        ...,
        description=(
            "PURPOSE: Genuine boundary contenders for where this "
            "trait should live, with each candidate's fit and "
            "non-fit reasoning spelled out. Forces explicit "
            "boundary comparison rather than gestalt one-shot "
            "category picking.\n"
            "\n"
            "CAP: 2-4 entries typically. Single-candidate cases "
            "still emit one entry (with potentially empty "
            "doesnt_fit) — schema uniformity matters for "
            "downstream merging.\n"
            "\n"
            "DO NOT enumerate every plausible category. Only "
            "include categories you actually weighed — not "
            "categories you can construct an argument for after "
            "the fact."
        ),
    )
    best_fit_category: CategoryName = Field(
        ...,
        description=(
            "PURPOSE: The committed category pick — the trait's "
            "home for downstream handler dispatch.\n"
            "\n"
            "HOW TO DERIVE FROM category_candidates: Pick the "
            "candidate whose `fits` reasoning describes the most "
            "direct, prototypical alignment AND whose `doesnt_fit` "
            "concern is weakest (or empty). The pick must come "
            "from the candidate list — do NOT pick a category you "
            "didn't list as a candidate. If two candidates are "
            "close, pick the one whose `fits` reasoning grounds "
            "alignment in surface-form prototypicality rather than "
            "indirect implication.\n"
            "\n"
            "CONSISTENCY: best_fit_category MUST be one of the "
            "`candidate` enums in category_candidates. Mismatch is "
            "a schema violation."
        ),
    )
    role: Role = Field(
        ...,
        description=(
            "PURPOSE: Whether this trait gates the eligible pool "
            "(carver) or ranks within it (qualifier).\n"
            "\n"
            "HOW TO DERIVE FROM purpose_in_query:\n"
            "- CARVER if purpose_in_query indicates a movie "
            "failing this trait should be eliminated, not just "
            "demoted. Hard requirements, definitional constraints, "
            "and named entities (actors, directors, franchises, "
            "genres, era ranges) are typically carvers.\n"
            "- QUALIFIER if purpose_in_query indicates a movie "
            "failing this trait should be demoted but not "
            "eliminated. Tonal preferences, fuzzy quality wants, "
            "and hedged language ('ideally', 'preferably', 'a "
            "bit') are typically qualifiers.\n"
            "\n"
            "CATEGORY PRIOR (tiebreaker only): Some categories are "
            "inherently carver-shaped (person credits, franchise "
            "membership, hard metadata like year ranges); others "
            "are inherently qualifier-shaped (tonal vectors, "
            "viewer experience, vague reception prose). Use this "
            "as a tiebreaker — purpose_in_query is the primary "
            "signal.\n"
            "\n"
            "EXAMPLES:\n"
            "- 'Anthony Hopkins' (named actor, no hedge): CARVER.\n"
            "- 'creepy' with 'ideally' hedge: QUALIFIER.\n"
            "- 'horror' (named genre, hard ask): CARVER.\n"
            "- 'feel-good' (tonal want, no hedge but fuzzy): "
            "QUALIFIER.\n"
            "- 'from the 90s' (era constraint, hard): CARVER.\n"
            "- 'a bit dark' (hedged tonal want): QUALIFIER.\n"
            "- 'about AI' (topical subject, hard ask): CARVER.\n"
            "- 'not too long' (runtime preference): QUALIFIER.\n"
            "- 'not depressing' (negative tonal preference): "
            "QUALIFIER (the polarity is negative but the role is "
            "qualifier — fuzzy tonal want, not a hard exclusion)."
        ),
    )
    polarity: Polarity = Field(
        ...,
        description=(
            "PURPOSE: Whether the user wants this trait present "
            "(POSITIVE) or absent (NEGATIVE). Surface-grammar rule "
            "— mechanical lookup of polarity setters in the "
            "modifiers list.\n"
            "\n"
            "HOW TO DERIVE: If any modifier in the trait's "
            "modifiers list contains a polarity setter ('not', "
            "'without', 'no', 'avoid'), polarity = NEGATIVE. "
            "Otherwise POSITIVE.\n"
            "\n"
            "DO NOT INTERPRET INTENT. 'Not too dark' has 'not' as "
            "a polarity setter, so polarity = NEGATIVE on the "
            "'dark' trait — even though the user is expressing a "
            "preference rather than absolute exclusion. Downstream "
            "handlers interpret what negative-polarity-with-'not "
            "too' means; this field just records the surface "
            "signal.\n"
            "\n"
            "INTENSITY/HEDGE WORDS ARE NOT POLARITY SETTERS. 'Too', "
            "'very', 'a bit', 'ideally', 'preferably', 'kind of' "
            "do NOT flip polarity. They're intensity or salience "
            "signals, captured elsewhere.\n"
            "\n"
            "EXAMPLES:\n"
            "- Trait 'scary' with modifier 'not': NEGATIVE.\n"
            "- Trait 'boring' with modifier 'not too': NEGATIVE "
            "(presence of 'not' triggers it).\n"
            "- Trait 'creepy' with modifier 'ideally': POSITIVE "
            "(no polarity setter).\n"
            "- Trait 'Anthony Hopkins' with modifier 'starring': "
            "POSITIVE.\n"
            "- Trait 'horror' with no modifiers: POSITIVE (default "
            "for unmarked traits).\n"
            "- Trait 'sequels' with modifier 'avoid': NEGATIVE.\n"
            "- Trait 'depressing' with modifier 'not': NEGATIVE.\n"
            "- Trait 'long' with modifier 'too': POSITIVE ('too' "
            "alone is not a polarity setter — note: 'not too long' "
            "would have BOTH 'not' and 'too' in modifiers and "
            "would be NEGATIVE because of the 'not')."
        ),
    )
    salience: Optional[Salience] = Field(
        None,
        description=(
            "PURPOSE: For qualifiers, how load-bearing the trait "
            "is to the user's request. CARVERS get NULL — salience "
            "doesn't apply to gating constraints (rarity does the "
            "weight work for carvers downstream).\n"
            "\n"
            "VALUES:\n"
            "- CENTRAL: a headline want; the query would feel "
            "fundamentally different without this qualifier. The "
            "user is leaning on this to define what they want.\n"
            "- SUPPORTING: meaningful but not load-bearing; rounds "
            "out the picture but a movie missing it would still "
            "feel right.\n"
            "- NULL: trait is a carver; salience does not apply.\n"
            "\n"
            "HOW TO DERIVE FROM purpose_in_query AND modifiers:\n"
            "- If role is CARVER → NULL. Skip the rest of this "
            "logic.\n"
            "- Hedges in modifiers ('ideally', 'preferably', 'a "
            "bit', 'kind of', 'maybe') push toward SUPPORTING — "
            "the user is signaling the want is not load-bearing.\n"
            "- Headline position in purpose_in_query (described "
            "as 'the headline want', 'the tonal center', 'what "
            "the user is actually asking for') pushes toward "
            "CENTRAL.\n"
            "- Supporting language in purpose_in_query ('rounds "
            "out', 'supports the frame', 'marginal preference', "
            "'sits in service of') pushes toward SUPPORTING.\n"
            "- A hedge does NOT automatically demote a headline "
            "want — 'ideally creepy and atmospheric horror' has "
            "'creepy' as the tonal headline despite the 'ideally'. "
            "Headline position in the request structure beats "
            "softener vocabulary.\n"
            "\n"
            "EXAMPLES:\n"
            "- Trait 'creepy' (qualifier) in 'ideally creepy and "
            "atmospheric horror': CENTRAL (paired headline tonal "
            "want; the 'ideally' hedge softens but doesn't demote "
            "the headline).\n"
            "- Trait 'not too long' (qualifier) in 'something "
            "funny for date night, not too long': SUPPORTING "
            "(rounds out the date-night frame, not the headline).\n"
            "- Trait 'a bit dark' (qualifier) in 'a thriller, "
            "ideally a bit dark': SUPPORTING ('a bit' hedge plus "
            "soft 'ideally' both indicate non-headline weight).\n"
            "- Trait 'feel-good' (qualifier) in 'feel-good 90s "
            "comedy for date night': CENTRAL ('feel-good' is the "
            "tonal headline of the request).\n"
            "- Trait 'Anthony Hopkins' (carver): NULL.\n"
            "- Trait 'horror' (carver): NULL.\n"
            "- Trait 'from the 90s' (carver): NULL."
        ),
    )


# ---------------------------------------------------------------------
# Top-level response
# ---------------------------------------------------------------------


class Step2Response(BaseModel):
    """v3 step 2 output: span analysis + trait list."""

    span_analysis: List[SpanAnalysis] = Field(
        ...,
        description=(
            "PURPOSE: Pre-classification reasoning pass. One entry "
            "per content-bearing span in the query. Establishes "
            "boundaries, modifiers, cross-span dependencies, and "
            "decomposition before any trait-level commitments are "
            "made.\n"
            "\n"
            "WHAT TO SKIP: Pure filler ('movies', 'films', 'I "
            "want', 'help me find', 'please', 'show me'). These "
            "are not spans."
        ),
    )
    traits: List[Trait] = Field(
        ...,
        description=(
            "PURPOSE: The committed trait list. One entry per "
            "trait, after any decomposition splits from "
            "span_analysis. Length is at least len(span_analysis); "
            "more if any spans split into multiple traits.\n"
            "\n"
            "RELATIONSHIP TO span_analysis: Each trait inherits "
            "its modifiers from the corresponding span_analysis "
            "entry's modifiers list, narrowed to the specific "
            "sub-span if decomposition split it. The query_context "
            "from span_analysis informs purpose_in_query at the "
            "trait level."
        ),
    )

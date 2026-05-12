# Step 4 chronological endpoint structured output model.
#
# Translates one Step 3 CategoryCall (expressions + retrieval_intent)
# into a single direction commit: should the candidate pool be
# scored so the OLDEST movie wins, or the NEWEST movie wins?
#
# Enum values are named after WHICH EXTREME SCORES 1.0 so the
# choice is unambiguous in the LLM output and in human reads —
# "chronological / reverse_chronological" was previously confusing
# because the English adjective has competing conventions (chronicle
# order = oldest first; modern feed order = newest first).
#
# This is a POOL_RERANKER that emits a continuous percentile curve
# over the candidate pool's release_dates — every distinct release
# date occupies its own slot in [0, 1] (a 1-day difference always
# matters). The scoring math lives in db/chronological_scoring.py;
# the executor wrapper lives in
# search_v2/endpoint_fetching/chronological_query_execution.py.
#
# Distinct from RELEASE_DATE (Cat 13), which expresses date RANGES /
# windows with a saturation cap. CHRONOLOGICAL never saturates —
# the most-extreme movie in the pool always wins.
#
# No class docstrings — this class is the LLM `response_format`
# schema; anything declared here propagates to every API call. See
# the convention note in schemas/endpoint_parameters.py.

from enum import StrEnum

from pydantic import Field

from schemas.endpoint_parameters import EndpointParameters


class ChronologicalDirection(StrEnum):
    OLDEST_FIRST = "oldest_first"   # oldest movie in pool scores 1.0
    NEWEST_FIRST = "newest_first"   # newest movie in pool scores 1.0


class ChronologicalQuerySpec(EndpointParameters):
    direction: ChronologicalDirection = Field(
        ...,
        description=(
            "Which extreme of the candidate pool should score 1.0. "
            "`oldest_first` — the OLDEST movie in the pool scores "
            "1.0; pick this when the phrasing prefers old / "
            "earliest movies ('first', 'earliest', 'oldest', "
            "'original'). `newest_first` — the NEWEST movie in the "
            "pool scores 1.0; pick this when the phrasing prefers "
            "new / latest movies ('latest', 'newest', 'most "
            "recent'). The enum value names the winning end of the "
            "curve directly — don't reason about sort order or "
            "'chronological direction', reason about which extreme "
            "the user wants on top."
        ),
    )

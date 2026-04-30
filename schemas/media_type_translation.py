# Step 3 media-type endpoint structured output model.
#
# Translates a media-type dealbreaker or preference description from
# step 2 into a concrete query specification that step 4 can execute
# against public.movie_card.release_format.
#
# Receives: intent_rewrite (step 1) + one item's description and
# route_rationale (step 2).
#
# Closed-enum design: this endpoint exists specifically for queries that
# explicitly name a non-default release format. Of ReleaseFormat's five
# values, the LLM-facing subset is just (TV_MOVIE, SHORT, VIDEO):
#   - MOVIE is the default release container; "show me movies" carries no
#     media-type signal at all, so the trait should not have fired and
#     emitting MOVIE here would be a no-op at best, a misroute at worst.
#   - UNKNOWN is the audit sentinel for unsupported / missing IMDB title
#     types and is never user-requestable.
# Encoding the subset as a Literal type constrains the JSON schema sent on
# every API call so the model physically cannot return either excluded
# value. Mirrors the closed-enum brand path on the studio endpoint
# (schemas/studio_translation.py) and avoids the freeform-name complexity
# that studio needs only because production companies are unbounded.
#
# No class-level docstrings or Field descriptions inside the LLM-facing
# Pydantic models — all guidance lives in the system prompt + the
# wrapper-level Field.description.

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, conlist, constr

from schemas.endpoint_parameters import (
    POLARITY_DESCRIPTION,
    ROLE_DESCRIPTION,
    EndpointParameters,
)
from schemas.enums import Polarity, ReleaseFormat, Role


# Subset of ReleaseFormat the LLM is allowed to emit. See the file-
# level comment for why MOVIE and UNKNOWN are both excluded.
_LLM_FACING_RELEASE_FORMAT = Literal[
    ReleaseFormat.TV_MOVIE,
    ReleaseFormat.SHORT,
    ReleaseFormat.VIDEO,
]


# Step 3 media-type endpoint output.
#
# Single matching path:
#   formats — non-empty list of ReleaseFormat members. Executor maps
#             each member's int `release_format_id` attribute to the
#             SMALLINT column on public.movie_card.release_format and
#             runs a single ANY(...) lookup. Flat 1.0 score per match
#             (no prominence signal on a movie's media type).
#
# Schema ordering is load-bearing. `thinking` is first so the LLM
# names which non-default formats the user wants and why, before
# committing to the enum members in `formats`.
class MediaTypeQuerySpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Scoped reasoning field. One or two sentences naming which
    # non-default media formats the phrasing requests (e.g. "TV
    # movies", "shorts", "direct-to-video") and the corresponding
    # ReleaseFormat members. LLM-facing guidance lives in the system
    # prompt.
    thinking: constr(strip_whitespace=True, min_length=1) = Field(...)

    # Non-empty list of ReleaseFormat members the user wants. At most
    # 3 (the size of the LLM-facing subset). MOVIE and UNKNOWN are
    # both excluded by the Literal type — see the file-level comment.
    formats: conlist(
        _LLM_FACING_RELEASE_FORMAT,
        min_length=1,
    ) = Field(...)


# Category-handler wrapper. Direction flows through role +
# polarity on the wrapper. Fields are declared in the order
# role → parameters → polarity so polarity is emitted last.
# See endpoint_parameters.py for the rationale.
class MediaTypeEndpointParameters(EndpointParameters):
    role: Role = Field(..., description=ROLE_DESCRIPTION)
    parameters: MediaTypeQuerySpec = Field(...)
    polarity: Polarity = Field(..., description=POLARITY_DESCRIPTION)

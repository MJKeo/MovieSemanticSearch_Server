from pydantic import BaseModel, ConfigDict, Field


class ScoredCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    movie_id: int = Field(..., ge=1)
    score: float = Field(..., ge=0.0, le=1.0)


class EndpointResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scores: list[ScoredCandidate] = Field(default_factory=list)

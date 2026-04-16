"""
Unified classification registry for the step 3 keyword endpoint.

Merges three source enums into one namespace the step 3 LLM selects
from: OverallKeyword (225), SourceMaterialType (10), and the seven
ConceptTag category enums aggregated via ALL_CONCEPT_TAGS (25).

The LLM returns one UnifiedClassification member. Execution code uses
entry_for(member) to get the originating source and its ID so it can
issue a GIN `&&` overlap query against the correct movie_card array
column (keyword_ids / source_material_type_ids / concept_tag_ids).

Collision rule: OverallKeyword is iterated first, so any name that
also appears in SourceMaterialType or ConceptTag is dropped from the
latter. OverallKeyword has broader coverage and is the stronger
retrieval signal, so it wins by design. In the current vocabulary the
only real collision is BIOGRAPHY (OverallKeyword vs SourceMaterialType).
"""

from dataclasses import dataclass
from enum import StrEnum

from implementation.classes.overall_keywords import OverallKeyword
from schemas.enums import ALL_CONCEPT_TAGS, SourceMaterialType


class ClassificationSource(StrEnum):
    KEYWORD = "keyword"
    SOURCE_MATERIAL = "source_material"
    CONCEPT_TAG = "concept_tag"


# movie_card array column each source resolves to at query time.
_BACKING_COLUMN: dict[ClassificationSource, str] = {
    ClassificationSource.KEYWORD: "keyword_ids",
    ClassificationSource.SOURCE_MATERIAL: "source_material_type_ids",
    ClassificationSource.CONCEPT_TAG: "concept_tag_ids",
}


# SourceMaterialType carries no display or definition column on its
# members — both are hand-written here so the step 3 LLM prompt can
# present each option with a clean display label (acronyms preserved,
# e.g. "TV Adaptation") plus a meaning-based definition. Keyed by
# SourceMaterialType member name; value is (display, definition).
_SOURCE_MATERIAL_METADATA: dict[str, tuple[str, str]] = {
    "NOVEL_ADAPTATION": (
        "Novel Adaptation",
        "A movie adapted from a novel or full-length book.",
    ),
    "SHORT_STORY_ADAPTATION": (
        "Short Story Adaptation",
        "A movie adapted from a short story or novella.",
    ),
    "STAGE_ADAPTATION": (
        "Stage Adaptation",
        "A movie adapted from a stage play, musical, or other theatrical work.",
    ),
    "TRUE_STORY": (
        "True Story",
        "A movie based on real events or real people, dramatizing actual history without focusing on one person's full life.",
    ),
    "COMIC_ADAPTATION": (
        "Comic Adaptation",
        "A movie adapted from a comic book, graphic novel, or manga.",
    ),
    "FOLKLORE_ADAPTATION": (
        "Folklore Adaptation",
        "A movie adapted from folklore, mythology, fairy tales, or legends.",
    ),
    "VIDEO_GAME_ADAPTATION": (
        "Video Game Adaptation",
        "A movie adapted from a video game.",
    ),
    "REMAKE": (
        "Remake",
        "A movie that retells the story of an earlier film — a new version of a previously made movie.",
    ),
    "TV_ADAPTATION": (
        "TV Adaptation",
        "A movie adapted from a television series.",
    ),
}


@dataclass(frozen=True)
class ClassificationEntry:
    name: str
    display: str
    definition: str
    source: ClassificationSource
    source_id: int

    @property
    def backing_column(self) -> str:
        return _BACKING_COLUMN[self.source]


def _build_registry() -> dict[str, ClassificationEntry]:
    registry: dict[str, ClassificationEntry] = {}

    for kw in OverallKeyword:
        registry[kw.name] = ClassificationEntry(
            name=kw.name,
            display=kw.value,
            definition=kw.definition,
            source=ClassificationSource.KEYWORD,
            source_id=kw.keyword_id,
        )

    for smt in SourceMaterialType:
        if smt.name in registry:
            continue
        metadata = _SOURCE_MATERIAL_METADATA.get(smt.name)
        if metadata is None:
            raise RuntimeError(
                f"Missing _SOURCE_MATERIAL_METADATA entry for SourceMaterialType.{smt.name}"
            )
        display, definition = metadata
        registry[smt.name] = ClassificationEntry(
            name=smt.name,
            display=display,
            definition=definition,
            source=ClassificationSource.SOURCE_MATERIAL,
            source_id=smt.source_material_type_id,
        )

    for tag in ALL_CONCEPT_TAGS:
        if tag.name in registry:
            continue
        registry[tag.name] = ClassificationEntry(
            name=tag.name,
            display=tag.name.replace("_", " ").title(),
            definition=tag.description,
            source=ClassificationSource.CONCEPT_TAG,
            source_id=tag.concept_tag_id,
        )

    return registry


CLASSIFICATION_ENTRIES: dict[str, ClassificationEntry] = _build_registry()


# Dynamically built StrEnum covering every registry key. Used as the
# structured-output type for the step 3 keyword LLM — Pydantic emits a
# finite JSON-schema enum constraint with every valid choice. Member
# value equals member name so prompt text and wire JSON match.
UnifiedClassification = StrEnum(
    "UnifiedClassification",
    {name: name for name in CLASSIFICATION_ENTRIES},
)


def entry_for(member: UnifiedClassification) -> ClassificationEntry:
    return CLASSIFICATION_ENTRIES[member.value]

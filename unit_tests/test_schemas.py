"""Unit tests for schema validators, __str__ methods, and Pydantic constraint enforcement."""

import pytest

from implementation.classes.enums import WatchMethodType
from implementation.classes.schemas import (
    WatchProvider,
    MajorCharacter,
    PlotEventsMetadata,
    CoreConcept,
    CharacterArc,
    MajorTheme,
    MajorLessonLearned,
    PlotAnalysisMetadata,
    ViewerExperienceSection,
    OptionalViewerExperienceSection,
    ViewerExperienceMetadata,
    GenericTermsSection,
    WatchContextMetadata,
    NarrativeTechniquesMetadata,
    SourceOfInspirationSection,
    ProductionMetadata,
    IMDBReviewTheme,
    DatePreferenceResult,
)


def _base_provider_payload() -> dict:
    """Return a valid baseline payload for constructing WatchProvider."""
    return {
        "id": 7,
        "name": "Prime Video",
        "logo_path": "/prime.png",
        "display_priority": 2,
        "types": [],
    }


def test_watch_provider_parse_types_from_strings() -> None:
    """WatchProvider should parse valid string watch method types."""
    payload = _base_provider_payload()
    payload["types"] = ["subscription", "purchase", "rent"]
    provider = WatchProvider(**payload)
    assert provider.types == [1, 2, 3]


def test_watch_provider_parse_types_from_enum_members() -> None:
    """WatchProvider should keep enum members when already provided."""
    payload = _base_provider_payload()
    payload["types"] = [WatchMethodType.SUBSCRIPTION, WatchMethodType.RENT]
    provider = WatchProvider(**payload)
    assert provider.types == [1, 3]


def test_watch_provider_parse_types_from_ints() -> None:
    """WatchProvider should parse valid integer enum values."""
    payload = _base_provider_payload()
    payload["types"] = [1, 2, 3]
    provider = WatchProvider(**payload)
    assert provider.types == [1, 2, 3]


def test_watch_provider_parse_types_mixed_and_invalid_entries_filtered() -> None:
    """WatchProvider should ignore invalid entries while preserving valid entries in order."""
    payload = _base_provider_payload()
    payload["types"] = ["subscription", "invalid", 2, WatchMethodType.RENT, 999]
    provider = WatchProvider(**payload)
    assert provider.types == [1, 2, 3]


@pytest.mark.parametrize("non_list_value", [None, "subscription", 1, {"a": 1}])
def test_watch_provider_parse_types_non_list_becomes_empty(non_list_value: object) -> None:
    """WatchProvider should convert non-list type inputs into an empty list."""
    payload = _base_provider_payload()
    payload["types"] = non_list_value
    provider = WatchProvider(**payload)
    assert provider.types == []


# ================================
#    __str__ method tests
# ================================


def test_major_character_str_formats_name_description_motivations() -> None:
    """MajorCharacter.__str__ should include name, description, and motivations."""
    char = MajorCharacter(
        name="Peter Parker",
        description="A young student with powers",
        role="protagonist",
        primary_motivations="Protect the city.",
    )
    assert str(char) == "Peter Parker: A young student with powers Motivations: Protect the city."


def test_plot_events_metadata_str_joins_all_parts() -> None:
    """PlotEventsMetadata.__str__ should join lowercased summary, setting, and characters."""
    metadata = PlotEventsMetadata(
        plot_summary="A Hero Saves the World.",
        setting="New York City, 2002",
        major_characters=[
            MajorCharacter(
                name="Peter",
                description="A student",
                role="protagonist",
                primary_motivations="Save the city.",
            ),
        ],
    )
    result = str(metadata)
    assert "a hero saves the world." in result
    assert "new york city, 2002" in result
    assert "peter: a student motivations: save the city." in result


def test_plot_events_metadata_str_without_characters() -> None:
    """PlotEventsMetadata.__str__ should omit character section when list is empty."""
    metadata = PlotEventsMetadata(
        plot_summary="A Hero Saves the World.",
        setting="New York City",
        major_characters=[],
    )
    result = str(metadata)
    assert "a hero saves the world." in result
    assert "new york city" in result


def test_core_concept_str() -> None:
    """CoreConcept.__str__ should format label and explanation."""
    cc = CoreConcept(
        explanation_and_justification="Because it explores identity.",
        core_concept_label="Identity Crisis",
    )
    assert str(cc) == "Identity Crisis: Because it explores identity."


def test_character_arc_str_returns_label() -> None:
    """CharacterArc.__str__ should return arc_transformation_label."""
    arc = CharacterArc(
        character_name="Peter Parker",
        arc_transformation_description="From naive student to responsible hero.",
        arc_transformation_label="Coming of age",
    )
    assert str(arc) == "Coming of age"


def test_major_theme_str_returns_label() -> None:
    """MajorTheme.__str__ should return theme_label."""
    theme = MajorTheme(
        explanation_and_justification="Power requires responsibility.",
        theme_label="Responsibility and power",
    )
    assert str(theme) == "Responsibility and power"


def test_major_lesson_learned_str_returns_label() -> None:
    """MajorLessonLearned.__str__ should return lesson_label."""
    lesson = MajorLessonLearned(
        explanation_and_justification="The hero learns to be selfless.",
        lesson_label="Selflessness over selfishness",
    )
    assert str(lesson) == "Selflessness over selfishness"


def test_plot_analysis_metadata_str_combines_all_components() -> None:
    """PlotAnalysisMetadata.__str__ should combine overview, concept, genres, arcs, themes, and lessons."""
    metadata = PlotAnalysisMetadata(
        core_concept=CoreConcept(
            explanation_and_justification="Identity.",
            core_concept_label="Identity",
        ),
        genre_signatures=["superhero origin", "coming of age"],
        conflict_scale="Personal",
        character_arcs=[
            CharacterArc(
                character_name="Peter",
                arc_transformation_description="Grows up.",
                arc_transformation_label="maturation",
            ),
        ],
        themes_primary=[
            MajorTheme(
                explanation_and_justification="Power.",
                theme_label="responsibility",
            ),
        ],
        lessons_learned=[
            MajorLessonLearned(
                explanation_and_justification="Be selfless.",
                lesson_label="selflessness",
            ),
        ],
        generalized_plot_overview="A student gains powers and learns responsibility.",
    )
    result = str(metadata)
    assert "a student gains powers and learns responsibility." in result
    assert "identity: identity." in result
    assert "superhero origin, coming of age" in result
    assert "personal conflict" in result
    assert "maturation" in result
    assert "responsibility" in result
    assert "selflessness" in result


def test_viewer_experience_metadata_str_skips_flagged_sections() -> None:
    """ViewerExperienceMetadata.__str__ should include non-skipped and exclude skipped sections."""
    base_section = ViewerExperienceSection(justification="J.", terms=["tense"], negations=["not boring"])
    empty_section = ViewerExperienceSection(justification="J.", terms=[], negations=[])
    included = OptionalViewerExperienceSection(
        should_skip=False,
        section_data=ViewerExperienceSection(justification="R.", terms=["disturbing"], negations=[]),
    )
    skipped = OptionalViewerExperienceSection(
        should_skip=True,
        section_data=ViewerExperienceSection(justification="Skip.", terms=["should not appear"], negations=["also skipped"]),
    )
    metadata = ViewerExperienceMetadata(
        emotional_palette=base_section,
        tension_adrenaline=ViewerExperienceSection(justification="J.", terms=["high tension"], negations=[]),
        tone_self_seriousness=empty_section,
        cognitive_complexity=empty_section,
        disturbance_profile=included,
        sensory_load=skipped,
        emotional_volatility=skipped,
        ending_aftertaste=ViewerExperienceSection(justification="J.", terms=["satisfying"], negations=[]),
    )
    result = str(metadata)
    assert "tense" in result
    assert "not boring" in result
    assert "high tension" in result
    assert "disturbing" in result
    assert "satisfying" in result
    assert "should not appear" not in result
    assert "also skipped" not in result


def test_watch_context_metadata_str_joins_all_terms() -> None:
    """WatchContextMetadata.__str__ should join terms from all four sections."""
    metadata = WatchContextMetadata(
        self_experience_motivations=GenericTermsSection(justification="J.", terms=["escape"]),
        external_motivations=GenericTermsSection(justification="J.", terms=["family night"]),
        key_movie_feature_draws=GenericTermsSection(justification="J.", terms=["visual effects"]),
        watch_scenarios=GenericTermsSection(justification="J.", terms=["rainy day"]),
    )
    assert str(metadata) == "escape, family night, visual effects, rainy day"


def test_narrative_techniques_metadata_str_combines_sections() -> None:
    """NarrativeTechniquesMetadata.__str__ should combine terms from all 11 sections."""
    empty = GenericTermsSection(justification="J.", terms=[])
    metadata = NarrativeTechniquesMetadata(
        pov_perspective=GenericTermsSection(justification="J.", terms=["first person"]),
        narrative_delivery=empty,
        narrative_archetype=empty,
        information_control=empty,
        characterization_methods=empty,
        character_arcs=empty,
        audience_character_perception=empty,
        conflict_stakes_design=empty,
        thematic_delivery=empty,
        meta_techniques=empty,
        additional_plot_devices=GenericTermsSection(justification="J.", terms=["flashback"]),
    )
    assert str(metadata) == "first person, flashback"


def test_production_metadata_str_combines_all_sources() -> None:
    """ProductionMetadata.__str__ should combine keywords, inspirations, and mediums."""
    metadata = ProductionMetadata(
        production_keywords=GenericTermsSection(justification="J.", terms=["practical effects"]),
        sources_of_inspiration=SourceOfInspirationSection(
            justification="J.",
            sources_of_inspiration=["comic books"],
            production_mediums=["live action"],
        ),
    )
    assert str(metadata) == "practical effects, comic books, live action"


def test_imdb_review_theme_str_formats_attribute_and_sentiment() -> None:
    """IMDBReviewTheme.__str__ should format name and sentiment."""
    theme = IMDBReviewTheme(name="acting", sentiment="positive")
    assert str(theme) == "Attribute: acting; audience reception: positive"


# ================================
#  Pydantic validation tests
# ================================


def test_major_character_rejects_empty_name() -> None:
    """MajorCharacter should reject empty name due to min_length=1 constraint."""
    with pytest.raises(Exception):
        MajorCharacter(
            name="",
            description="A student",
            role="protagonist",
            primary_motivations="Save the city.",
        )


def test_major_character_rejects_whitespace_only_name() -> None:
    """MajorCharacter should reject whitespace-only name (strip + min_length=1)."""
    with pytest.raises(Exception):
        MajorCharacter(
            name="   ",
            description="A student",
            role="protagonist",
            primary_motivations="Save the city.",
        )


def test_plot_analysis_rejects_fewer_than_two_genre_signatures() -> None:
    """PlotAnalysisMetadata should reject fewer than 2 genre_signatures (conlist min)."""
    with pytest.raises(Exception):
        PlotAnalysisMetadata(
            core_concept=CoreConcept(explanation_and_justification="X.", core_concept_label="X"),
            genre_signatures=["only one"],
            conflict_scale="Personal",
            character_arcs=[
                CharacterArc(character_name="P", arc_transformation_description="G.", arc_transformation_label="m"),
            ],
            themes_primary=[MajorTheme(explanation_and_justification="P.", theme_label="r")],
            generalized_plot_overview="A student gains powers.",
        )


def test_plot_analysis_rejects_more_than_six_genre_signatures() -> None:
    """PlotAnalysisMetadata should reject more than 6 genre_signatures (conlist max)."""
    with pytest.raises(Exception):
        PlotAnalysisMetadata(
            core_concept=CoreConcept(explanation_and_justification="X.", core_concept_label="X"),
            genre_signatures=["a", "b", "c", "d", "e", "f", "g"],
            conflict_scale="Personal",
            character_arcs=[
                CharacterArc(character_name="P", arc_transformation_description="G.", arc_transformation_label="m"),
            ],
            themes_primary=[MajorTheme(explanation_and_justification="P.", theme_label="r")],
            generalized_plot_overview="A student gains powers.",
        )


def test_viewer_experience_section_rejects_extra_fields() -> None:
    """ViewerExperienceSection should reject unexpected fields (extra='forbid')."""
    with pytest.raises(Exception):
        ViewerExperienceSection(
            justification="Because.",
            terms=["tense"],
            negations=[],
            unknown_field="should fail",
        )


def test_date_preference_result_rejects_invalid_date_format() -> None:
    """DatePreferenceResult should reject dates not matching YYYY-MM-DD pattern."""
    with pytest.raises(Exception):
        DatePreferenceResult(
            first_date="2024/01/01",
            match_operation="exact",
        )


def test_date_preference_result_accepts_valid_date() -> None:
    """DatePreferenceResult should accept properly formatted ISO dates."""
    result = DatePreferenceResult(first_date="2024-01-15", match_operation="exact")
    assert result.first_date == "2024-01-15"

"""
Unit tests for movie_ingestion.metadata_generation.schemas.

Covers:
  - ReceptionOutput.__str__() excludes review_insights_brief
"""

from movie_ingestion.metadata_generation.schemas import ReceptionOutput


class TestReceptionOutputStr:
    def test_reception_output_str_excludes_review_insights_brief(self):
        output = ReceptionOutput(
            reception_summary="Widely acclaimed for visual effects.",
            praise_attributes=["groundbreaking visuals"],
            complaint_attributes=["thin plot"],
            review_insights_brief="Critics noted philosophical depth and innovative cinematography.",
        )
        result = str(output)

        # The review_insights_brief text should NOT appear in str() output
        assert "philosophical depth" not in result
        assert "innovative cinematography" not in result
        assert "critics noted" not in result.lower()

        # But the other fields SHOULD appear (lowercased)
        assert "widely acclaimed" in result
        assert "groundbreaking visuals" in result
        assert "thin plot" in result

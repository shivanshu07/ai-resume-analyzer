from typing import Any, Dict, List

from src.evaluation.ats_scorer import ATSScorer
from src.evaluation.gap_analyzer import ResumeGapAnalyzer


class ResumeAnalysisEngine:
    """
    Main Day-6 analysis engine.

    Combines:

        1. ATS scoring
        2. Requirement summaries
        3. Skill analysis
        4. Concept analysis
        5. Education analysis
        6. Experience analysis
        7. Strength identification
        8. Gap identification
        9. Priority gap analysis
    """

    def __init__(self):

        self.ats_scorer = ATSScorer()

        self.gap_analyzer = (
            ResumeGapAnalyzer()
        )

    # ========================================================
    # ANALYZE
    # ========================================================

    def analyze(
        self,
        hybrid_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:

        if not isinstance(
            hybrid_results,
            list
        ):

            raise TypeError(
                "hybrid_results must be a list."
            )

        # ----------------------------------------------------
        # ATS scoring
        # ----------------------------------------------------

        ats_analysis = (
            self.ats_scorer.generate_analysis(
                hybrid_results
            )
        )

        # ----------------------------------------------------
        # Gap analysis
        # ----------------------------------------------------

        gap_analysis = (
            self.gap_analyzer.analyze(
                hybrid_results
            )
        )

        # ----------------------------------------------------
        # Final output
        # ----------------------------------------------------

        return {
            "overall_ats_score": (
                ats_analysis[
                    "overall_ats_score"
                ]
            ),

            "score_interpretation": (
                ats_analysis[
                    "score_interpretation"
                ]
            ),

            "requirement_summary": (
                ats_analysis[
                    "requirement_summary"
                ]
            ),

            "category_summary": (
                ats_analysis[
                    "category_summary"
                ]
            ),

            "skills": (
                gap_analysis[
                    "skills"
                ]
            ),

            "concepts": (
                gap_analysis[
                    "concepts"
                ]
            ),

            "education": (
                gap_analysis[
                    "education"
                ]
            ),

            "experience": (
                gap_analysis[
                    "experience"
                ]
            ),

            "requirements": (
                gap_analysis[
                    "requirements"
                ]
            ),

            "priority_gaps": (
                gap_analysis[
                    "priority_gaps"
                ]
            )
        }
from typing import Any, Dict, List


class ATSScorer:
    """
    Calculates an ATS-style overall score.

    This is a project-specific scoring system and should
    not be interpreted as an actual company's ATS formula.
    """

    IMPORTANCE_WEIGHTS = {
        "high": 2.00,
        "medium": 1.00,
        "low": 0.50
    }

    CATEGORY_WEIGHTS = {
        "required": 1.50,
        "preferred": 1.00,
        "responsibility": 0.75,
        "education": 1.25,
        "other": 0.50
    }

    def get_hybrid_score(
        self,
        result: Dict[str, Any]
    ) -> float:

        value = result.get(
            "hybrid_score",
            0.0
        )

        try:
            score = float(value)
        except (
            TypeError,
            ValueError
        ):
            score = 0.0

        if score > 1.0:
            score /= 100.0

        return max(
            0.0,
            min(
                score,
                1.0
            )
        )

    def get_weight(
        self,
        result: Dict[str, Any]
    ) -> float:

        importance = str(
            result.get(
                "importance",
                "medium"
            )
        ).lower()

        category = str(
            result.get(
                "category",
                "other"
            )
        ).lower()

        importance_weight = (
            self.IMPORTANCE_WEIGHTS.get(
                importance,
                1.0
            )
        )

        category_weight = (
            self.CATEGORY_WEIGHTS.get(
                category,
                0.5
            )
        )

        return (
            importance_weight
            *
            category_weight
        )

    def calculate_score(
        self,
        results: List[Dict[str, Any]]
    ) -> float:

        if not results:
            return 0.0

        numerator = 0.0
        denominator = 0.0

        for result in results:

            if not isinstance(
                result,
                dict
            ):
                continue

            score = self.get_hybrid_score(
                result
            )

            weight = self.get_weight(
                result
            )

            numerator += (
                score * weight
            )

            denominator += weight

        if denominator == 0:
            return 0.0

        return round(
            (
                numerator
                /
                denominator
            )
            * 100,
            2
        )

    def summarize_assessments(
        self,
        results: List[Dict[str, Any]]
    ) -> Dict[str, int]:

        summary = {
            "total": 0,
            "strong": 0,
            "partial": 0,
            "weak": 0,
            "no_alignment": 0
        }

        for result in results:

            if not isinstance(
                result,
                dict
            ):
                continue

            summary["total"] += 1

            assessment = str(
                result.get(
                    "assessment",
                    ""
                )
            ).upper()

            if assessment == "STRONG_ALIGNMENT":
                summary["strong"] += 1

            elif assessment == "PARTIAL_ALIGNMENT":
                summary["partial"] += 1

            elif assessment == "WEAK_ALIGNMENT":
                summary["weak"] += 1

            elif assessment == "NO_ALIGNMENT":
                summary["no_alignment"] += 1

        return summary

    def summarize_categories(
        self,
        results: List[Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:

        grouped = {}

        for result in results:

            category = str(
                result.get(
                    "category",
                    "other"
                )
            ).lower()

            grouped.setdefault(
                category,
                []
            ).append(
                result
            )

        output = {}

        for category, items in grouped.items():

            output[category] = {
                "score": self.calculate_score(
                    items
                ),
                "summary": self.summarize_assessments(
                    items
                )
            }

        return output

    def interpret_score(
        self,
        score: float
    ) -> str:

        if score >= 80:
            return "Strong Match"

        if score >= 65:
            return "Good Match"

        if score >= 50:
            return "Moderate Match"

        if score >= 35:
            return "Weak Match"

        return "Low Match"

    def generate_analysis(
        self,
        results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:

        score = self.calculate_score(
            results
        )

        return {
            "overall_ats_score": score,

            "score_interpretation": (
                self.interpret_score(
                    score
                )
            ),

            "requirement_summary": (
                self.summarize_assessments(
                    results
                )
            ),

            "category_summary": (
                self.summarize_categories(
                    results
                )
            )
        }
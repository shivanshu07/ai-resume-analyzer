from typing import Any, Dict, List


class ResumeGapAnalyzer:
    """
    Converts requirement-level matching results into:

        - skill matches/gaps
        - concept matches/gaps
        - education analysis
        - experience analysis
        - requirement summaries
        - priority gaps
    """

    @staticmethod
    def as_list(
        value: Any
    ) -> List[str]:

        if value is None:
            return []

        if isinstance(
            value,
            list
        ):

            return [
                str(item).strip()
                for item in value
                if str(item).strip()
            ]

        if isinstance(
            value,
            str
        ):

            value = value.strip()

            if not value:
                return []

            return [value]

        return [
            str(value).strip()
        ]

    @staticmethod
    def unique(
        values: List[str]
    ) -> List[str]:

        seen = set()
        output = []

        for value in values:

            value = str(
                value
            ).strip()

            if not value:
                continue

            key = value.lower()

            if key not in seen:

                seen.add(key)
                output.append(value)

        return output

    # ========================================================
    # SKILLS
    # ========================================================

    def analyze_skills(
        self,
        results: List[Dict[str, Any]]
    ) -> Dict[str, List[str]]:

        matched = []
        missing = []

        for result in results:

            data = result.get(
                "skill_match",
                {}
            )

            if not isinstance(
                data,
                dict
            ):
                continue

            matched.extend(
                self.as_list(
                    data.get(
                        "matched"
                    )
                )
            )

            missing.extend(
                self.as_list(
                    data.get(
                        "missing"
                    )
                )
            )

        matched = self.unique(
            matched
        )

        missing = self.unique(
            missing
        )

        matched_lower = {
            item.lower()
            for item in matched
        }

        missing = [
            item
            for item in missing
            if item.lower()
            not in matched_lower
        ]

        return {
            "matched": matched,
            "missing": missing
        }

    # ========================================================
    # CONCEPTS
    # ========================================================

    def analyze_concepts(
        self,
        results: List[Dict[str, Any]]
    ) -> Dict[str, List[str]]:

        matched = []
        missing = []

        for result in results:

            data = result.get(
                "concept_match",
                {}
            )

            if not isinstance(
                data,
                dict
            ):
                continue

            matched.extend(
                self.as_list(
                    data.get(
                        "matched"
                    )
                )
            )

            missing.extend(
                self.as_list(
                    data.get(
                        "missing"
                    )
                )
            )

        matched = self.unique(
            matched
        )

        missing = self.unique(
            missing
        )

        matched_lower = {
            item.lower()
            for item in matched
        }

        missing = [
            item
            for item in missing
            if item.lower()
            not in matched_lower
        ]

        return {
            "matched": matched,
            "missing": missing
        }

    # ========================================================
    # EDUCATION
    # ========================================================

    def analyze_education(
        self,
        results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:

        matched = []
        related = []
        missing = []
        degree_levels = []

        for result in results:

            data = result.get(
                "education_match",
                {}
            )

            if not isinstance(
                data,
                dict
            ):
                continue

            matched.extend(
                self.as_list(
                    data.get(
                        "matched"
                    )
                )
            )

            related.extend(
                self.as_list(
                    data.get(
                        "related"
                    )
                )
            )

            missing.extend(
                self.as_list(
                    data.get(
                        "missing"
                    )
                )
            )

            level = data.get(
                "degree_level"
            )

            if level:
                degree_levels.append(
                    str(level)
                )

        return {
            "matched": self.unique(
                matched
            ),
            "related": self.unique(
                related
            ),
            "missing": self.unique(
                missing
            ),
            "degree_levels": self.unique(
                degree_levels
            )
        }

    # ========================================================
    # EXPERIENCE
    # ========================================================

    def analyze_experience(
        self,
        results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:

        required = []
        evidence = []
        estimated = []

        for result in results:

            data = result.get(
                "experience_match",
                {}
            )

            if not isinstance(
                data,
                dict
            ):
                continue

            required.extend(
                self.as_list(
                    data.get(
                        "required"
                    )
                )
            )

            evidence.extend(
                self.as_list(
                    data.get(
                        "evidence"
                    )
                )
            )

            years = data.get(
                "estimated_years"
            )

            if isinstance(
                years,
                (int, float)
            ):
                estimated.append(
                    float(years)
                )

        return {
            "required": self.unique(
                required
            ),
            "evidence": self.unique(
                evidence
            ),
            "estimated_years": (
                max(estimated)
                if estimated
                else 0.0
            )
        }

    # ========================================================
    # REQUIREMENTS
    # ========================================================

    def analyze_requirements(
        self,
        results: List[Dict[str, Any]]
    ) -> Dict[str, List[Dict[str, Any]]]:

        strengths = []
        partial = []
        weak = []
        missing = []

        for result in results:

            assessment = str(
                result.get(
                    "assessment",
                    ""
                )
            ).upper()

            item = {
                "requirement_id": result.get(
                    "requirement_id"
                ),
                "requirement": result.get(
                    "requirement",
                    ""
                ),
                "category": result.get(
                    "category",
                    "other"
                ),
                "importance": result.get(
                    "importance",
                    "medium"
                ),
                "hybrid_score": float(
                    result.get(
                        "hybrid_score",
                        0.0
                    )
                ),
                "best_evidence": result.get(
                    "best_evidence",
                    {}
                )
            }

            if assessment == "STRONG_ALIGNMENT":
                strengths.append(item)

            elif assessment == "PARTIAL_ALIGNMENT":
                partial.append(item)

            elif assessment == "WEAK_ALIGNMENT":
                weak.append(item)

            elif assessment == "NO_ALIGNMENT":
                missing.append(item)

        return {
            "strengths": strengths,
            "partial_matches": partial,
            "weak_matches": weak,
            "missing_requirements": missing
        }

    # ========================================================
    # PRIORITY GAPS
    # ========================================================

    def identify_priority_gaps(
        self,
        results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:

        importance_weight = {
            "high": 3.0,
            "medium": 2.0,
            "low": 1.0
        }

        assessment_weight = {
            "NO_ALIGNMENT": 3.0,
            "WEAK_ALIGNMENT": 2.0,
            "PARTIAL_ALIGNMENT": 1.0,
            "STRONG_ALIGNMENT": 0.0
        }

        gaps = []

        for result in results:

            assessment = str(
                result.get(
                    "assessment",
                    ""
                )
            ).upper()

            if assessment == "STRONG_ALIGNMENT":
                continue

            importance = str(
                result.get(
                    "importance",
                    "medium"
                )
            ).lower()

            score = float(
                result.get(
                    "hybrid_score",
                    0.0
                )
            )

            # Explicitly use hybrid_score.
            # This prevents the old KeyError.
            priority_score = (
                importance_weight.get(
                    importance,
                    2.0
                )
                *
                10.0
                +
                assessment_weight.get(
                    assessment,
                    1.0
                )
                *
                5.0
                +
                (
                    1.0
                    -
                    max(
                        0.0,
                        min(
                            score,
                            1.0
                        )
                    )
                )
                *
                5.0
            )

            gaps.append(
                {
                    "requirement_id": result.get(
                        "requirement_id"
                    ),
                    "requirement": result.get(
                        "requirement",
                        ""
                    ),
                    "category": result.get(
                        "category",
                        "other"
                    ),
                    "importance": importance,
                    "assessment": assessment,
                    "hybrid_score": round(
                        score,
                        4
                    ),
                    "priority_score": round(
                        priority_score,
                        2
                    )
                }
            )

        gaps.sort(
            key=lambda item: item[
                "priority_score"
            ],
            reverse=True
        )

        return gaps

    # ========================================================
    # FULL ANALYSIS
    # ========================================================

    def analyze(
        self,
        results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:

        return {
            "skills": self.analyze_skills(
                results
            ),

            "concepts": self.analyze_concepts(
                results
            ),

            "education": self.analyze_education(
                results
            ),

            "experience": self.analyze_experience(
                results
            ),

            "requirements": self.analyze_requirements(
                results
            ),

            "priority_gaps": self.identify_priority_gaps(
                results
            )
        }
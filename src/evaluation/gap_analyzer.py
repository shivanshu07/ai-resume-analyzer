from typing import Any, Dict, List


class ResumeGapAnalyzer:
    """
    Analyzes Day-5 hybrid matching results and converts
    requirement-level evidence into candidate-level
    strengths and gaps.
    """

    def __init__(self):
        pass

    # ========================================================
    # SAFE LIST CONVERSION
    # ========================================================

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

    # ========================================================
    # UNIQUE VALUES
    # ========================================================

    @staticmethod
    def unique(
        values: List[str]
    ) -> List[str]:

        seen = set()

        output = []

        for value in values:

            cleaned = str(
                value
            ).strip()

            if not cleaned:
                continue

            key = cleaned.lower()

            if key not in seen:

                seen.add(key)

                output.append(
                    cleaned
                )

        return output

    # ========================================================
    # SKILL ANALYSIS
    # ========================================================

    def analyze_skills(
        self,
        results: List[Dict[str, Any]]
    ) -> Dict[str, List[str]]:

        matched = []

        missing = []

        for result in results:

            skill_match = result.get(
                "skill_match",
                {}
            )

            if not isinstance(
                skill_match,
                dict
            ):
                continue

            matched.extend(
                self.as_list(
                    skill_match.get(
                        "matched"
                    )
                )
            )

            missing.extend(
                self.as_list(
                    skill_match.get(
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

        # ----------------------------------------------------
        # If something appears in both lists, matched evidence
        # takes precedence.
        # ----------------------------------------------------

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
    # CONCEPT ANALYSIS
    # ========================================================

    def analyze_concepts(
        self,
        results: List[Dict[str, Any]]
    ) -> Dict[str, List[str]]:

        matched = []

        missing = []

        for result in results:

            concept_match = result.get(
                "concept_match",
                {}
            )

            if not isinstance(
                concept_match,
                dict
            ):
                continue

            matched.extend(
                self.as_list(
                    concept_match.get(
                        "matched"
                    )
                )
            )

            missing.extend(
                self.as_list(
                    concept_match.get(
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
    # EDUCATION ANALYSIS
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

            education = result.get(
                "education_match",
                {}
            )

            if not isinstance(
                education,
                dict
            ):
                continue

            matched.extend(
                self.as_list(
                    education.get(
                        "matched"
                    )
                )
            )

            related.extend(
                self.as_list(
                    education.get(
                        "related"
                    )
                )
            )

            missing.extend(
                self.as_list(
                    education.get(
                        "missing"
                    )
                )
            )

            degree_level = education.get(
                "degree_level"
            )

            if degree_level:

                degree_levels.append(
                    str(
                        degree_level
                    ).strip()
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
    # EXPERIENCE ANALYSIS
    # ========================================================

    def analyze_experience(
        self,
        results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:

        required = []

        evidence = []

        estimated_years = []

        for result in results:

            experience = result.get(
                "experience_match",
                {}
            )

            if not isinstance(
                experience,
                dict
            ):
                continue

            required.extend(
                self.as_list(
                    experience.get(
                        "required"
                    )
                )
            )

            evidence.extend(
                self.as_list(
                    experience.get(
                        "evidence"
                    )
                )
            )

            years = experience.get(
                "estimated_years"
            )

            if isinstance(
                years,
                (int, float)
            ):

                estimated_years.append(
                    float(years)
                )

        maximum_estimated_years = (
            max(
                estimated_years
            )
            if estimated_years
            else 0.0
        )

        return {
            "required": self.unique(
                required
            ),
            "evidence": self.unique(
                evidence
            ),
            "estimated_years": (
                maximum_estimated_years
            )
        }

    # ========================================================
    # REQUIREMENT ANALYSIS
    # ========================================================

    def analyze_requirements(
        self,
        results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:

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
            ).strip().upper()

            requirement = result.get(
                "requirement",
                ""
            )

            requirement_id = result.get(
                "requirement_id"
            )

            category = result.get(
                "category",
                "other"
            )

            importance = result.get(
                "importance",
                "medium"
            )

            score = result.get(
                "hybrid_score",
                0.0
            )

            evidence = result.get(
                "best_evidence",
                {}
            )

            item = {
                "requirement_id": requirement_id,
                "requirement": requirement,
                "category": category,
                "importance": importance,
                "hybrid_score": score,
                "best_evidence": evidence
            }

            if assessment == "STRONG_ALIGNMENT":

                strengths.append(
                    item
                )

            elif assessment == "PARTIAL_ALIGNMENT":

                partial.append(
                    item
                )

            elif assessment == "WEAK_ALIGNMENT":

                weak.append(
                    item
                )

            elif assessment == "NO_ALIGNMENT":

                missing.append(
                    item
                )

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
        """
        Prioritizes gaps using:

            importance
            +
            assessment
            +
            hybrid score
        """

        importance_priority = {
            "high": 3,
            "medium": 2,
            "low": 1
        }

        assessment_priority = {
            "NO_ALIGNMENT": 3,
            "WEAK_ALIGNMENT": 2,
            "PARTIAL_ALIGNMENT": 1,
            "STRONG_ALIGNMENT": 0
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

            importance_score = (
                importance_priority.get(
                    importance,
                    2
                )
            )

            assessment_score = (
                assessment_priority.get(
                    assessment,
                    0
                )
            )

            hybrid_score = float(
                result.get(
                    "hybrid_score",
                    0.0
                )
            )

            priority_score = (
                importance_score * 10
                +
                assessment_score * 5
                +
                (1 - hybrid_score) * 5
            )

            gaps.append({
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
                "hybrid_score": hybrid_score,
                "priority_score": round(
                    priority_score,
                    2
                )
            })

        gaps.sort(
            key=lambda item: item[
                "priority_score"
            ],
            reverse=True
        )

        return gaps

    # ========================================================
    # FULL GAP ANALYSIS
    # ========================================================

    def analyze(
        self,
        results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:

        skills = self.analyze_skills(
            results
        )

        concepts = self.analyze_concepts(
            results
        )

        education = self.analyze_education(
            results
        )

        experience = self.analyze_experience(
            results
        )

        requirements = self.analyze_requirements(
            results
        )

        priority_gaps = self.identify_priority_gaps(
            results
        )

        return {
            "skills": skills,

            "concepts": concepts,

            "education": education,

            "experience": experience,

            "requirements": requirements,

            "priority_gaps": priority_gaps
        }
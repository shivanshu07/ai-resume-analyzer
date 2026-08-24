import re
from typing import Any, Dict, List


class HybridMatcher:
    """
    Requirement-aware hybrid resume/JD matcher.

    The scorer combines:

        1. Semantic similarity
        2. Explicit skill evidence
        3. Concept evidence
        4. Education evidence
        5. Experience evidence

    Important design principle:

        Semantic similarity can support a match,
        but it cannot replace explicit evidence
        when a requirement asks for a concrete skill,
        degree, experience level, or responsibility.

    Scores are normalized to [0, 1].
    """

    # ========================================================
    # NORMALIZATION
    # ========================================================

    @staticmethod
    def normalize(value: Any) -> str:

        return re.sub(
            r"[^a-z0-9+#.]",
            " ",
            str(value).lower()
        ).strip()

    @staticmethod
    def normalize_list(value: Any) -> List[str]:

        if value is None:
            return []

        if isinstance(value, list):
            return [
                str(item).strip()
                for item in value
                if str(item).strip()
            ]

        if isinstance(value, str):
            return [
                item.strip()
                for item in value.split(",")
                if item.strip()
            ]

        return [str(value).strip()]

    # ========================================================
    # TEXT MATCHING
    # ========================================================

    def text_contains(
        self,
        text: str,
        phrase: str
    ) -> bool:

        text_normalized = self.normalize(text)
        phrase_normalized = self.normalize(phrase)

        if not phrase_normalized:
            return False

        return phrase_normalized in text_normalized

    def find_skill_matches(
        self,
        required_skills: List[str],
        resume_text: str
    ) -> Dict[str, Any]:

        matched = []
        missing = []

        for skill in required_skills:

            if self.text_contains(
                resume_text,
                skill
            ):
                matched.append(skill)
            else:
                missing.append(skill)

        if not required_skills:

            score = 0.0

        else:

            score = (
                len(matched)
                /
                len(required_skills)
            )

        return {
            "score": round(score, 4),
            "matched": matched,
            "missing": missing
        }

    # ========================================================
    # CONCEPT MATCHING
    # ========================================================

    CONCEPT_ALIASES = {

        "business problems": [
            "business problem",
            "business problems",
            "business solution",
            "business solutions",
            "business requirements"
        ],

        "business insights": [
            "business insights",
            "insights",
            "data driven insights",
            "data-driven insights"
        ],

        "stakeholder management": [
            "stakeholder",
            "stakeholders",
            "stakeholder management",
            "cross functional",
            "cross-functional"
        ],

        "customer collaboration": [
            "customer",
            "customers",
            "client",
            "clients",
            "client collaboration"
        ],

        "client engagement": [
            "client",
            "clients",
            "client engagement",
            "customer engagement"
        ],

        "decision making": [
            "decision making",
            "decision-making",
            "decision support"
        ],

        "business processes": [
            "business process",
            "business processes",
            "process improvement",
            "workflow"
        ],

        "strategic insights": [
            "strategic insights",
            "strategic recommendations",
            "strategy"
        ],

        "proof of concept": [
            "proof of concept",
            "poc",
            "prototype",
            "prototyping"
        ],

        "product engineering collaboration": [
            "product team",
            "engineering team",
            "product/engineering",
            "cross functional",
            "cross-functional"
        ],

        "innovation": [
            "innovation",
            "innovative",
            "new solutions",
            "new approaches"
        ],

        "metrics": [
            "metrics",
            "metric",
            "kpi",
            "kpis"
        ],

        "data structures": [
            "data structure",
            "data structures",
            "database structure",
            "data architecture"
        ],

        "marketing effectiveness": [
            "marketing effectiveness",
            "marketing analytics",
            "marketing performance"
        ],

        "marketing portfolio management": [
            "marketing portfolio",
            "portfolio management"
        ]
    }

    def concept_is_present(
        self,
        concept: str,
        resume_text: str
    ) -> bool:

        normalized_concept = self.normalize(
            concept
        )

        aliases = self.CONCEPT_ALIASES.get(
            normalized_concept,
            [concept]
        )

        for alias in aliases:

            if self.text_contains(
                resume_text,
                alias
            ):
                return True

        return False

    def match_concepts(
        self,
        concepts: List[str],
        resume_text: str
    ) -> Dict[str, Any]:

        matched = []
        missing = []

        for concept in concepts:

            if self.concept_is_present(
                concept,
                resume_text
            ):
                matched.append(concept)
            else:
                missing.append(concept)

        if not concepts:

            score = 0.0

        else:

            score = (
                len(matched)
                /
                len(concepts)
            )

        return {
            "score": round(score, 4),
            "matched": matched,
            "missing": missing
        }

    # ========================================================
    # EDUCATION MATCHING
    # ========================================================

    EDUCATION_ALIASES = {

        "statistics": [
            "statistics",
            "statistical"
        ],

        "data science": [
            "data science",
            "data analytics"
        ],

        "mathematics": [
            "mathematics",
            "mathematics and computing",
            "mathematical"
        ],

        "physics": [
            "physics"
        ],

        "economics": [
            "economics",
            "econometrics"
        ],

        "operations research": [
            "operations research",
            "operations research and analytics"
        ],

        "engineering": [
            "engineering",
            "bachelor of technology",
            "b tech",
            "b.tech",
            "bachelor's technology"
        ],

        "bioinformatics": [
            "bioinformatics"
        ]
    }

    def detect_degree_level(
        self,
        text: str
    ) -> str:

        normalized = self.normalize(
            text
        )

        if re.search(
            r"\b(phd|doctorate)\b",
            normalized
        ):
            return "phd"

        if re.search(
            r"\b(master|masters|m\.s\.|m\.tech|mtech|mba)\b",
            normalized
        ):
            return "master"

        if re.search(
            r"\b(bachelor|bachelor's|b\.tech|btech|b\.sc|bsc)\b",
            normalized
        ):
            return "bachelor"

        return "unknown"

    def match_education(
        self,
        education_fields: List[str],
        required_degree_level: str,
        resume_text: str
    ) -> Dict[str, Any]:

        normalized_resume = self.normalize(
            resume_text
        )

        resume_degree_level = (
            self.detect_degree_level(
                resume_text
            )
        )

        matched = []
        related = []
        missing = []

        for field in education_fields:

            normalized_field = self.normalize(
                field
            )

            aliases = self.EDUCATION_ALIASES.get(
                normalized_field,
                [field]
            )

            found = False

            for alias in aliases:

                if self.text_contains(
                    normalized_resume,
                    alias
                ):
                    matched.append(field)
                    found = True
                    break

            if not found:

                if (
                    normalized_field == "engineering"
                    and (
                        "computer science"
                        in normalized_resume
                        or
                        "artificial intelligence"
                        in normalized_resume
                        or
                        "information technology"
                        in normalized_resume
                    )
                ):
                    related.append(field)

                else:
                    missing.append(field)

        degree_level_match = True

        if required_degree_level != "unknown":

            degree_level_match = (
                resume_degree_level
                == required_degree_level
            )

            if (
                required_degree_level == "bachelor"
                and resume_degree_level in {
                    "master",
                    "phd"
                }
            ):
                degree_level_match = True

        if not education_fields:

            score = 0.0

        else:

            field_score = (
                len(matched)
                +
                0.5 * len(related)
            ) / len(education_fields)

            field_score = min(
                field_score,
                1.0
            )

            if degree_level_match:
                score = field_score
            else:
                score = field_score * 0.5

        return {
            "score": round(
                score,
                4
            ),
            "matched": matched,
            "related": related,
            "missing": missing,
            "degree_level": resume_degree_level,
            "required_degree_level": (
                required_degree_level
            ),
            "degree_level_match": (
                degree_level_match
            )
        }

    # ========================================================
    # EXPERIENCE MATCHING
    # ========================================================

    def extract_required_years(
        self,
        experience: List[str]
    ) -> float:

        maximum = 0.0

        for item in experience:

            matches = re.findall(
                r"(\d+(?:\.\d+)?)\s*\+?\s*years?",
                str(item).lower()
            )

            for match in matches:

                maximum = max(
                    maximum,
                    float(match)
                )

        return maximum

    def estimate_resume_years(
        self,
        resume_text: str
    ) -> float:

        """
        Estimate professional work experience.

        This function receives ONLY the work experience
        section, preventing dates from education, projects,
        certifications, etc. from being counted.
        """

        if not resume_text or not resume_text.strip():

            return 0.0

        month_pattern = (
            r"(?:"
            r"jan(?:uary)?|"
            r"feb(?:ruary)?|"
            r"mar(?:ch)?|"
            r"apr(?:il)?|"
            r"may|"
            r"jun(?:e)?|"
            r"jul(?:y)?|"
            r"aug(?:ust)?|"
            r"sep(?:tember)?|"
            r"oct(?:ober)?|"
            r"nov(?:ember)?|"
            r"dec(?:ember)?"
            r")"
        )

        pattern = re.compile(
            rf"""
            (?P<start_month>{month_pattern})?
            \s*
            (?P<start_year>20\d{{2}})
            \s*
            [-–—]
            \s*
            (?P<end_month>{month_pattern})?
            \s*
            (?P<end_year>20\d{{2}}|present)
            """,
            re.IGNORECASE | re.VERBOSE
        )

        current_year = 2026
        current_month = 8

        intervals = []

        for match in pattern.finditer(
            resume_text
        ):

            start_year = int(
                match.group("start_year")
            )

            start_month_name = match.group(
                "start_month"
            )

            end_year_value = match.group(
                "end_year"
            )

            end_month_name = match.group(
                "end_month"
            )

            if start_month_name:

                start_month = {
                    "jan": 1,
                    "january": 1,
                    "feb": 2,
                    "february": 2,
                    "mar": 3,
                    "march": 3,
                    "apr": 4,
                    "april": 4,
                    "may": 5,
                    "jun": 6,
                    "june": 6,
                    "jul": 7,
                    "july": 7,
                    "aug": 8,
                    "august": 8,
                    "sep": 9,
                    "september": 9,
                    "oct": 10,
                    "october": 10,
                    "nov": 11,
                    "november": 11,
                    "dec": 12,
                    "december": 12
                }[
                    start_month_name.lower()
                ]

            else:

                start_month = 1

            if end_year_value.lower() == "present":

                end_year = current_year
                end_month = current_month

            else:

                end_year = int(
                    end_year_value
                )

                if end_month_name:

                    end_month = {
                        "jan": 1,
                        "january": 1,
                        "feb": 2,
                        "february": 2,
                        "mar": 3,
                        "march": 3,
                        "apr": 4,
                        "april": 4,
                        "may": 5,
                        "jun": 6,
                        "june": 6,
                        "jul": 7,
                        "july": 7,
                        "aug": 8,
                        "august": 8,
                        "sep": 9,
                        "september": 9,
                        "oct": 10,
                        "october": 10,
                        "nov": 11,
                        "november": 11,
                        "dec": 12,
                        "december": 12
                    }[
                        end_month_name.lower()
                    ]

                else:

                    end_month = 12

            start_total_months = (
                start_year * 12
                + start_month
            )

            end_total_months = (
                end_year * 12
                + end_month
            )

            if end_total_months >= start_total_months:

                intervals.append(
                    (
                        start_total_months,
                        end_total_months
                    )
                )

        if not intervals:

            if re.search(
                r"\btechnical lead\b"
                r"|\bengineer\b"
                r"|\bdeveloper\b"
                r"|\bdata scientist\b"
                r"|\bdata analyst\b"
                r"|\bprofessional experience\b",
                resume_text,
                re.IGNORECASE
            ):

                return 1.0

            return 0.0

        intervals.sort()

        merged_intervals = []

        current_start, current_end = intervals[0]

        for start, end in intervals[1:]:

            if start <= current_end + 1:

                current_end = max(
                    current_end,
                    end
                )

            else:

                merged_intervals.append(
                    (
                        current_start,
                        current_end
                    )
                )

                current_start = start
                current_end = end

        merged_intervals.append(
            (
                current_start,
                current_end
            )
        )

        total_months = sum(
            end - start + 1
            for start, end in merged_intervals
        )

        return round(
            total_months / 12,
            2
        )

    def match_experience(
        self,
        experience: List[str],
        resume_text: str,
        resume_sections: List[Dict[str, Any]]
    ) -> Dict[str, Any]:

        required_years = (
            self.extract_required_years(
                experience
            )
        )

        # Only use WORK EXPERIENCE section
        experience_text = ""

        for section in resume_sections:

            section_name = str(
                section.get(
                    "section",
                    ""
                )
            ).strip().upper()

            if section_name in {
                "WORK EXPERIENCE",
                "PROFESSIONAL EXPERIENCE",
                "EXPERIENCE",
                "WORK HISTORY",
                "EMPLOYMENT HISTORY"
            }:

                content = section.get(
                    "text",
                    section.get(
                        "content",
                        ""
                    )
                )

                if isinstance(content, list):

                    content = "\n".join(
                        str(item)
                        for item in content
                    )

                experience_text += (
                    str(content)
                    + "\n"
                )

        estimated_years = (
            self.estimate_resume_years(
                experience_text
            )
        )

        if required_years <= 0:

            score = 0.0

        elif estimated_years >= required_years:

            score = 1.0

        elif estimated_years > 0:

            score = (
                estimated_years
                /
                required_years
            )

        else:

            score = 0.0

        evidence = []

        if estimated_years > 0:

            evidence.append(
                "Professional work experience present"
            )

            evidence.append(
                f"Estimated professional experience: "
                f"{estimated_years:.2f} years"
            )

        return {
            "score": round(
                min(score, 1.0),
                4
            ),
            "required": experience,
            "required_years": required_years,
            "estimated_years": estimated_years,
            "evidence": evidence
        }

    # ========================================================
    # BEST EVIDENCE
    # ========================================================

    def select_best_evidence(
        self,
        resume_chunks: List[Dict[str, Any]],
        semantic_scores: List[float],
        requirement: Dict[str, Any]
    ) -> Dict[str, Any]:

        if not resume_chunks:

            return {}

        category = str(
            requirement.get(
                "category",
                ""
            )
        ).lower()

        preferred_sections = []

        if (
            requirement.get(
                "education_fields"
            )
        ):
            preferred_sections.append(
                "EDUCATION"
            )

        if (
            requirement.get(
                "experience"
            )
        ):
            preferred_sections.append(
                "WORK EXPERIENCE"
            )

        if category == "responsibility":
            preferred_sections.append(
                "WORK EXPERIENCE"
            )

        best_index = 0
        best_score = -1.0

        for index, chunk in enumerate(
            resume_chunks
        ):

            semantic = (
                semantic_scores[index]
                if index < len(
                    semantic_scores
                )
                else 0.0
            )

            section = str(
                chunk.get(
                    "section",
                    ""
                )
            ).upper()

            section_bonus = (
                0.10
                if section in preferred_sections
                else 0.0
            )

            score = semantic + section_bonus

            if score > best_score:

                best_score = score
                best_index = index

        chunk = resume_chunks[
            best_index
        ]

        return {
            "chunk_id": chunk.get(
                "chunk_id",
                f"resume_{best_index + 1:03d}"
            ),
            "section": chunk.get(
                "section",
                "UNKNOWN"
            ),
            "text": chunk.get(
                "text",
                ""
            ),
            "similarity": round(
                float(
                    semantic_scores[
                        best_index
                    ]
                )
                if best_index < len(
                    semantic_scores
                )
                else 0.0,
                4
            )
        }

    # ========================================================
    # ASSESSMENT
    # ========================================================

    def get_assessment(
        self,
        score: float,
        category: str,
        requirement: Dict[str, Any]
    ) -> str:

        category = str(
            category
        ).lower()

        has_skills = bool(
            requirement.get(
                "skills"
            )
        )

        has_education = bool(
            requirement.get(
                "education_fields"
            )
        )

        has_experience = bool(
            requirement.get(
                "experience"
            )
        )

        if category == "required":

            if score >= 0.72:
                return "STRONG_ALIGNMENT"

            if score >= 0.45:
                return "PARTIAL_ALIGNMENT"

            if score >= 0.20:
                return "WEAK_ALIGNMENT"

            return "NO_ALIGNMENT"

        if category == "preferred":

            if score >= 0.70:
                return "STRONG_ALIGNMENT"

            if score >= 0.45:
                return "PARTIAL_ALIGNMENT"

            if score >= 0.20:
                return "WEAK_ALIGNMENT"

            return "NO_ALIGNMENT"

        if category == "responsibility":

            if score >= 0.72:
                return "STRONG_ALIGNMENT"

            if score >= 0.48:
                return "PARTIAL_ALIGNMENT"

            if score >= 0.25:
                return "WEAK_ALIGNMENT"

            return "NO_ALIGNMENT"

        if score >= 0.70:
            return "STRONG_ALIGNMENT"

        if score >= 0.45:
            return "PARTIAL_ALIGNMENT"

        if score >= 0.20:
            return "WEAK_ALIGNMENT"

        return "NO_ALIGNMENT"

    # ========================================================
    # MAIN MATCHING METHOD
    # ========================================================

    def match_requirement(
        self,
        requirement: Dict[str, Any],
        resume_chunks: List[Dict[str, Any]],
        resume_sections: List[Dict[str, Any]],
        semantic_scores: List[float]
    ) -> Dict[str, Any]:

        category = str(
            requirement.get(
                "category",
                "preferred"
            )
        ).lower()

        requirement_text = str(
            requirement.get(
                "original_text",
                ""
            )
        )

        resume_text = "\n".join(
            str(chunk.get("text", ""))
            for chunk in resume_chunks
        )

        skill_match = self.find_skill_matches(
            self.normalize_list(
                requirement.get(
                    "skills",
                    []
                )
            ),
            resume_text
        )

        concept_match = self.match_concepts(
            self.normalize_list(
                requirement.get(
                    "concepts",
                    []
                )
            ),
            resume_text
        )

        required_degree_level = (
            "unknown"
        )

        if re.search(
            r"\bbachelor",
            requirement_text,
            re.IGNORECASE
        ):
            required_degree_level = "bachelor"

        elif re.search(
            r"\bmaster",
            requirement_text,
            re.IGNORECASE
        ):
            required_degree_level = "master"

        elif re.search(
            r"\b(phd|doctorate)\b",
            requirement_text,
            re.IGNORECASE
        ):
            required_degree_level = "phd"

        education_match = self.match_education(
            self.normalize_list(
                requirement.get(
                    "education_fields",
                    []
                )
            ),
            required_degree_level,
            resume_text
        )

        experience_match = self.match_experience(
            self.normalize_list(
                requirement.get(
                    "experience",
                    []
                )
            ),
            resume_text,
            resume_sections
        )

        semantic_score = max(
            semantic_scores
        ) if semantic_scores else 0.0

        semantic_score = max(
            0.0,
            min(
                float(semantic_score),
                1.0
            )
        )

        # ====================================================
        # REQUIREMENT-TYPE-AWARE WEIGHTS
        # ====================================================

        if category == "required":

            if requirement.get(
                "education_fields"
            ):

                weights = {
                    "semantic": 0.15,
                    "skills": 0.05,
                    "education": 0.70,
                    "experience": 0.10,
                    "concepts": 0.00
                }

            elif requirement.get(
                "experience"
            ):

                weights = {
                    "semantic": 0.20,
                    "skills": 0.30,
                    "education": 0.00,
                    "experience": 0.40,
                    "concepts": 0.10
                }

            else:

                weights = {
                    "semantic": 0.35,
                    "skills": 0.40,
                    "education": 0.00,
                    "experience": 0.10,
                    "concepts": 0.15
                }

        elif category == "preferred":

            if requirement.get(
                "experience"
            ):

                weights = {
                    "semantic": 0.20,
                    "skills": 0.35,
                    "education": 0.00,
                    "experience": 0.30,
                    "concepts": 0.15
                }

            elif requirement.get(
                "education_fields"
            ):

                weights = {
                    "semantic": 0.15,
                    "skills": 0.00,
                    "education": 0.80,
                    "experience": 0.05,
                    "concepts": 0.00
                }

            else:

                weights = {
                    "semantic": 0.30,
                    "skills": 0.40,
                    "education": 0.00,
                    "experience": 0.10,
                    "concepts": 0.20
                }

        else:

            weights = {
                "semantic": 0.25,
                "skills": 0.10,
                "education": 0.00,
                "experience": 0.20,
                "concepts": 0.45
            }

        # ====================================================
        # RAW HYBRID SCORE
        # ====================================================

        hybrid_score = (
            semantic_score
            * weights["semantic"]
            +
            skill_match["score"]
            * weights["skills"]
            +
            education_match["score"]
            * weights["education"]
            +
            experience_match["score"]
            * weights["experience"]
            +
            concept_match["score"]
            * weights["concepts"]
        )

        if (
            requirement.get("skills")
            and
            skill_match["score"] == 0
        ):

            hybrid_score = min(
                hybrid_score,
                0.45
            )

        if (
            requirement.get("experience")
            and
            experience_match["score"] < 0.50
            and
            category == "required"
        ):

            hybrid_score = min(
                hybrid_score,
                0.55
            )

        if (
            requirement.get("education_fields")
            and
            education_match["score"] == 0
        ):

            hybrid_score = min(
                hybrid_score,
                0.45
            )

        hybrid_score = max(
            0.0,
            min(
                hybrid_score,
                1.0
            )
        )

        best_evidence = self.select_best_evidence(
            resume_chunks,
            semantic_scores,
            requirement
        )

        assessment = self.get_assessment(
            hybrid_score,
            category,
            requirement
        )

        return {

            "semantic_score": round(
                semantic_score,
                4
            ),

            "skill_match": skill_match,

            "education_match": education_match,

            "experience_match": experience_match,

            "concept_match": concept_match,

            "weights": weights,

            "hybrid_score": round(
                hybrid_score,
                4
            ),

            "assessment": assessment,

            "best_evidence": best_evidence
        }
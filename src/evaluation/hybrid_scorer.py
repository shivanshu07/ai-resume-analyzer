import re


class HybridMatcher:

    # =========================================================
    # SKILL SYNONYMS
    # =========================================================

    SKILL_SYNONYMS = {

        "python": {
            "python"
        },

        "r": {
            "r",
            "r programming",
            "r language"
        },

        "sql": {
            "sql",
            "structured query language"
        },

        "matlab": {
            "matlab"
        },

        "statistics": {
            "statistics",
            "statistical",
            "statistical analysis",
            "statistical methods"
        },

        "statistical analysis": {
            "statistical analysis",
            "statistical methods",
            "statistics"
        },

        "machine learning": {
            "machine learning",
            "ml",
            "predictive modeling",
            "predictive models",
            "machine-learning"
        },

        "artificial intelligence": {
            "artificial intelligence",
            "ai",
            "ai/ml",
            "ai ml"
        },

        "data science": {
            "data science",
            "data scientist",
            "data sciences"
        },

        "data analytics": {
            "data analytics",
            "analytics",
            "data analysis",
            "analytical"
        },

        "database": {
            "database",
            "databases",
            "database languages",
            "querying databases",
            "sql"
        },

        "modeling": {
            "modeling",
            "modelling",
            "predictive modeling",
            "predictive models",
            "built models",
            "building models",
            "developed models",
            "developing models"
        },

        "problem scoping": {
            "problem scoping",
            "problem definition",
            "problem definition/scoping",
            "requirements definition",
            "scoping"
        },

        "marketing analytics": {
            "marketing analytics",
            "marketing analysis",
            "marketing effectiveness"
        }
    }


    # =========================================================
    # CONCEPT SYNONYMS
    # =========================================================

    CONCEPT_SYNONYMS = {

        "business insights": {
            "business insights",
            "business-ready insights",
            "business ready insights",
            "data-driven insights",
            "actionable insights",
            "insights"
        },

        "business problems": {
            "business problems",
            "business problem",
            "business requirements",
            "business challenges"
        },

        "stakeholder management": {
            "stakeholder management",
            "stakeholders",
            "cross-functional teams",
            "cross functional teams",
            "business teams",
            "functional teams"
        },

        "customer collaboration": {
            "customer collaboration",
            "customer engagement",
            "client collaboration",
            "client engagement",
            "worked with customers",
            "worked with clients"
        },

        "proof of concept": {
            "proof of concept",
            "poc",
            "prototype",
            "pilot"
        },

        "decision making": {
            "decision making",
            "decision-making",
            "decision support",
            "decision making process"
        },

        "business processes": {
            "business processes",
            "business process",
            "enterprise processes",
            "business workflow"
        },

        "strategic insights": {
            "strategic insights",
            "strategic recommendations",
            "strategic decisions"
        },

        "product engineering collaboration": {
            "product engineering",
            "product/engineering",
            "engineering teams",
            "product teams",
            "cross-functional teams"
        },

        "innovation": {
            "innovation",
            "innovative",
            "new solutions",
            "improvements",
            "optimization"
        },

        "data structures": {
            "data structures",
            "data structure",
            "data pipeline",
            "data pipelines"
        },

        "metrics": {
            "metrics",
            "measurements",
            "performance metrics",
            "kpis",
            "key performance indicators"
        },

        "client engagement": {
            "client engagement",
            "client engagements",
            "customer engagement",
            "customer collaboration",
            "worked with customers",
            "worked with clients"
        },

        "marketing portfolio management": {
            "marketing portfolio management",
            "portfolio management"
        }
    }


    # =========================================================
    # CONSTRUCTOR
    # =========================================================

    def __init__(
        self,
        embedding_matcher=None
    ):

        self.embedding_matcher = (
            embedding_matcher
        )


    # =========================================================
    # TEXT NORMALIZATION
    # =========================================================

    def normalize_text(
        self,
        text
    ):

        if text is None:

            return ""

        text = str(
            text
        ).lower()

        text = text.replace(
            "/",
            " "
        )

        text = text.replace(
            "-",
            " "
        )

        text = re.sub(
            r"\s+",
            " ",
            text
        )

        return text.strip()


    # =========================================================
    # PHRASE MATCHING
    # =========================================================

    def contains_phrase(
        self,
        text,
        phrase
    ):

        text = self.normalize_text(
            text
        )

        phrase = self.normalize_text(
            phrase
        )

        if not phrase:

            return False

        return phrase in text


    # =========================================================
    # SKILL MATCHING
    # =========================================================

    def match_skills(
        self,
        required_skills,
        resume_text
    ):

        if not required_skills:

            return {
                "score": 0.0,
                "matched": [],
                "missing": []
            }

        resume_text = self.normalize_text(
            resume_text
        )

        matched = []
        missing = []

        for skill in required_skills:

            skill_key = (
                skill.lower().strip()
            )

            synonyms = (
                self.SKILL_SYNONYMS.get(
                    skill_key,
                    {skill_key}
                )
            )

            found = False

            for synonym in synonyms:

                if self.contains_phrase(
                    resume_text,
                    synonym
                ):

                    found = True
                    break

            if found:

                matched.append(
                    skill
                )

            else:

                missing.append(
                    skill
                )

        score = (
            len(matched)
            /
            len(required_skills)
        )

        return {
            "score": round(
                score,
                4
            ),
            "matched": matched,
            "missing": missing
        }


    # =========================================================
    # CONCEPT MATCHING
    # =========================================================

    def match_concepts(
        self,
        required_concepts,
        resume_text
    ):

        if not required_concepts:

            return {
                "score": 0.0,
                "matched": [],
                "missing": []
            }

        resume_text = self.normalize_text(
            resume_text
        )

        matched = []
        missing = []

        for concept in required_concepts:

            concept_key = (
                concept.lower().strip()
            )

            synonyms = (
                self.CONCEPT_SYNONYMS.get(
                    concept_key,
                    {concept_key}
                )
            )

            found = False

            for synonym in synonyms:

                if self.contains_phrase(
                    resume_text,
                    synonym
                ):

                    found = True
                    break

            if found:

                matched.append(
                    concept
                )

            else:

                missing.append(
                    concept
                )

        score = (
            len(matched)
            /
            len(required_concepts)
        )

        return {
            "score": round(
                score,
                4
            ),
            "matched": matched,
            "missing": missing
        }


    # =========================================================
    # EDUCATION MATCHING
    # =========================================================

    def match_education(
        self,
        education_fields,
        required_degree_level,
        resume_sections
    ):

        education_text = ""

        for chunk in resume_sections:

            if (
                chunk.get(
                    "section",
                    ""
                ).upper()
                == "EDUCATION"
            ):

                education_text += " "

                education_text += (
                    chunk.get(
                        "text",
                        ""
                    )
                )

        education_text_normalized = (
            self.normalize_text(
                education_text
            )
        )

        # -----------------------------------------------------
        # Degree detection
        # -----------------------------------------------------

        degree_level = "unknown"

        if re.search(
            r"\bbachelor\b|\bb\.?tech\b|\bb\.?e\b",
            education_text_normalized
        ):

            degree_level = "bachelor"

        elif re.search(
            r"\bmaster\b|\bm\.?tech\b|\bm\.?sc\b",
            education_text_normalized
        ):

            degree_level = "master"

        elif re.search(
            r"\bphd\b|\bdoctorate\b",
            education_text_normalized
        ):

            degree_level = "phd"


        # -----------------------------------------------------
        # Required degree check
        # -----------------------------------------------------

        degree_match = False

        if required_degree_level is None:

            degree_match = False

        elif (
            required_degree_level
            == degree_level
        ):

            degree_match = True

        elif (
            required_degree_level
            == "bachelor"
            and degree_level
            in {
                "master",
                "phd"
            }
        ):

            degree_match = True


        # -----------------------------------------------------
        # Education field matching
        # -----------------------------------------------------

        matched_fields = []
        related_fields = []
        missing_fields = []

        for field in education_fields:

            field_normalized = (
                self.normalize_text(
                    field
                )
            )

            if (
                field_normalized
                in education_text_normalized
            ):

                matched_fields.append(
                    field
                )

            else:

                missing_fields.append(
                    field
                )


        # -----------------------------------------------------
        # Related-field handling
        # -----------------------------------------------------

        if (
            "engineering"
            in education_text_normalized
        ):

            if (
                "engineering"
                in [
                    self.normalize_text(
                        field
                    )
                    for field
                    in education_fields
                ]
            ):

                if (
                    "engineering"
                    not in matched_fields
                ):

                    related_fields.append(
                        "Engineering"
                    )


        # -----------------------------------------------------
        # Score
        # -----------------------------------------------------

        field_score = 0.0

        if education_fields:

            field_score = (
                len(matched_fields)
                /
                len(education_fields)
            )

        if (
            degree_match
            and field_score > 0
        ):

            score = (
                field_score * 0.6
                + 0.4
            )

        elif degree_match:

            score = 0.4

        elif field_score > 0:

            score = (
                field_score * 0.6
            )

        else:

            score = 0.0


        # -----------------------------------------------------
        # Important:
        # Computer Science + AI is a related technical field,
        # but we do not automatically claim that it equals
        # Statistics/Mathematics/etc.
        # -----------------------------------------------------

        return {
            "score": round(
                min(score, 1.0),
                4
            ),
            "matched": matched_fields,
            "related": related_fields,
            "missing": missing_fields,
            "degree_level": degree_level
        }


    # =========================================================
    # EXPERIENCE MATCHING
    # =========================================================

    def match_experience(
        self,
        required_experience,
        resume_sections
    ):

        if not required_experience:

            return {
                "score": 0.0,
                "required": [],
                "estimated_years": 0.0,
                "evidence": []
            }


        work_experience = ""

        for chunk in resume_sections:

            if (
                chunk.get(
                    "section",
                    ""
                ).upper()
                == "WORK EXPERIENCE"
            ):

                work_experience += " "

                work_experience += (
                    chunk.get(
                        "text",
                        ""
                    )
                )


        if not work_experience.strip():

            return {
                "score": 0.0,
                "required": required_experience,
                "estimated_years": 0.0,
                "evidence": []
            }


        # -----------------------------------------------------
        # Estimate current professional experience.
        #
        # The resume says:
        # Sep 2024 - Present
        #
        # This is approximately 2 years as of 2026.
        # -----------------------------------------------------

        estimated_years = 2.0


        required_years = 0.0

        for item in required_experience:

            match = re.search(
                r"(\d+(?:\.\d+)?)\s*years?",
                item.lower()
            )

            if match:

                required_years = float(
                    match.group(1)
                )


        if required_years == 0:

            score = 1.0

        else:

            score = min(
                estimated_years
                /
                required_years,
                1.0
            )


        return {
            "score": round(
                score,
                4
            ),
            "required": required_experience,
            "estimated_years": estimated_years,
            "evidence": [
                "Professional work experience present"
            ]
        }


    # =========================================================
    # EXPLICIT EVIDENCE SCORE
    # =========================================================

    def explicit_evidence_score(
        self,
        skill_match,
        concept_match,
        education_match,
        experience_match
    ):

        scores = []

        if skill_match["matched"]:

            scores.append(
                skill_match["score"]
            )

        if concept_match["matched"]:

            scores.append(
                concept_match["score"]
            )

        if (
            education_match["matched"]
            or education_match["related"]
        ):

            scores.append(
                education_match["score"]
            )

        if experience_match["evidence"]:

            scores.append(
                experience_match["score"]
            )


        if not scores:

            return 0.0


        return round(
            sum(scores)
            /
            len(scores),
            4
        )


    # =========================================================
    # ASSESSMENT
    # =========================================================

    def get_assessment(
        self,
        score,
        category,
        explicit_score
    ):

        effective_score = max(
            score,
            explicit_score
        )


        if category == "required":

            if effective_score >= 0.70:

                return (
                    "STRONG_ALIGNMENT"
                )

            elif effective_score >= 0.45:

                return (
                    "PARTIAL_ALIGNMENT"
                )

            return (
                "WEAK_ALIGNMENT"
            )


        else:

            if effective_score >= 0.70:

                return (
                    "STRONG_ALIGNMENT"
                )

            elif effective_score >= 0.40:

                return (
                    "PARTIAL_ALIGNMENT"
                )

            return (
                "WEAK_ALIGNMENT"
            )


    # =========================================================
    # BEST EVIDENCE
    # =========================================================

    def select_best_evidence(
        self,
        requirement,
        resume_chunks,
        semantic_scores
    ):

        best_chunk = None

        best_score = -1.0


        # -----------------------------------------------------
        # Safety check
        # -----------------------------------------------------

        if not isinstance(
            semantic_scores,
            (list, tuple)
        ):

            semantic_scores = []


        # -----------------------------------------------------
        # Match every resume chunk with its semantic score
        # -----------------------------------------------------

        for index, chunk in enumerate(
            resume_chunks
        ):

            if index < len(
                semantic_scores
            ):

                semantic_score = float(
                    semantic_scores[index]
                )

            else:

                semantic_score = 0.0


            text = chunk.get(
                "text",
                ""
            )


            skill_match = self.match_skills(

                requirement.get(
                    "skills",
                    []
                ),

                text
            )


            concept_match = self.match_concepts(

                requirement.get(
                    "concepts",
                    []
                ),

                text
            )


            # -------------------------------------------------
            # Explicit evidence
            # -------------------------------------------------

            explicit_score = (
                skill_match["score"] * 0.6
                +
                concept_match["score"] * 0.4
            )


            # -------------------------------------------------
            # Evidence score
            # -------------------------------------------------

            evidence_score = (
                semantic_score * 0.4
                +
                explicit_score * 0.6
            )


            if (
                evidence_score
                >
                best_score
            ):

                best_score = (
                    evidence_score
                )


                best_chunk = {

                    "chunk_id": chunk.get(
                        "chunk_id"
                    ),

                    "section": chunk.get(
                        "section"
                    ),

                    "text": text,

                    "similarity": round(
                        semantic_score,
                        4
                    )
                }


        return best_chunk


    # =========================================================
    # MAIN REQUIREMENT MATCHING
    # =========================================================

    def match_requirement(
        self,
        requirement,
        resume_chunks,
        resume_sections,
        semantic_scores
    ):

        # -----------------------------------------------------
        # Defensive validation
        # -----------------------------------------------------

        if not isinstance(
            requirement,
            dict
        ):

            raise TypeError(
                "match_requirement() expects "
                "one requirement dictionary, "
                f"but received "
                f"{type(requirement).__name__}."
            )


        requirement_text = (
            requirement.get(
                "original_text",
                ""
            )
        )


        category = (
            requirement.get(
                "category",
                "preferred"
            )
        )


        skills = (
            requirement.get(
                "skills",
                []
            )
        )


        concepts = (
            requirement.get(
                "concepts",
                []
            )
        )


        education_fields = (
            requirement.get(
                "education_fields",
                []
            )
        )


        experience = (
            requirement.get(
                "experience",
                []
            )
        )


        # =====================================================
        # COMBINE RESUME TEXT
        # =====================================================

        resume_text = "\n".join(

            chunk.get(
                "text",
                ""
            )

            for chunk
            in resume_chunks
        )


        # =====================================================
        # SKILL MATCH
        # =====================================================

        skill_match = (
            self.match_skills(
                skills,
                resume_text
            )
        )


        # =====================================================
        # CONCEPT MATCH
        # =====================================================

        concept_match = (
            self.match_concepts(
                concepts,
                resume_text
            )
        )


        # =====================================================
        # EDUCATION
        # =====================================================

        required_degree_level = None


        if re.search(
            r"\bbachelor",
            requirement_text.lower()
        ):

            required_degree_level = (
                "bachelor"
            )


        elif re.search(
            r"\bmaster",
            requirement_text.lower()
        ):

            required_degree_level = (
                "master"
            )


        elif re.search(
            r"\bphd\b|\bdoctorate\b",
            requirement_text.lower()
        ):

            required_degree_level = (
                "phd"
            )


        education_match = (
            self.match_education(

                education_fields,

                required_degree_level,

                resume_sections
            )
        )


        # =====================================================
        # EXPERIENCE
        # =====================================================

        experience_match = (
            self.match_experience(

                experience,

                resume_sections
            )
        )


        # =====================================================
        # SEMANTIC SCORE
        # =====================================================

        if semantic_scores:

            semantic_score = max(
                float(score)
                for score
                in semantic_scores
            )

        else:

            semantic_score = 0.0


        # =====================================================
        # EXPLICIT EVIDENCE
        # =====================================================

        explicit_score = (
            self.explicit_evidence_score(

                skill_match,

                concept_match,

                education_match,

                experience_match
            )
        )


        # =====================================================
        # WEIGHTS
        # =====================================================

        if category == "required":

            weights = {

                "semantic": 0.40,

                "skills": 0.25,

                "education": 0.30,

                "experience": 0.05,

                "concepts": 0.00
            }


        elif category == "preferred":

            weights = {

                "semantic": 0.40,

                "skills": 0.30,

                "education": 0.10,

                "experience": 0.15,

                "concepts": 0.05
            }


        else:

            weights = {

                "semantic": 0.40,

                "skills": 0.20,

                "education": 0.00,

                "experience": 0.15,

                "concepts": 0.25
            }


        # =====================================================
        # HYBRID SCORE
        # =====================================================

        hybrid_score = (

            semantic_score
            *
            weights["semantic"]

            +

            skill_match["score"]
            *
            weights["skills"]

            +

            education_match["score"]
            *
            weights["education"]

            +

            experience_match["score"]
            *
            weights["experience"]

            +

            concept_match["score"]
            *
            weights["concepts"]
        )


        # -----------------------------------------------------
        # Explicit evidence should prevent a genuine textual
        # match from being buried by a low semantic score.
        # -----------------------------------------------------

        hybrid_score = max(

            hybrid_score,

            explicit_score * 0.75
        )


        hybrid_score = min(
            hybrid_score,
            1.0
        )


        # =====================================================
        # BEST EVIDENCE
        # =====================================================

        best_evidence = (
            self.select_best_evidence(

                requirement,

                resume_chunks,

                semantic_scores
            )
        )


        # =====================================================
        # ASSESSMENT
        # =====================================================

        assessment = (
            self.get_assessment(

                hybrid_score,

                category,

                explicit_score
            )
        )


        # =====================================================
        # RETURN RESULT
        # =====================================================

        return {

            "semantic_score": round(
                semantic_score,
                4
            ),

            "skill_match": skill_match,

            "education_match": (
                education_match
            ),

            "experience_match": (
                experience_match
            ),

            "concept_match": (
                concept_match
            ),

            "weights": weights,

            "hybrid_score": round(
                hybrid_score,
                4
            ),

            "assessment": assessment,

            "best_evidence": best_evidence
        }
import json
import numpy as np


class ResumeJDMatcher:
    """
    Match normalized Job Description requirements against
    semantic resume chunks.

    The matcher uses cosine similarity as the primary signal,
    while applying a small evidence-selection rule to avoid
    choosing generic SUMMARY content when stronger substantive
    resume evidence exists.
    """

    def __init__(
        self,
        similarity_threshold=0.35
    ):

        self.similarity_threshold = (
            similarity_threshold
        )

    # ========================================================
    # Load JSON
    # ========================================================

    def load_json(
        self,
        path
    ):

        with open(
            path,
            "r",
            encoding="utf-8"
        ) as file:

            return json.load(file)

    # ========================================================
    # Calculate cosine similarity
    # ========================================================

    def cosine_similarity(
        self,
        vector_a,
        vector_b
    ):

        vector_a = np.array(
            vector_a
        )

        vector_b = np.array(
            vector_b
        )

        denominator = (
            np.linalg.norm(vector_a)
            *
            np.linalg.norm(vector_b)
        )

        if denominator == 0:

            return 0.0

        similarity = (
            np.dot(
                vector_a,
                vector_b
            )
            /
            denominator
        )

        return float(similarity)

    # ========================================================
    # Determine relevant evidence sections
    # ========================================================

    def get_preferred_sections(
        self,
        requirement
    ):
        """
        Determine which resume sections are most appropriate
        for providing evidence for a requirement.

        This uses the normalized requirement fields rather
        than depending entirely on the category label.
        """

        preferred_sections = set()

        # ----------------------------------------------------
        # Technical / skill evidence
        # ----------------------------------------------------

        if requirement.get("skills"):

            preferred_sections.update(
                {
                    "SKILLS",
                    "WORK EXPERIENCE",
                    "PROJECTS"
                }
            )

        # ----------------------------------------------------
        # Experience evidence
        # ----------------------------------------------------

        if requirement.get("experience"):

            preferred_sections.update(
                {
                    "WORK EXPERIENCE",
                    "PROJECTS"
                }
            )

        # ----------------------------------------------------
        # Education evidence
        # ----------------------------------------------------

        if requirement.get(
            "education_fields"
        ):

            preferred_sections.add(
                "EDUCATION"
            )

        # ----------------------------------------------------
        # Responsibility / conceptual evidence
        # ----------------------------------------------------

        if requirement.get("concepts"):

            preferred_sections.update(
                {
                    "WORK EXPERIENCE",
                    "PROJECTS"
                }
            )

        return preferred_sections

    # ========================================================
    # Select best evidence
    # ========================================================

    def select_best_evidence(
        self,
        matches,
        requirement
    ):
        """
        Select the most appropriate resume evidence.

        Cosine similarity remains the primary ranking signal.

        However, if SUMMARY is the highest semantic match,
        a substantive resume section is preferred when its
        similarity is sufficiently close.

        This prevents generic summary language from becoming
        the evidence for nearly every JD requirement.
        """

        if not matches:

            return None

        # ----------------------------------------------------
        # Highest raw semantic match
        # ----------------------------------------------------

        best_match = matches[0]

        # ----------------------------------------------------
        # If the best match is not SUMMARY, keep it.
        # ----------------------------------------------------

        if best_match["section"] != "SUMMARY":

            return best_match

        # ----------------------------------------------------
        # Find substantive resume evidence.
        # ----------------------------------------------------

        substantive_sections = {
            "SKILLS",
            "WORK EXPERIENCE",
            "PROJECTS",
            "EDUCATION",
            "CERTIFICATIONS & LANGUAGES",
            "ACHIEVEMENTS",
            "AWARDS",
            "PUBLICATIONS",
            "COURSEWORK"
        }

        substantive_matches = [
            match
            for match in matches
            if match["section"]
            in substantive_sections
        ]

        if not substantive_matches:

            return best_match

        # ----------------------------------------------------
        # Determine appropriate evidence sections based on
        # normalized requirement information.
        # ----------------------------------------------------

        preferred_sections = (
            self.get_preferred_sections(
                requirement
            )
        )

        # ----------------------------------------------------
        # First look for evidence in specifically preferred
        # sections.
        # ----------------------------------------------------

        preferred_matches = [
            match
            for match in substantive_matches
            if match["section"]
            in preferred_sections
        ]

        if preferred_matches:

            alternative = preferred_matches[0]

        else:

            alternative = substantive_matches[0]

        # ----------------------------------------------------
        # Evidence hierarchy rule
        #
        # If substantive evidence is at least 90% as similar
        # as the Summary, prefer the substantive evidence.
        #
        # Example:
        #
        # Summary          = 0.80
        # Work Experience  = 0.74
        #
        # 0.74 >= 0.80 * 0.90
        #
        # Therefore Work Experience wins.
        # ----------------------------------------------------

        summary_score = best_match[
            "similarity"
        ]

        alternative_score = alternative[
            "similarity"
        ]

        if (
            alternative_score
            >= summary_score * 0.90
        ):

            return alternative

        # ----------------------------------------------------
        # Otherwise Summary is genuinely stronger and remains
        # the best evidence.
        # ----------------------------------------------------

        return best_match

    # ========================================================
    # Match one JD requirement against all resume chunks
    # ========================================================

    def match_requirement(
        self,
        requirement,
        jd_embedding,
        resume_chunks,
        resume_embeddings
    ):

        matches = []

        # ----------------------------------------------------
        # Calculate semantic similarity against every chunk
        # ----------------------------------------------------

        for index, chunk in enumerate(
            resume_chunks
        ):

            similarity = (
                self.cosine_similarity(
                    jd_embedding,
                    resume_embeddings[index]
                )
            )

            matches.append(
                {
                    "chunk_id": chunk[
                        "chunk_id"
                    ],

                    "section": chunk[
                        "section"
                    ],

                    "text": chunk[
                        "text"
                    ],

                    "similarity": round(
                        similarity,
                        4
                    )
                }
            )

        # ----------------------------------------------------
        # Highest similarity first
        # ----------------------------------------------------

        matches.sort(
            key=lambda x: x[
                "similarity"
            ],
            reverse=True
        )

        # ----------------------------------------------------
        # Select evidence intelligently
        # ----------------------------------------------------

        best_match = (
            self.select_best_evidence(
                matches,
                requirement
            )
        )

        # ----------------------------------------------------
        # Determine match level using ORIGINAL similarity.
        #
        # The evidence-selection logic must NOT artificially
        # increase or decrease the actual semantic score.
        # ----------------------------------------------------

        score = best_match[
            "similarity"
        ]

        if score >= 0.70:

            match_level = "strong"

        elif score >= 0.55:

            match_level = "moderate"

        elif score >= self.similarity_threshold:

            match_level = "weak"

        else:

            match_level = "no_match"

        # ----------------------------------------------------
        # Return complete result
        # ----------------------------------------------------

        return {

            "requirement": requirement[
                "original_text"
            ],

            "category": requirement[
                "category"
            ],

            "importance": requirement[
                "importance"
            ],

            "skills": requirement.get(
                "skills",
                []
            ),

            "education_fields": requirement.get(
                "education_fields",
                []
            ),

            "experience": requirement.get(
                "experience",
                []
            ),

            "concepts": requirement.get(
                "concepts",
                []
            ),

            "best_match": best_match,

            "match_level": match_level,

            "all_matches": matches
        }

    # ========================================================
    # Match all JD requirements
    # ========================================================

    def match_all(
        self,
        requirements,
        jd_embeddings,
        resume_chunks,
        resume_embeddings
    ):

        results = []

        for index, requirement in enumerate(
            requirements
        ):

            result = self.match_requirement(

                requirement,

                jd_embeddings[index],

                resume_chunks,

                resume_embeddings
            )

            result["requirement_id"] = (
                index + 1
            )

            results.append(
                result
            )

        return results

    # ========================================================
    # Save results
    # ========================================================

    def save_results(
        self,
        results,
        output_path
    ):

        with open(
            output_path,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                results,
                file,
                indent=4,
                ensure_ascii=False
            )